from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from common.environments.qa_retrieval_env import (
    QAEnvironment,
    QARunner,
    _clean_visible_feedback_text,
    _format_ranked_search_results,
    parse_action,
    parse_expected_answer,
    retrieval_quality,
)
from common.environments.search_utils import (
    DocumentSection,
    DocumentSectionHit,
    QuestionBankHit,
    QuestionBankItem,
    SearchChunk,
    SearchHit,
    SimpleSearchEngine,
    clean_query,
    clean_retrieval_snippet,
    count_evidence_score,
    format_hits_with_refs,
    format_structured_read_context,
    is_quantity_query,
    proper_term_match_count_from_terms,
    split_search_intents,
)
from common.rewards.qa_reward import qa_rule_reward_fn


class _FakeTensor:
    def __init__(self, text: str):
        self.text = text

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return len(self.text)


class _FakeTokenizer:
    def __init__(self):
        self.last_tokenized = ""

    def apply_chat_template(self, messages, **_kwargs):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, **_kwargs):
        self.last_tokenized = text
        return {"input_ids": _FakeTensor(text)}


def _load_agent_run_module():
    def module(name: str, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    module("omegaconf", OmegaConf=object)
    module("torch")
    module("torch.utils")
    module("torch.utils.data", IterableDataset=object)
    module("nemo_rl")
    module("nemo_rl.algorithms")
    module("nemo_rl.algorithms.grpo", MasterConfig=dict, grpo_train=lambda *a, **k: None, setup=lambda *a, **k: None)
    module("nemo_rl.algorithms.utils", get_tokenizer=lambda *a, **k: None, set_seed=lambda *a, **k: None)
    module("nemo_rl.data")
    module("nemo_rl.data.interfaces", DatumSpec=dict, LLMMessageLogType=list)
    module("nemo_rl.distributed")
    module("nemo_rl.distributed.virtual_cluster", init_ray=lambda *a, **k: None)
    module("nemo_rl.models")
    module("nemo_rl.models.generation", configure_generation_config=lambda *a, **k: None)
    module(
        "nemo_rl.utils.config",
        load_config=lambda *a, **k: {},
        parse_hydra_overrides=lambda c, o: c,
        register_omegaconf_resolvers=lambda *a, **k: None,
    )
    module("nemo_rl.utils.logger", get_next_experiment_dir=lambda p: p)

    path = Path(__file__).parents[1] / "experiments" / "grpo_qwen3.5-9b_qa-rl-agent_v1" / "run.py"
    spec = importlib.util.spec_from_file_location("qa_agent_run_for_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _first_question_bank_id(metadata: dict) -> int:
    return next(int(ref["id"]) for ref in metadata["last_search_hits"] if ref.get("question_bank"))


def test_parse_expected_answer():
    parsed = parse_expected_answer("[multiple] A,B,C")
    assert parsed.answer_type == "multiple"
    assert parsed.gold_answer == "A,B,C"


def test_agent_datum_uses_raw_prompt_by_default():
    mod = _load_agent_run_module()
    tokenizer = _FakeTokenizer()
    datum = mod.generate_datum(
        tokenizer,
        {"use_few_shot": False, "use_chat_template": False, "max_turns": 3},
        {"query": "题目：OFD格式电子发票必须提供OFD格式源文件。", "expected_answer": "[bool] A"},
        0,
        split="validation",
    )
    content = datum["message_log"][0]["content"]
    assert "<|im_start|>" not in content
    assert "<|im_start|>" not in tokenizer.last_tokenized
    assert "真实题目如下" in content


def test_agent_default_few_shot_uses_safe_protocol_examples():
    mod = _load_agent_run_module()
    prompt = mod.build_prompt(
        "题目：OFD格式电子发票必须提供OFD格式源文件。",
        {},
    )
    few_shot = prompt.split("Few-shot 协议示范", 1)[1].split("真实题目如下", 1)[0]
    assert "不能照抄" in few_shot
    assert "<search>Y事件 应急处置</search>" in few_shot
    assert "<search>X系统 Y 创建方式</search>" in few_shot
    assert "<read>1</read>" in few_shot
    assert "<think>" in few_shot
    assert "</think>" in few_shot
    assert r"\boxed{Z}" in few_shot
    assert "普通技术文档也可能藏着答案" in prompt
    assert "不要把所有选项词机械拼进 search" in prompt
    assert "优先保留专有名词" in prompt
    assert "by pass / by-pass 应合并为 bypass" in prompt
    assert "换词" in prompt
    assert "同义词" in prompt
    assert "合写英文术语" in prompt
    assert "错误说法、正确说法、规范" in prompt
    assert "检索结果像目录" in prompt
    assert "安全卫生、行为规范、明显常识题" in prompt
    assert "不要放弃作答" in prompt
    assert "多选题不要只凭一个命中词直接作答" in prompt
    assert "OFD" not in few_shot
    assert "<search>bypass 安全设施 处罚 绩效</search>" not in few_shot
    assert "安全卫生常识" not in few_shot
    assert r"\boxed{A}" not in few_shot
    assert r"\boxed{A,D}" not in few_shot
    assert r"\boxed{X,Y}" not in few_shot


def test_agent_prompt_can_disable_few_shot():
    mod = _load_agent_run_module()
    prompt = mod.build_prompt("题目：OFD格式电子发票必须提供OFD格式源文件。", {"use_few_shot": False})
    assert "Few-shot 协议示范" not in prompt
    assert "<search>bypass 安全设施 处罚 绩效</search>" not in prompt
    assert r"\boxed{X,Y}" not in prompt


def test_agent_prompt_strips_raw_single_turn_answer_instructions():
    mod = _load_agent_run_module()
    raw_query = (
        "下面是一道多选题。选出所有正确的选项（可能不止一个）。请先简要分析再作答。\n"
        "把最终答案放入 \\boxed{}，按字母顺序列出所有正确字母、用逗号分隔（如 \\boxed{A,C,D}）。\n\n"
        "题目：登高作业需搭设脚手架时，应注意（）\n\n"
        "选项：\n"
        "A. 脚手架上有人时，不得移动脚手架\n"
        "B. 佩戴安全带"
    )
    prompt = mod.build_prompt(raw_query, {"use_few_shot": False})
    question_part = prompt.split("真实题目如下", 1)[1]
    assert "题型：多选题" in question_part
    assert "题目：登高作业需搭设脚手架时" in question_part
    assert "A. 脚手架上有人时" in question_part
    assert "请先简要分析再作答" not in question_part
    assert "把最终答案放入" not in question_part
    assert r"\boxed{A,C,D}" not in question_part
    assert "[题目开始]" not in prompt
    assert "[题目结束]" not in prompt


def test_parse_search_and_boxed_is_invalid_mixed_action():
    action = parse_action(r"<search>控制图分析</search> 现在确定 \boxed{A}")
    assert action.action == "invalid"
    assert action.error == "mixed_action"
    assert action.mixed_action is True
    assert action.boxed_answers == ["A"]


def test_parse_read_action():
    action = parse_action(r"<think>需要读第1条。</think><read>1</read>")
    assert action.action == "read"
    assert action.read_ids == (1,)
    assert action.has_extra_text is False


def test_parse_read_and_boxed_is_invalid_mixed_action():
    action = parse_action(r"<read>1</read>\boxed{A}")
    assert action.action == "invalid"
    assert action.error == "mixed_action"


def test_runner_mixed_action_does_not_grade_answer():
    runner = QARunner({"docs_dir": "missing", "max_turns": 3})
    meta = {"query": "q", "expected_answer": "[single] A", "num_turns": 0, "max_turns": 3}
    result = runner.process_turn(
        [{"role": "assistant", "content": r"<search>q</search>\boxed{A}"}],
        meta,
    )
    assert result.terminated is False
    assert result.answer is None
    assert result.stats.format_error is True
    assert "动作无效" in result.observation["content"]
    assert "同一轮同时出现多个动作" in result.observation["content"]
    assert "本轮没有执行检索、阅读或评分" in result.observation["content"]
    assert "下一步" in result.observation["content"]


def test_parse_boxed_allows_space_before_brace():
    action = parse_action(r"答案是 \boxed {A}")
    assert action.action == "answer"
    assert action.boxed_answers == ["A"]


def test_parse_multiple_boxed_takes_last_in_runner():
    runner = QARunner({"docs_dir": "missing"})
    meta = {"expected_answer": "[single] A", "num_turns": 0, "max_turns": 3}
    result = runner.process_turn(
        [{"role": "assistant", "content": r"\boxed{B} 修正为 \boxed{A}"}],
        meta,
    )
    assert result.terminated is True
    assert result.answer == "A"
    assert result.reward > 0.9


def test_parse_think_blocks_do_not_trigger_action():
    action = parse_action(r"<think>\boxed{A}</think><search>OFD 发票 源文件</search>")
    assert action.action == "search"
    assert action.search_queries[0] == "OFD 发票 源文件"
    assert action.has_extra_text is False


def test_parse_unclosed_think_does_not_hide_later_search():
    action = parse_action(r"<think>我需要查询资料。<search>OFD 发票 源文件</search>")
    assert action.action == "search"
    assert action.search_queries[0] == "OFD 发票 源文件"


def test_parse_rejects_placeholder_search_query():
    action = parse_action("<search>关键词</search>")
    assert action.action == "invalid"
    assert action.error == "empty_search"


def test_search_with_visible_draft_before_tag_is_allowed():
    action = parse_action("我需要先查发票源文件。<search>OFD 发票 源文件</search>")
    assert action.action == "search"
    assert action.has_extra_text is False


def test_search_with_trailing_text_is_marked():
    action = parse_action("<search>OFD 发票 源文件</search>我猜答案是 A")
    assert action.action == "search"
    assert action.has_extra_text is True


def test_runner_final_answer_uses_official_reward():
    runner = QARunner({"docs_dir": "missing"})
    meta = {"query": "q", "expected_answer": "[multiple] A,C", "num_turns": 0, "max_turns": 3}
    result = runner.process_turn(
        [{"role": "assistant", "content": r"\boxed{A,B,C}"}],
        meta,
    )
    official = qa_rule_reward_fn(["q"], [r"\boxed{A,B,C}"], ["[multiple] A,C"])[0]
    assert result.reward == official
    assert result.reward == 0.75


def test_mixed_language_search(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spc.md").write_text(
        "# 控制图分析\n\n控制图分析时，Exclude 功能可以永久排除 Sample，Point Disable 只是临时禁用。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("控制图分析 永久排除 Sample", top_k=2)
    assert hits
    assert "Exclude" in hits[0].chunk.text
    assert "控制图分析" in hits[0].matched_terms


def test_bypass_query_is_normalized_and_expanded(tmp_path: Path):
    assert clean_query("私自 by pass 安全设施 处罚") == "私自 bypass 安全设施 处罚"
    intents = split_search_intents("私自 by-pass 安全设施 处罚", max_queries=3)
    assert intents[0] == "私自 bypass 安全设施 处罚"
    assert "bypass" in intents

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "safety.md").write_text(
        "# 安全设施管理\n\n私自旁路或屏蔽安全联锁，即使只造成虚惊事件，也会影响季度绩效和年度绩效，并给予警告。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("私自 by pass 安全设施 处罚", top_k=2)
    assert hits
    assert "安全联锁" in hits[0].chunk.text
    assert "季度绩效" in hits[0].chunk.text


def test_annual_leave_query_requires_leave_domain_anchor(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "11.3 LITHO KPI.md").write_text(
        "# 11.3 LITHO KPI\n\n"
        "121337 97839 170510 125694 2023M09 2023M10 2024W01 "
        "LARF202 Performance 87.33% 12.79% Move Target Average Move",
        encoding="utf-8",
    )
    (docs / "r2r.md").write_text(
        "# OVL 控制\n\nOVL(3)控制功能 FB exclusion feature 反馈有效期控制 Function MES Process。",
        encoding="utf-8",
    )
    (docs / "员工假期管理.md").write_text(
        "# 员工假期管理\n\n年度带薪年休假有效期为当年度，逾期按公司假期管理制度处理。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("带薪年假有效期", top_k=3)
    assert hits
    joined = "\n".join(hit.chunk.text for hit in hits)
    assert "带薪年休假有效期" in hits[0].chunk.text
    assert "LITHO KPI" not in joined
    assert "反馈有效期控制" not in joined


def test_annual_leave_query_returns_empty_instead_of_weak_noise(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "11.3 LITHO KPI.md").write_text(
        "# 11.3 LITHO KPI\n\n2023M09 2023M10 2024W01 Performance 87.33% 12.79% Move Target。",
        encoding="utf-8",
    )
    (docs / "r2r.md").write_text("# OVL 控制\n\n反馈有效期控制 Function MES Process。", encoding="utf-8")
    engine = SimpleSearchEngine(docs)
    assert engine.search("带薪年假有效期", top_k=3) == tuple()


def test_search_prioritizes_hits_with_more_matched_keywords(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "weak.md").write_text("# 弱结果\n\n控制图分析可以用于过程控制。", encoding="utf-8")
    (docs / "strong.md").write_text(
        "# 强结果\n\n控制图分析时，Exclude 功能可以永久排除 Sample。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("控制图分析 Exclude 永久排除 Sample", top_k=2)
    assert hits
    assert "Exclude" in hits[0].chunk.text
    assert len(hits[0].matched_terms) > len(hits[1].matched_terms)


def test_search_prioritizes_proper_terms_over_generic_words(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "generic.md").write_text(
        "# 创建功能\n\n系统支持创建功能，以下说法包括正确、错误、操作、流程等通用说明。",
        encoding="utf-8",
    )
    (docs / "carrier.md").write_text(
        "# Carrier 创建\n\nCarrier 页面支持手动创建 FOUP/FOSB，也支持一次创建多个 FOUP/FOSB。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("Carrier 创建 FOUP FOSB 错误说法", top_k=2)
    assert hits
    assert "FOUP/FOSB" in hits[0].chunk.text


def test_search_english_quantity_question_requires_counted_objects_and_count_evidence(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "acc.md").write_text(
        "# Process Overview\n\n"
        "Automatic Chuck Chamber Clean cleans process module chambers and medium dispenser nozzles "
        "with DI water after a process.",
        encoding="utf-8",
    )
    (docs / "description.md").write_text(
        "# Description\n\n"
        "1 Medium 1 Position 2 Medium 2 Position 3 DI Position. "
        "Dispenser arm moves above the wafer during process.",
        encoding="utf-8",
    )
    (docs / "pm_config.md").write_text(
        "# Process Module Configuration\n\n"
        "Each process module has 2 medium dispensers and 1 dispenser arm.",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("process module dispenser arm 数量", top_k=3)
    assert hits
    assert "2 medium dispensers and 1 dispenser arm" in hits[0].chunk.text
    assert all("dispenser nozzles" not in hit.chunk.text for hit in hits[:1])

    count_hits = engine.search("process module dispenser arm count", top_k=3)
    assert count_hits
    assert "2 medium dispensers and 1 dispenser arm" in count_hits[0].chunk.text
    assert all("dispenser nozzles" not in hit.chunk.text for hit in count_hits[:1])


def test_search_downgrades_command_table_for_quantity_question(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "commands.md").write_text(
        "# Machine Administration Service\n\n"
        "Table 6-1 MSP Commands. Move to Work moves the dispenser arm to the work position. "
        "This command is enabled after Home Position Outlet Search of the Medium Dispenser, "
        "1-5 Lift Cylinders are in upper position, and PM/Process Module Axis Power Off.",
        encoding="utf-8",
    )
    (docs / "config.md").write_text(
        "# Process Module Configuration\n\n"
        "For each process module, the configuration is 2 medium dispensers and 1 dispenser arm.",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("process module dispenser arm count", top_k=2)
    assert hits
    assert "2 medium dispensers and 1 dispenser arm" in hits[0].chunk.text
    assert "MSP Commands" not in hits[0].chunk.text


def test_search_removes_near_duplicate_snippets(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    repeated = (
        "# 控制图\n\n"
        "第二阶段:产品设计和开发 2.5 样件制造控制计划是对样件制造过程中的尺寸测量和材料与功能试验的描述。"
        "为小组和顾客提供评估产品或服务符合规范和报告产品状态的资料。"
    )
    (docs / "a.md").write_text(repeated, encoding="utf-8")
    (docs / "b.md").write_text(repeated + " 产出记录略有不同。", encoding="utf-8")
    (docs / "c.md").write_text(
        "# 控制图分析\n\n控制图分析时，Point Disable 可以永久排除 Sample。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("控制图 永久排除 Sample 功能", top_k=3)
    texts = [hit.chunk.text for hit in hits]
    duplicate_like = sum("第二阶段:产品设计和开发" in text for text in texts)
    assert duplicate_like <= 1


def test_retrieval_quality_for_quantity_question_requires_quantity_evidence():
    query = "题目：How many dispenser and arm are there for each process module"
    weak = (
        "Operate Package - Wafer Processing Process Overview. "
        "Automatic Chuck Chamber Clean cleans process module chambers and medium dispenser nozzles."
    )
    strong = "Each process module has 2 medium dispensers and 1 dispenser arm."
    assert is_quantity_query(query)
    assert count_evidence_score(query, text=weak) < 5.0
    assert retrieval_quality(query, weak) in {"weak", "none"}
    assert count_evidence_score(query, text=strong) >= 5.0
    assert retrieval_quality(query, strong) == "strong"


def test_approval_question_partial_definition_hit_is_not_strong():
    query = (
        "题目：有限空间作业票需审批到谁才可以作业？\n"
        "选项：\n"
        "A. 石厂\n"
        "B. EHS 工程师\n"
        "C. 部门负责人"
    )
    partial = (
        "作业票的意义及分类 风险控制 作业票的意义。"
        "通过现场作业票的公示，呈现作业单位、作业内容、作业时间、作业主管部门等信息。\n"
        "有限空间作业 有限空间是指与外界相对隔离，进出口受限，自然通风不良。"
    )
    strong = "有限空间作业票必须经 EHS 工程师审批通过后才可以作业。"
    assert retrieval_quality(query, partial) == "weak"
    assert retrieval_quality(query, strong) == "strong"


def test_runner_approval_question_prompts_read_when_snippet_lacks_approval_answer(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "permit.md").write_text(
        "# 有限空间作业\n\n"
        "作业票的意义及分类 风险控制 作业票用于呈现作业单位、作业内容、作业时间、作业主管部门等信息。\n\n"
        "有限空间作业是指与外界相对隔离，进出口受限，自然通风不良的作业空间。\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4, "max_search_attempts": 3})
    meta = {
        "query": (
            "题目：有限空间作业票需审批到谁才可以作业？\n"
            "选项：\n"
            "A. 石厂\n"
            "B. EHS 工程师\n"
            "C. 部门负责人"
        ),
        "expected_answer": "[single] B",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>有限空间作业</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert result.metadata["evidence_quality"] == "weak"
    assert result.metadata.get("answer_mode") is not True
    assert "如果目录相关就只输出 <read>编号</read>" in result.observation["content"]
    assert "不要继续 search" not in result.observation["content"]


def test_weak_retrieval_at_max_search_attempts_allows_read_before_answer_mode(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "permit.md").write_text(
        "# 有限空间作业\n\n"
        "作业票的意义及分类 风险控制 作业票用于呈现作业单位、作业内容、作业时间、作业主管部门等信息。\n\n"
        "有限空间作业是指与外界相对隔离，进出口受限，自然通风不良的作业空间。\n",
        encoding="utf-8",
    )
    runner = QARunner(
        {
            "docs_dir": str(docs),
            "max_turns": 4,
            "max_search_attempts": 1,
            "max_read_attempts": 1,
        }
    )
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>有限空间作业</search>"}],
        {
            "query": (
                "题目：有限空间作业票需审批到谁才可以作业？\n"
                "选项：\n"
                "A. 石厂\n"
                "B. EHS 工程师\n"
                "C. 部门负责人"
            ),
            "expected_answer": "[single] B",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert result.metadata is not None
    assert result.metadata["search_attempts"] == 1
    assert result.metadata["evidence_quality"] == "weak"
    assert result.metadata.get("answer_mode") is not True
    assert result.metadata["last_search_hits"]
    assert "如果目录相关就只输出 <read>编号</read>" in result.observation["content"]
    assert "不要继续 search" not in result.observation["content"]


def test_search_does_not_return_generic_source_file_without_mandatory_anchor(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "server.md").write_text(
        "# Server开发\n\n上传 C++ 源文件，检查项目路径和编译文件。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    assert engine.search("OFD格式 电子发票 源文件 要求", top_k=3) == tuple()

    (docs / "invoice.md").write_text(
        "# OFD 发票\n\nOFD格式电子发票必须提供OFD格式源文件。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("OFD格式 电子发票 源文件 要求", top_k=3)
    assert hits
    assert "OFD格式电子发票" in hits[0].chunk.text


def test_ofd_invoice_query_does_not_split_to_ambiguous_short_acronym():
    intents = split_search_intents("OFD 电子发票 源文件 必须", max_queries=3)
    assert intents == ["OFD 电子发票 源文件 必须"]


def test_short_acronym_matching_rejects_ocr_glued_lowercase_noise():
    noisy = (
        "The rate ofd change of each parameter is determined by MOSFET process details. "
        "characteristicsofdropletgeneratedbyRayleigh breakup."
    )
    assert proper_term_match_count_from_terms(("OFD",), text=noisy) == 0
    assert proper_term_match_count_from_terms(("OFD",), text="OFD格式电子发票必须提供OFD格式源文件。") == 1


def test_search_ofd_invoice_ignores_semiconductor_ocr_noise(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "laser_review.md").write_text(
        "# 内封面文章\n\n"
        "激光与光电子学进展 Atomic absorption cross section Alkaline metals Transition metals "
        "光刻胶性能不断发展。",
        encoding="utf-8",
    )
    (docs / "mosfet.md").write_text(
        "# MOSFET\n\n"
        "The rate ofd change of each parameter is determined by the MOSFET design. "
        "characteristicsofdropletgeneratedbyRayleigh breakup.",
        encoding="utf-8",
    )
    (docs / "apqp.md").write_text(
        "# APQP各阶段常用质量工具\n\n"
        "计划与定义 SOP移交 产品设计与验证 反馈、评定与纠正措施 DFMADFMEA 质量目标 图纸规范变更。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    assert engine.search("OFD 电子发票 源文件 必须", top_k=3) == tuple()

    (docs / "invoice.md").write_text(
        "# 电子发票要求\n\nOFD格式电子发票必须提供OFD格式源文件。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("OFD 电子发票 源文件 必须", top_k=3)
    assert hits
    assert "OFD格式电子发票" in hits[0].chunk.text
    assert "MOSFET" not in "\n".join(hit.chunk.text for hit in hits)


def test_catalog_search_ofd_invoice_ignores_quality_and_litho_noise(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "apqp.md").write_text(
        "# APQP&CP培训材料\n\n"
        "APQP各阶段常用质量工具包括 TRIZ、QFD、OFD、DFMEA、FMEA、SPC、MSA。",
        encoding="utf-8",
    )
    (docs / "litho.md").write_text(
        "# 极紫外光刻研究\n\n"
        "EUV lithography and optical source research, ofd optical droplet generation.",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    assert engine.search_catalog("OFD格式电子发票 必须提供源文件", top_k=3) == tuple()

    (docs / "invoice.md").write_text(
        "# 电子发票要求\n\nOFD格式电子发票必须提供OFD格式源文件。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search_catalog("OFD格式电子发票 必须提供源文件", top_k=3)
    assert hits
    assert hits[0].section.doc_title == "电子发票要求"
    assert all("APQP" not in hit.section.doc_title for hit in hits)


def test_catalog_search_excludes_unpaired_assessment_files(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "19.MO case summary试题.md").write_text(
        "# 19.MO case summary试题\n\nMO case summary 题目内容。",
        encoding="utf-8",
    )
    (docs / "25.对准原理和对准Mark,光源介绍试题.md").write_text(
        "# 25.对准原理和对准Mark,光源介绍试题\n\n对准 Mark 试题内容。",
        encoding="utf-8",
    )
    (docs / "Error-Proof培训.md").write_text(
        "# Error-Proof培训\n\nError proof 防呆培训正文。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    titles = [hit.section.doc_title for hit in engine.search_catalog("MO case 对准 Error Proof", top_k=6)]
    assert "Error-Proof培训" in titles
    assert all("试题" not in title for title in titles)


def test_catalog_search_prefers_deepest_complete_numbered_section_path(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "incoming.md").write_text(
        "# 【4.1】来料检验范围、标准及流程定义\n\n"
        "【4】来料质量管理\n"
        "本章定义来料质量管理的整体流程。\n\n"
        "【4.1】来料检验范围、标准及流程定义\n"
        "本节定义来料检验范围、检验标准和流程。\n\n"
        "【4.1.1】检验范围\n"
        "检验范围包括原材料、零配件和供应商来料。\n\n",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search_catalog("来料检验范围 标准 流程定义", top_k=3)
    assert hits
    assert hits[0].section.section_path == (
        "【4】来料质量管理",
        "【4.1】来料检验范围、标准及流程定义",
        "【4.1.1】检验范围",
    )


def test_catalog_search_infers_plain_dotted_sections_without_decimal_noise(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "incoming_plain.md").write_text(
        "# 来料检验手册\n\n"
        "【4】来料质量管理\n"
        "本章定义来料质量管理的整体流程。\n\n"
        "4.1 来料检验范围、标准及流程定义\n"
        "本节定义来料检验范围、检验标准和流程。\n\n"
        "0.30 0.27 0.25 0.20 2024Q1 2024Q2\n"
        "4.5V 电压参数不是章节标题\n\n"
        "4.1.1 检验范围\n"
        "检验范围包括原材料、零配件和供应商来料。\n\n",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search_catalog("供应商 来料 检验范围", top_k=3)
    assert hits
    assert hits[0].section.section_path == (
        "【4】来料质量管理",
        "【4.1】来料检验范围、标准及流程定义",
        "【4.1.1】检验范围",
    )
    section_paths = [" > ".join(section.section_path) for section in engine._catalog_sections]
    assert all("【0.30】" not in path for path in section_paths)
    assert all("【4.5】V" not in path for path in section_paths)


def test_catalog_search_skips_dedicated_toc_and_returns_content_heading(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "incoming_book.md").write_text(
        "# 目录\n\n"
        "4 来料质量管理 ........ 3\n"
        "4.1 来料检验范围、标准及流程定义 ........ 8\n"
        "4.1.1 检验范围 ........ 9\n\n"
        "# 【4】来料质量管理\n\n"
        "本章定义来料质量管理的整体流程。\n\n"
        "## 【4.1】来料检验范围、标准及流程定义\n\n"
        "本节定义来料检验范围、检验标准和流程。\n\n"
        "### 【4.1.1】检验范围\n\n"
        "检验范围包括原材料、零配件和供应商来料。\n",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search_catalog("供应商 来料 检验范围", top_k=3)
    assert hits
    assert hits[0].section.section_path == (
        "【4】来料质量管理",
        "【4.1】来料检验范围、标准及流程定义",
        "【4.1.1】检验范围",
    )
    assert all("目录" not in " > ".join(hit.section.section_path) for hit in hits)


def test_catalog_search_does_not_return_toc_only_entries(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "toc_only.md").write_text(
        "# 目录\n\n"
        "4 来料质量管理 ........ 3\n"
        "4.1 来料检验范围、标准及流程定义 ........ 8\n"
        "4.1.1 检验范围 ........ 9\n",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    assert engine.search_catalog("来料检验范围 标准 流程定义", top_k=3) == tuple()


def test_catalog_output_shows_one_complete_leaf_path():
    hits = [
        DocumentSectionHit(
            DocumentSection(
                "incoming.md",
                "来料检验手册",
                "来料检验范围、标准及流程定义",
                ("【4】来料质量管理", "【4.1】来料检验范围、标准及流程定义"),
                2,
                1,
                "来料检验范围、标准及流程定义。",
            ),
            6.0,
            "来料检验范围 标准 流程定义",
            ("来料检验", "标准"),
        ),
        DocumentSectionHit(
            DocumentSection(
                "incoming.md",
                "来料检验手册",
                "检验范围",
                ("【4】来料质量管理", "【4.1】来料检验范围、标准及流程定义", "【4.1.1】检验范围"),
                3,
                2,
                "检验范围包括原材料、零配件和供应商来料。",
            ),
            5.8,
            "来料检验范围 标准 流程定义",
            ("来料检验", "标准"),
        ),
    ]
    text, refs = _format_ranked_search_results(["来料检验范围 标准 流程定义"], [], hits, max_tokens=260)
    assert len(refs) == 1
    assert "文档：来料检验手册" in text
    assert "目录：来料检验手册 > 【4】来料质量管理 > 【4.1】来料检验范围、标准及流程定义 > 【4.1.1】检验范围" in text
    assert "；" not in text


def test_catalog_output_merges_numbered_duplicate_document_titles():
    hits = [
        DocumentSectionHit(
            DocumentSection(
                "mes.md",
                "MES系统介绍",
                "MES系统介绍",
                ("MES系统介绍",),
                1,
                0,
                "MES系统介绍包含系统模块、创建 Carrier 和生产流程。",
            ),
            5.0,
            "MES系统介绍",
            ("MES", "系统"),
        ),
        DocumentSectionHit(
            DocumentSection(
                "mes_part.md",
                "1.3 MES系统介绍",
                "1.3 MES系统介绍",
                ("1.3 MES系统介绍",),
                1,
                0,
                "MES系统介绍包含系统模块、创建 Carrier 和生产流程。",
            ),
            6.0,
            "MES系统介绍",
            ("MES", "系统"),
        ),
    ]
    text, refs = _format_ranked_search_results(["MES系统介绍"], [], hits, max_tokens=260)
    assert len(refs) == 1
    assert text.count("文档：") == 1
    assert "文档：MES系统介绍" in text
    assert "1.3 MES系统介绍" not in text


def test_catalog_search_rejects_numeric_table_rows_as_numbered_headings(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "incoming_abnormal.md").write_text(
        "# 【4.2】来料异常处理标准及流程\n\n"
        "原材料来料异常类型说明。\n\n"
        "【7.4】Area Count >= 3 um 3 0 5 0 5 0 5 0 5 0 5 0\n"
        "这是一行表格数值，不是章节标题。\n\n"
        "【7.5】异常处理流程\n"
        "发现来料异常时需要隔离、标识并通知相关负责人处理。\n",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    section_paths = [" > ".join(section.section_path) for section in engine._catalog_sections]
    assert all("Area Count" not in path for path in section_paths)


def test_feedback_cleaner_dedupes_repeated_numbered_catalog_list():
    duplicated = (
        "1. 文档：4.5 Disco DFD6860 Schedule PM&常见 Alarm介绍与实际操作\n"
        "目录：4.5 Disco DFD6860 Schedule PM&常见 Alarm介绍与实际操作 > 【7.3】安全卫生及环保事故处理程序\n"
        "2. 文档：7.2 危险废弃物的厂内管理\n"
        "目录：7.2 危险废弃物的厂内管理\n"
        "3. 文档：6.6 MPG Schedule PM & 常见 Alarm介绍与实际操作\n"
        "目录：6.6 MPG Schedule PM & 常见 Alarm介绍与实际操作 > Page 27"
        "1. 文档：4.5 Disco DFD6860 Schedule PM&常见 Alarm介绍与实际操作\n"
        "目录：4.5 Disco DFD6860 Schedule PM&常见 Alarm介绍与实际操作 > 【7.3】安全卫生及环保事故处理程序\n"
        "2. 文档：7.2 危险废弃物的厂内管理\n"
        "目录：7.2 危险废弃物的厂内管理\n"
        "3. 文档：6.6 MPG Schedule PM & 常见 Alarm介绍与实际操作\n"
        "目录：6.6 MPG Schedule PM & 常见 Alarm介绍与实际操作 > Page 27\n"
        "\n下一步：如果目录相关就只输出 <read>编号</read>，打开该文档章节继续阅读。\n"
    )
    cleaned = _clean_visible_feedback_text(duplicated)
    assert cleaned.count("文档：4.5 Disco DFD6860") == 1
    assert cleaned.count("文档：7.2 危险废弃物") == 1
    assert cleaned.count("文档：6.6 MPG") == 1
    assert "Page 27\n1. 文档" not in cleaned
    assert "下一步：如果目录相关" in cleaned


def test_question_bank_pairs_assessment_answer_across_directories(tmp_path: Path):
    docs = tmp_path / "docs"
    (docs / "questions").mkdir(parents=True)
    (docs / "answers").mkdir(parents=True)
    (docs / "questions" / "19.MO case summary试题.md").write_text(
        "一、选择题\n"
        "1. CD OOS 需要确认哪些数据( ) A.CD Image B.量测recipe是否有变更 C.下货条件及过货是否异常 D.以上都要确认\n",
        encoding="utf-8",
    )
    (docs / "answers" / "19.MO case summary答案.md").write_text(
        "选择题：\nD\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>CD OOS 确认 数据</search>"}],
        {
            "query": "题目：CD OOS 需要确认哪些数据( )\n选项：\nA. CD Image\nB. 量测recipe是否有变更\nC. 下货条件及过货是否异常\nD. 以上都要确认",
            "expected_answer": "[single] D",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "题目：CD OOS 需要确认哪些数据" in result.observation["content"]
    assert "答案：以上都要确认" in result.observation["content"]
    assert "试卷：19.MO case summary试题" not in result.observation["content"]
    assert "答案来源：" not in result.observation["content"]
    assert "文档：19.MO case summary试题" not in result.observation["content"]


def test_question_bank_parses_inline_answer_from_assessment_file(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "05-Zero MO试题.md").write_text(
        "Zero MO知识试题\n"
        "一、多选题\n"
        "1. 下列哪些属于优秀工程师应具备的工作技能?（A C） A. 分析能力 B. 同理心 C. 系统与结构方面的能力 D. 守纪律\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>优秀工程师 工作技能 分析能力 系统结构</search>"}],
        {
            "query": "题目：下列哪些属于优秀工程师应具备的工作技能?\n选项：\nA. 分析能力\nB. 同理心\nC. 系统与结构方面的能力\nD. 守纪律",
            "expected_answer": "[multiple] A,C",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "题目：下列哪些属于优秀工程师应具备的工作技能" in result.observation["content"]
    assert "答案：分析能力；系统与结构方面的能力" in result.observation["content"]
    assert "资料：05-Zero MO试题" not in result.observation["content"]
    assert "题答位置：同一文件" not in result.observation["content"]
    assert "文档：05-Zero MO试题" not in result.observation["content"]


def test_question_bank_skips_placeholder_format_questions(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "试题format.md").write_text(
        "判断题\n"
        "1. 。。。。。。。。（对）\n"
        "2. ________（错）\n",
        encoding="utf-8",
    )
    (docs / "cleanroom.md").write_text(
        "# 行为规范\n\n手套、无尘服、电脑、设备、工具等禁止写字乱涂乱画。",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>禁止 写字 乱涂乱画</search>"}],
        {
            "query": "题目：手套、无尘服、电脑、设备、工具等禁止写字乱涂乱画\n选项：\nA. 对\nB. 错",
            "expected_answer": "[bool] A",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "试题format" not in result.observation["content"]
    assert "题目：。。。。" not in result.observation["content"]
    assert "答案：对" not in result.observation["content"]
    assert "文档：行为规范" in result.observation["content"]


def test_search_ofd_invoice_allows_invoice_source_file_combination(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "invoice_source.md").write_text(
        "# 报销附件要求\n\n发票报销时必须提供源文件，纸质打印件不能替代源文件。",
        encoding="utf-8",
    )
    (docs / "mosfet.md").write_text(
        "# MOSFET\n\nThe rate ofd change of each parameter is determined by MOSFET details.",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("OFD格式电子发票 必须提供源文件", top_k=3)
    assert hits
    assert "发票报销时必须提供源文件" in hits[0].chunk.text
    assert all("MOSFET" not in hit.chunk.text for hit in hits)


def test_search_does_not_return_feedback_validity_for_paid_leave_query(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "r2r.md").write_text(
        "# OVL控制\n\n反馈有效期控制用于 R2R 过程控制。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    assert engine.search("带薪年假有效期", top_k=3) == tuple()

    (docs / "leave.md").write_text(
        "# 员工假期\n\n带薪年假有效期通常以公司制度为准。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("带薪年假有效期", top_k=3)
    assert hits
    assert "带薪年假" in hits[0].chunk.text


def test_ranked_results_keep_question_bank_hit_visible_when_present():
    query = "控制图 永久排除样本 功能"
    bank_hit = QuestionBankHit(
        QuestionBankItem(
            qtype="single",
            ordinal=1,
            question="控制图分析时，以下哪个功能可以永久排除Sample（ ）",
            answer="A",
            question_path="Group SPC-试卷.md",
            answer_path="Group SPC-答案.md",
            score_key="控制图分析永久排除sample功能",
            options=(("A", "Exclude"), ("B", "Point Disable")),
            answer_detail="Exclude",
        ),
        0.61,
    )
    doc_hits = [
        SearchHit(
            SearchChunk(
                f"doc{i}.md",
                "内部文档片段",
                i,
                "控制图 永久 排除 样本 功能 的普通文档说明，包含大量与查询相同的词，但不是题库答案。",
            ),
            20.0 - i,
            query,
            ("控制图", "永久", "排除", "样本", "功能"),
        )
        for i in range(4)
    ]
    text, refs = _format_ranked_search_results([query], [bank_hit], doc_hits, max_tokens=280, max_results=2)
    assert "题目：控制图分析时" in text
    assert "答案：Exclude" in text
    assert "试卷：Group SPC-试卷" not in text
    assert "答案来源：Group SPC-答案" not in text
    assert refs[0].get("question_bank") is True


def test_search_expands_from_outline_to_section_body(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spc_manual.md").write_text(
        "# 目录\n\n"
        "1. 控制图分析........7\n"
        "2. 报警配置........8\n\n"
        "# 控制图分析\n\n"
        "Exclude 功能可以永久排除 Sample，Point Disable 只是临时禁用异常点。",
        encoding="utf-8",
    )
    engine = SimpleSearchEngine(docs)
    hits = engine.search("控制图分析", top_k=2)
    assert hits
    assert any("Exclude" in hit.chunk.text for hit in hits)


def test_runner_suppresses_near_duplicate_retrieval_chunks():
    duplicate_a = (
        "第二阶段:产品设计和开发 2.5 -样件制造控制计划 样件控制计划是对样件制造过程中的"
        "尺寸测量和材料与功能试验的描述。为小组和顾客提供了一个极好的机会。"
    )
    duplicate_b = duplicate_a + " 服务符合所要求的规范和报告 产品。"
    chunks = [
        SearchChunk("manual.md", "内部文档片段", 0, "过程控制图和控制计划用于记录产品设计资料。"),
        SearchChunk("manual.md", "内部文档片段", 1, duplicate_a),
        SearchChunk("manual.md", "内部文档片段", 2, duplicate_b),
    ]
    runner = QARunner({"docs_dir": "missing", "max_turns": 3, "max_retrieval_tokens_per_turn": 360})
    engine = SimpleSearchEngine(chunks=chunks)
    runner.search_engine = engine
    runner.retrieval.search_engine = engine
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>控制图 样件制造控制计划 功能</search>"}],
        {
            "query": "题目：控制图分析时，以下哪个功能可以永久排除Sample（ ）",
            "expected_answer": "[single] A",
            "num_turns": 0,
            "max_turns": 3,
        },
    )
    assert "文档：manual" in result.observation["content"]
    assert "目录：" in result.observation["content"]
    assert "第二阶段:产品设计和开发" not in result.observation["content"]
    assert len(result.metadata["last_search_hits"]) == 1


def test_catalog_discovery_groups_sections_by_document_when_question_bank_misses():
    hits = [
        DocumentSectionHit(
            DocumentSection(
                "permit.md",
                "安全作业培训",
                "作业票",
                ("安全作业培训", "作业票"),
                2,
                0,
                "作业票的意义及分类：通过作业票检查表确认安全措施落实。",
            ),
            4.0,
            "有限空间作业票 审批",
        ),
        DocumentSectionHit(
            DocumentSection(
                "permit.md",
                "安全作业培训",
                "有限空间作业",
                ("安全作业培训", "作业票", "有限空间作业"),
                3,
                1,
                "有限空间是进出口受限、通风不良的空间，主要风险包括中毒窒息、火灾爆炸。",
            ),
            3.5,
            "有限空间作业票 审批",
        ),
        DocumentSectionHit(
            DocumentSection(
                "permit.md",
                "安全作业培训",
                "有限空间作业审批",
                ("安全作业培训", "作业票", "有限空间作业审批"),
                3,
                2,
                "有限空间作业票审批流程见本章节，作业前需要确认审批权限和现场安全措施。",
            ),
            5.0,
            "有限空间作业票 审批",
        ),
    ]
    text, refs = _format_ranked_search_results(["有限空间作业票 审批"], [], hits, max_tokens=360)

    assert len(refs) == 1
    assert refs[0]["catalog_entry"] is True
    assert refs[0]["doc_path"] == "permit.md"
    assert "文档：安全作业培训" in text
    assert "目录：" in text
    assert text.count("有限空间") >= 1


def test_runner_allows_followup_search_after_weak_read(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "permit.md").write_text(
        "# 有限空间作业\n\n"
        "有限空间是进出口受限、自然通风不良的空间，主要风险包括中毒窒息、火灾爆炸。\n\n"
        "# 作业票审批\n\n"
        "审批流程需结合现场作业类型确认。",
        encoding="utf-8",
    )
    runner = QARunner(
        {
            "docs_dir": str(docs),
            "max_turns": 6,
            "max_search_attempts": 3,
            "max_read_attempts": 2,
            "max_retrieval_tokens_per_turn": 260,
            "max_read_tokens_per_turn": 220,
        }
    )
    meta = {
        "query": "题目：有限空间作业票需审批到谁才可以作业？\n选项：\nA. 石厂\nB. EHS 工程师\nC. 部门负责人",
        "expected_answer": "[single] B",
        "num_turns": 0,
        "max_turns": 6,
    }
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>有限空间作业票 审批</search>"}],
        meta,
    )
    second = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        first.metadata,
    )

    assert second.metadata is not None
    assert second.metadata["answer_mode"] is False
    assert "资料标题、相关章节、专有名词继续 search" in second.observation["content"]


def test_conversion_failure_manifest_is_not_searchable(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "转换失败清单(共 1155 个).md").write_text(
        "# 转换失败清单(共 1155 个)\n\n"
        "/data/word/20260611160325_5. 货物复运出口操作流程试题.docx "
        "原因: FileNotFoundError: [Errno 2] No such file or directory: "
        "'/data/word/20260611160325_5. 货物复运出口操作流程试题.docx'\n"
        "/data/ppt/20260611160325_7. 零配件进口流程.pptx "
        "原因: FileNotFoundError: [Errno 2] No such file or directory: "
        "'/data/ppt/20260611160325_7. 零配件进口流程.pptx'\n"
        "/data/word/20260611160325_6. 原材料进口操作流程试题.docx "
        "原因: FileNotFoundError: [Errno 2] No such file or directory: "
        "'/data/word/20260611160325_6. 原材料进口操作流程试题.docx'\n",
        encoding="utf-8",
    )
    (docs / "原材料进口操作流程.md").write_text(
        "# 原材料进口操作流程\n\n进口原材料到厂后，需按采购、报关、仓储验收流程处理。",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 3, "max_retrieval_tokens_per_turn": 220})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>原材料进口操作流程</search>"}],
        {"query": "题目：原材料进口操作流程", "expected_answer": "[short] 采购 ||| 报关 ||| 仓储验收", "num_turns": 0, "max_turns": 3},
    )
    assert "原材料进口操作流程" in result.observation["content"]
    assert "转换失败清单" not in result.observation["content"]
    assert "FileNotFoundError" not in result.observation["content"]
    assert "No such file or directory" not in result.observation["content"]


def test_retrieval_output_filters_dense_ocr_panel_noise():
    codes = " ".join(f"E{i}" for i in range(5388, 5410))
    noisy = (
        f"{codes}\n"
        "DN ON DN ON DN ON DN ON DN ON DN ON DN ON DN ON\n"
        "MFC MFC MFC MFC MFC MFC MFC MFC\n"
        "FAB发生火灾或机台着火时，应按火灾应急处置流程处理，优先保证人员安全。"
    )
    chunk = SearchChunk("fab.md", "3.2FAB异常事件处理流程", 0, noisy)
    hit = SearchHit(chunk, 10.0, "火灾 应急处置")

    cleaned = clean_retrieval_snippet(noisy)
    search_text, _ = format_hits_with_refs([hit], max_tokens=180)
    read_text = format_structured_read_context(
        (chunk,),
        center_chunk_id=0,
        source_id=1,
        max_tokens=180,
    )

    for text in (cleaned, search_text, read_text):
        assert "E5388" not in text
        assert "E5409" not in text
        assert "DN ON" not in text
        assert "MFC MFC" not in text
        assert "火灾应急处置" in text


def test_retrieval_output_collapses_repeated_phrases_and_alpha_stutter():
    noisy = (
        "脚手架搭设安全规范 " * 16
        + "顶部应安装两层以上防护栏杆。\n"
        + "lll aaa iii ttt nnn eee ddd fff ooo CCC GGG "
        + "高处作业人员需要佩戴安全带。\n"
        + "lll aaa iii ttt nnnOVL(3)控制功能 eee异常处理 ddd fff "
        + "FB exclusion featureGGG 反馈有效期控制。"
    )
    cleaned = clean_retrieval_snippet(noisy)
    assert cleaned.count("脚手架搭设安全规范") == 1
    assert "lll aaa" not in cleaned
    assert "CCC GGG" not in cleaned
    assert "nnnOVL" not in cleaned
    assert "eee异常处理" not in cleaned
    assert "featureGGG" not in cleaned
    assert "顶部应安装两层以上防护栏杆" in cleaned
    assert "高处作业人员需要佩戴安全带" in cleaned
    assert "OVL(3)控制功能" in cleaned
    assert "异常处理" in cleaned
    assert "FB exclusion feature" in cleaned


def test_runner_search_feedback(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "invoice.md").write_text(
        "# OFD 发票\n\nOFD格式电子发票必须提供OFD格式源文件。",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 3})
    meta = {
        "query": "题目：OFD格式电子发票必须提供OFD格式源文件。\n选项：\nA. 对\nB. 错",
        "expected_answer": "[bool] A",
        "num_turns": 0,
        "max_turns": 3,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>OFD格式电子发票 源文件</search>"}],
        meta,
    )
    assert result.terminated is False
    assert result.reward > 0
    assert result.next_stop_strings == ["</search>", "</read>"]
    assert "文档：OFD 发票" in result.observation["content"]
    assert "目录：OFD 发票" in result.observation["content"]
    assert "OFD格式电子发票" not in result.observation["content"]
    assert "命中关键词" not in result.observation["content"]
    assert "本轮查询" not in result.observation["content"]
    assert "OFD格式电子发票 源文件" not in result.observation["content"]
    assert "下一步" in result.observation["content"]
    assert "<read>编号</read>" in result.observation["content"]
    assert result.metadata is not None
    assert result.metadata.get("answer_mode") is not True
    assert result.metadata["last_search_hits"]
    assert result.answer is None


def test_runner_treats_space_variants_as_repeated_search(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spc.md").write_text(
        "# SPC\n\n控制图分析中可以对样本点进行排除处理。",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 5})
    meta = {
        "query": "题目：控制图分析时，以下哪个功能可以永久排除Sample（ ）",
        "expected_answer": "[single] A",
        "num_turns": 0,
        "max_turns": 5,
        "search_history": [],
    }
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>控制图 永久排除样本 功能</search>"}],
        dict(meta),
    )
    assert first.metadata is not None
    first.metadata["answer_mode"] = False
    second = runner.process_turn(
        [{"role": "assistant", "content": "<search>控制图 永久排除 样本 功能</search>"}],
        first.metadata,
    )
    assert second.stats.repeated_queries == 1
    assert second.stats.search_nonempty is False


def test_question_bank_pair_returns_fill_answer(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 1 CDSEM-试卷.md").write_text(
        "一、填空题\n"
        "1. Recipe 建立时需要先完成准备工作。\n"
        "2. SERVER ROOM 通过【1】与Clean room进行连接。\n",
        encoding="utf-8",
    )
    (docs / "Group 1 CDSEM-答案.md").write_text(
        "填空题：\n快速建立OPC Recipe\nSQL server\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    meta = {
        "query": "题目：SERVER ROOM 通过【1】与Clean room进行连接。",
        "expected_answer": "[fill] SQL server",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>SERVER ROOM Clean room 连接</search>"}],
        meta,
    )
    assert result.terminated is False
    assert result.metadata is not None
    assert result.metadata["answer_mode"] is True
    assert result.stats.search_nonempty is True
    assert result.metadata["evidence_quality"] == "strong"
    assert "题目：SERVER ROOM" in result.observation["content"]
    assert "答案：SQL server" in result.observation["content"]
    assert "试卷：Group 1 CDSEM-试卷" not in result.observation["content"]
    assert "答案来源：Group 1 CDSEM-答案" not in result.observation["content"]
    assert "SERVER ROOM Clean room 连接" not in result.observation["content"]


def test_question_bank_search_uses_model_query_not_hidden_full_question(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 1 CDSEM-试卷.md").write_text(
        "一、填空题\n"
        "1. Recipe 建立时需要先完成准备工作。\n"
        "2. SERVER ROOM 通过【1】与Clean room进行连接。\n",
        encoding="utf-8",
    )
    (docs / "Group 1 CDSEM-答案.md").write_text(
        "填空题：\n快速建立OPC Recipe\nSQL server\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    meta = {
        "query": "题目：SERVER ROOM 通过【1】与Clean room进行连接。",
        "expected_answer": "[fill] SQL server",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>完全无关的资料</search>"}],
        meta,
    )
    assert "答案：SQL server" not in result.observation["content"]
    assert result.metadata is not None
    assert result.metadata.get("answer_mode") is not True


def test_question_bank_pair_returns_multiple_answer_not_option_fragment(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group1-DIFF Metrology-试卷.md").write_text(
        "一、单选题\n"
        "1. 量测前应确认机台状态。\n"
        "二、多选题\n"
        "1. 日常点检包含哪些内容？ A. 光刻胶 B. 显影液 C. 水温 D. 记录\n"
        "2. 发现机台处于Down机状态时应（ ） A. 通知设备 B. 通知工艺 C. 切换Standby D. 正常作业监控\n",
        encoding="utf-8",
    )
    (docs / "Group1-DIFF Metrology-答案.md").write_text(
        "答案：\n"
        "1. 1:D\n"
        "2. 1:B C D 2:A B\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    meta = {
        "query": "题目：发现机台处于Down机状态时应（ ）\n选项：\nA. 通知设备\nB. 通知工艺\nC. 切换Standby\nD. 正常作业监控",
        "expected_answer": "[multiple] A,B",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>Down机 通知设备 通知工艺 Standby</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert result.metadata["answer_mode"] is True
    assert "题目：发现机台处于Down机状态时应" in result.observation["content"]
    assert "答案：通知设备；通知工艺" in result.observation["content"]
    assert "试卷：Group1-DIFF Metrology-试卷" not in result.observation["content"]
    assert "答案来源：Group1-DIFF Metrology-答案" not in result.observation["content"]
    assert "答案对应选项" not in result.observation["content"]
    assert "答案：A,B" not in result.observation["content"]
    assert "答案：C,D" not in result.observation["content"]


def test_choice_question_search_returns_regular_docs_alongside_question_bank(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group X certify-试卷.md").write_text(
        "一、单选题\n"
        "1. 关于Auto-2的描述错误的是（ ） A. Queue保留 B. Mode可切换\n",
        encoding="utf-8",
    )
    (docs / "Group X certify-答案.md").write_text(
        "单选题：\n"
        "A\n",
        encoding="utf-8",
    )
    (docs / "Auto模式说明.md").write_text(
        "# Auto 模式说明\n\n"
        "Auto-2下，需要User提前将Lot加到机台的Queue中。\n\n"
        "机台如果降Mode到Auto-1，此时Queue下的Lot不会自动删除，需要人工确认Queue状态。\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4, "max_retrieval_tokens_per_turn": 180})
    meta = {
        "query": "下面是一道单选题。\n题目：以下关于Auto-2的描述，错误的是（ ）\n选项：\nA. Auto-2下，需要User提前将Lot加到机台的Queue中\nB. 降Mode到Auto-1时Queue下的Lot会自动删除",
        "expected_answer": "[single] B",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>Auto-2 Queue Auto-1 自动删除</search>"}],
        meta,
    )
    assert "文档：Auto 模式说明" in result.observation["content"]
    assert "目录：Auto 模式说明" in result.observation["content"]
    assert "不会自动删除" not in result.observation["content"]
    assert result.metadata is not None
    assert result.metadata["last_search_hits"]
    assert any(not ref.get("question_bank", False) for ref in result.metadata["last_search_hits"])


def test_question_bank_does_not_pair_compact_answers_when_count_mismatches(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 9 Safety-试卷.md").write_text(
        "一、填空题\n"
        "1. 第一题需要答案。\n"
        "2. 第二题需要答案。\n"
        "3. 第三题需要答案。\n",
        encoding="utf-8",
    )
    (docs / "Group 9 Safety-答案.md").write_text(
        "填空题：\n"
        "答案一\n"
        "答案二\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    meta = {
        "query": "题目：第三题需要答案。",
        "expected_answer": "[fill] 答案三",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>第三题 需要答案</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert result.metadata["evidence_quality"] == "none"
    assert result.metadata.get("answer_mode") is not True
    assert "答案：答案二" not in result.observation["content"]
    assert "答案二" not in result.observation["content"]
    assert "Group 9 Safety-试卷" not in result.observation["content"]


def test_question_bank_explicit_numbered_answers_only_bind_matching_ordinal(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 8 Ops-试卷.md").write_text(
        "一、单选题\n"
        "1. 第一题是什么（ ） A. Alpha B. Beta\n"
        "2. 第二题是什么（ ） A. Gamma B. Delta\n",
        encoding="utf-8",
    )
    (docs / "Group 8 Ops-答案.md").write_text(
        "答案：\n"
        "1. 2:B\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>第一题 Alpha Beta</search>"}],
        {
            "query": "题目：第一题是什么（ ）\n选项：\nA. Alpha\nB. Beta",
            "expected_answer": "[single] A",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "题目：第一题是什么" not in first.observation["content"]
    assert "答案：Beta" not in first.observation["content"]

    second = runner.process_turn(
        [{"role": "assistant", "content": "<search>第二题 Gamma Delta</search>"}],
        {
            "query": "题目：第二题是什么（ ）\n选项：\nA. Gamma\nB. Delta",
            "expected_answer": "[single] B",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "题目：第二题是什么" in second.observation["content"]
    assert "答案：Delta" in second.observation["content"]
    assert "答案：B" not in second.observation["content"]
    assert "答案对应选项" not in second.observation["content"]
    qbank_id = _first_question_bank_id(second.metadata)
    read = runner.process_turn(
        [{"role": "assistant", "content": f"<read>{qbank_id}</read>"}],
        second.metadata,
    )
    assert f"读取对象：上一轮第 {qbank_id} 条资料" in read.observation["content"]
    assert "内容类型：题库题答详情" not in read.observation["content"]
    assert "答案：Delta" in read.observation["content"]
    assert "答案：B" not in read.observation["content"]
    assert "答案对应选项" not in read.observation["content"]


def test_question_bank_skips_ambiguous_question_answer_filename_pair(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Tool-A1-试卷.md").write_text(
        "一、单选题\n"
        "1. Alpha流程应该选择（ ） A. A1 B. A2\n",
        encoding="utf-8",
    )
    (docs / "Tool-A2-试卷.md").write_text(
        "一、单选题\n"
        "1. Beta流程应该选择（ ） A. B1 B. B2\n",
        encoding="utf-8",
    )
    (docs / "Tool-A-答案.md").write_text(
        "单选题：\n"
        "A\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>Beta流程 B1 B2</search>"}],
        {
            "query": "题目：Beta流程应该选择（ ）\n选项：\nA. B1\nB. B2",
            "expected_answer": "[single] B",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "答案：" not in result.observation["content"]
    assert "答案来源：Tool-A-答案" not in result.observation["content"]
    assert "文档：Tool-A2-试卷" not in result.observation["content"]
    assert result.metadata is not None
    assert result.metadata["evidence_quality"] == "none"
    assert result.metadata.get("answer_mode") is not True


def test_question_bank_does_not_treat_repeated_exam_options_as_answers(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 7 Cert-试卷.md").write_text(
        "一、单选题\n"
        "1. 第一项要求是什么（ ） A. Alpha B. Beta\n"
        "2. 第二项要求是什么（ ） A. Gamma B. Delta\n",
        encoding="utf-8",
    )
    (docs / "Group 7 Cert-答案.md").write_text(
        "Group 7 Cert\n"
        "一、单选题\n"
        "1. 第一项要求是什么（ ） A. Alpha B. Beta\n"
        "2. 第二项要求是什么（ ） A. Gamma B. Delta\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>第二项 Gamma Delta</search>"}],
        {
            "query": "题目：第二项要求是什么（ ）\n选项：\nA. Gamma\nB. Delta",
            "expected_answer": "[single] B",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert "答案：" not in result.observation["content"]
    assert "答案：A" not in result.observation["content"]
    assert "答案：B" not in result.observation["content"]
    assert result.metadata is not None
    assert result.metadata["evidence_quality"] == "none"
    assert result.metadata.get("answer_mode") is not True
    assert "Group 7 Cert-试卷" not in result.observation["content"]


def test_question_answer_same_file_returns_question_and_answer(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "AMAT系列设备培训考试答案.md").write_text(
        "AMAT系列设备培训考试试题\n"
        "1. EBSE20*、EOXE25*的机型名称分别是？（A）\n"
        "A. MESA, Producer GT\n"
        "B. Syndion FS, FlexFSE\n"
        "C. FlexFSE, Syndion FS\n"
        "2. EOXE251有几个chamber？每个chamber有几个head?（C）\n"
        "A. 3, 1\nB. 2, 2\nC. 3, 2\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    meta = {
        "query": "题目：EOXE251有几个chamber？每个chamber有几个head?\n选项：\nA. 3, 1\nB. 2, 2\nC. 3, 2",
        "expected_answer": "[single] C",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>EOXE251 chamber head</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert result.metadata["answer_mode"] is True
    assert "题目：EOXE251有几个chamber" in result.observation["content"]
    assert "答案：3, 2" in result.observation["content"]
    assert "资料：AMAT系列设备培训考试答案" not in result.observation["content"]
    assert "题答位置：同一文件" not in result.observation["content"]
    assert "答案：C" not in result.observation["content"]
    assert "答案对应选项" not in result.observation["content"]


def test_question_bank_parses_answer_tail_when_answer_file_repeats_exam(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group1-DIFF Metrology-试卷.md").write_text(
        "一、单选题\n"
        "1. IMP monitor会用到的量测机台是（ ） A. MXPS201 B. MEPT201 C. MEPD201 D. MTHW201\n"
        "二、多选题\n"
        "1. 机台作业时发生报警该如何处理（ ） A. Abort程序 B. 通知工艺 C. 通知设备 D. 暂停流片\n"
        "2. 发现机台处于Down机状态时应（ ） A. 通知设备 B. 通知工艺 C. 切换机台状态至Standby D. 正常作业监控\n"
        "三、判断题\n"
        "1. 作业过程中量测机台卡住了可直接Abort（ ）\n",
        encoding="utf-8",
    )
    (docs / "Group1-DIFF Metrology-答案.md").write_text(
        "**DIFF Metrology Certify**\n"
        "**一、单选题 （每题4分）**\n"
        "**1. IMP monitor会用到的量测机台是（ ）**\n"
        "**A. MXPS201 B. MEPT201 C. MEPD201 D. MTHW201**\n"
        "**二、多选题（每题4分）**\n"
        "**2. 发现机台处于Down机状态时应（ ）**\n"
        "**A.通知设备 B. 通知工艺 C.切换机台状态至Standby D. 正常作业监控**\n"
        "**三、判断题（每题4分）**\n"
        "**1. 作业过程中量测机台卡住了可直接Abort（ ）**\n"
        "**答案：**\n"
        "1. **1:D**\n"
        "2. **1:B C D 2:A B**\n"
        "**三、1:×**\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    meta = {
        "query": "题目：发现机台处于Down机状态时应（ ）\n选项：\nA. 通知设备\nB. 通知工艺\nC. 切换机台状态至Standby\nD. 正常作业监控",
        "expected_answer": "[multiple] A,B",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>发现机台处于Down机状态时应</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert result.metadata["answer_mode"] is True
    assert "答案：通知设备；通知工艺" in result.observation["content"]
    assert "答案：×" not in result.observation["content"]


def test_short_answer_document_search_returns_neighboring_evidence(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "implant.md").write_text(
        "# 离子注入系统\n\n"
        "离子注入系统用于将指定离子注入晶圆。\n\n"
        "系统组成包括离子源 Ion source、分析磁场 AMU、加速器 ACC。\n\n"
        "后续束线包含聚焦扫描 Focus scan、法拉第 Faraday 和反应室 Process chamber。\n",
        encoding="utf-8",
    )
    runner = QARunner(
        {
            "docs_dir": str(docs),
            "max_turns": 4,
            "max_retrieval_tokens_per_turn": 90,
            "read_radius": 1,
        }
    )
    meta = {
        "query": "下面是一道简答题。\n\n题目：离子注入系统的组成",
        "expected_answer": "[short] 离子源 ||| 分析磁场 ||| 加速器 ||| 聚焦 ||| 法拉第 ||| 反应室",
        "num_turns": 0,
        "max_turns": 4,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>离子注入系统 组成</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert "文档：离子注入系统" in result.observation["content"]
    assert "目录：离子注入系统" in result.observation["content"]
    assert "离子源" not in result.observation["content"]
    read = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        result.metadata,
    )
    assert "离子源" in read.observation["content"]
    assert "法拉第" in read.observation["content"]
    assert "反应室" in read.observation["content"]


def test_short_answer_search_does_not_hit_choice_question_bank(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 2 certify-试卷.md").write_text(
        "一、单选题\n"
        "1. 关于GC Fab2 PETEOS 机台说法错误的是（ ） "
        "A. FTEO201的process location是BM "
        "B. FTEO202的process location是BF "
        "C. FTEO202 monitor后值量测的location是BM\n",
        encoding="utf-8",
    )
    (docs / "Group 2 certify-答案.md").write_text(
        "单选题：\n"
        "A\n",
        encoding="utf-8",
    )
    (docs / "疏散路线图说明.md").write_text(
        "# FAB 疏散路线图\n\n"
        "一级疏散集合点在FAB东侧出口外广场，二级疏散集合点在厂区南门集合区。\n\n"
        "从工作岗位沿主通道离开FAB，经过安全出口和灭火器位置，按疏散指示牌到一级集合点；"
        "如需继续疏散，再沿厂区道路到二级集合点。\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 5, "max_retrieval_tokens_per_turn": 180})
    meta = {
        "query": "下面是一道简答题。\n题目：请画出自己工作岗位到达一级和二级疏散集合点的疏散路线图；可标注FAB 的灭火器等位置",
        "expected_answer": "[short] 一级疏散集合点 ||| 二级疏散集合点 ||| 疏散路线图 ||| 灭火器 ||| FAB",
        "num_turns": 0,
        "max_turns": 5,
    }
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>一级疏散集合点 二级疏散集合点 疏散路线图 灭火器 FAB</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert "答案：" not in result.observation["content"]
    assert "PETEOS" not in result.observation["content"]
    assert "FTEO201" not in result.observation["content"]
    assert "疏散路线图" in result.observation["content"]
    assert "一级疏散集合点" not in result.observation["content"]
    assert "二级疏散集合点" not in result.observation["content"]
    assert result.metadata["last_search_hits"]
    assert not result.metadata["last_search_hits"][0].get("question_bank", False)


def test_runner_read_expands_neighboring_context(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spc.md").write_text(
        "# 控制图分析\n\n"
        "控制图分析页面介绍 SPC 图表和报警。\n\n"
        "Exclude 功能可以永久排除 Sample；控制图分析保存后会从后续计算中移除该 Sample。\n\n"
        "Point Disable 只是临时禁用当前异常点，不能永久排除 Sample。\n\n"
        "最终保存后，该 Sample 不再参与后续控制图计算。",
        encoding="utf-8",
    )
    runner = QARunner(
        {
            "docs_dir": str(docs),
            "max_turns": 6,
            "max_search_attempts": 3,
            "max_read_attempts": 2,
            "max_retrieval_tokens_per_turn": 35,
            "max_read_tokens_per_turn": 120,
            "read_radius": 1,
        }
    )
    meta = {
        "query": "题目：控制图分析时，以下哪个功能可以永久排除Sample（ ）\n选项：\nA. Exclude\nB. Point Disable",
        "expected_answer": "[single] A",
        "num_turns": 0,
        "max_turns": 6,
    }
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>控制图分析 永久排除 Sample</search>"}],
        meta,
    )
    assert first.metadata is not None
    assert first.metadata["last_search_hits"]
    second = runner.process_turn(
        [{"role": "assistant", "content": "<think>第1条需要展开。</think><read>1</read>"}],
        first.metadata,
    )
    assert second.terminated is False
    assert second.stats.valid_read is True
    assert second.stats.read_nonempty is True
    assert "### 深入阅读返回" not in second.observation["content"]
    assert "读取对象：上一轮第 1 条资料" in second.observation["content"]
    assert "内容类型：普通文档上下文" in second.observation["content"]
    assert "资料：控制图分析" in second.observation["content"]
    assert "[命中段]" not in second.observation["content"]
    assert "[上文" not in second.observation["content"]
    assert "[下文" not in second.observation["content"]
    assert "[同文档相关段]" not in second.observation["content"]
    assert "永久排除 Sample" in second.observation["content"]
    assert "Point Disable" in second.observation["content"]
    assert "控制图分析 永久排除 Sample" not in second.observation["content"]
    assert second.metadata is not None
    assert second.metadata["answer_mode"] is True


def test_runner_read_catalog_entry_prioritizes_matching_section_text(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    long_background = " ".join(["控制图基础介绍和数据分布类型"] * 80)
    (docs / "spc.md").write_text(
        "# GC-SPC过程控制教材\n\n"
        f"{long_background}\n\n"
        "控制图菜单说明。Exclude 功能可以永久排除 Sample；Point Disable 只是临时禁用当前点。\n",
        encoding="utf-8",
    )
    runner = QARunner(
        {
            "docs_dir": str(docs),
            "max_turns": 5,
            "max_read_attempts": 2,
            "max_retrieval_tokens_per_turn": 120,
            "max_read_tokens_per_turn": 120,
            "read_radius": 0,
        }
    )
    meta = {
        "query": "题目：控制图分析时，以下哪个功能可以永久排除Sample（ ）\n选项：\nA. Exclude\nB. Point Disable",
        "expected_answer": "[single] A",
        "num_turns": 0,
        "max_turns": 5,
    }
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>控制图 永久排除 Sample</search>"}],
        meta,
    )
    assert "文档：" in first.observation["content"]
    second = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        first.metadata,
    )
    assert "Exclude 功能可以永久排除 Sample" in second.observation["content"]


def test_runner_read_question_bank_result_returns_structured_qa(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Group 1 CDSEM-试卷.md").write_text(
        "一、填空题\n"
        "1. Recipe 建立时需要先完成准备工作。\n"
        "2. SERVER ROOM 通过【1】与Clean room进行连接。\n",
        encoding="utf-8",
    )
    (docs / "Group 1 CDSEM-答案.md").write_text(
        "填空题：\n快速建立OPC Recipe\nSQL server\n",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 5, "max_read_tokens_per_turn": 120})
    meta = {
        "query": "题目：SERVER ROOM 通过【1】与Clean room进行连接。",
        "expected_answer": "[fill] SQL server",
        "num_turns": 0,
        "max_turns": 5,
    }
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>SERVER ROOM Clean room 连接</search>"}],
        meta,
    )
    assert first.metadata is not None
    assert first.metadata["last_search_hits"]
    qbank_id = _first_question_bank_id(first.metadata)
    second = runner.process_turn(
        [{"role": "assistant", "content": f"<read>{qbank_id}</read>"}],
        first.metadata,
    )
    assert second.terminated is False
    assert second.stats.valid_read is True
    assert second.stats.read_nonempty is True
    assert "### 深入阅读返回" not in second.observation["content"]
    assert f"读取对象：上一轮第 {qbank_id} 条资料" in second.observation["content"]
    assert "内容类型：题库题答详情" not in second.observation["content"]
    assert "试卷：Group 1 CDSEM-试卷" not in second.observation["content"]
    assert "答案来源：Group 1 CDSEM-答案" not in second.observation["content"]
    assert "题目：SERVER ROOM" in second.observation["content"]
    assert "答案：SQL server" in second.observation["content"]
    assert "该编号已经包含命中题目和对应答案" not in second.observation["content"]
    assert second.metadata is not None
    assert second.metadata["read_history"] == [f"question_bank#{qbank_id}"]


def test_runner_read_without_previous_search_explains_missing_ref():
    runner = QARunner({"docs_dir": "missing", "max_turns": 3, "max_read_attempts": 2})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        {"query": "题目：任意题", "expected_answer": "[single] A", "num_turns": 0, "max_turns": 3},
    )
    assert result.terminated is False
    assert result.stats.valid_read is True
    assert result.stats.missing_reads == 1
    assert "没有编号 1" in result.observation["content"]
    assert "上一轮实际出现的编号" in result.observation["content"]


def test_runner_read_expands_past_cleaned_empty_chunk():
    dirty = " ".join(f"E{i}" for i in range(5388, 5420))
    chunks = [
        SearchChunk("doc.md", "控制图资料", 0, dirty),
        SearchChunk("doc.md", "控制图资料", 1, "真正可读内容：Exclude 功能可以永久排除 Sample。"),
    ]
    runner = QARunner(
        {
            "docs_dir": "missing",
            "max_turns": 3,
            "max_read_attempts": 2,
            "read_radius": 0,
            "max_read_tokens_per_turn": 180,
        }
    )
    engine = SimpleSearchEngine(chunks=chunks)
    runner.search_engine = engine
    runner.retrieval.search_engine = engine
    result = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        {
            "query": "题目：无关问题",
            "expected_answer": "[single] A",
            "num_turns": 0,
            "max_turns": 3,
            "last_search_hits": [{"id": 1, "doc_path": "doc.md", "chunk_id": 0}],
        },
    )
    assert "真正可读内容" in result.observation["content"]
    assert "Exclude" in result.observation["content"]
    assert "清洗后" not in result.observation["content"]
    assert "没有可展开正文" not in result.observation["content"]
    assert "没有更多可读内容" not in result.observation["content"]


def test_runner_read_scans_farther_when_nearby_chunks_clean_empty():
    dirty = "DN ON " * 80
    chunks = [SearchChunk("doc.md", "消防资料", idx, dirty) for idx in range(12)]
    chunks.append(SearchChunk("doc.md", "消防资料", 12, "可读正文：火灾发生时先确保人员安全，并按现场应急流程处理。"))
    runner = QARunner(
        {
            "docs_dir": "missing",
            "max_turns": 3,
            "max_read_attempts": 2,
            "read_radius": 0,
            "max_read_tokens_per_turn": 220,
        }
    )
    engine = SimpleSearchEngine(chunks=chunks)
    runner.search_engine = engine
    runner.retrieval.search_engine = engine
    result = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        {
            "query": "题目：机台着火了怎么办",
            "expected_answer": "[multiple] A;B",
            "num_turns": 0,
            "max_turns": 3,
            "last_search_hits": [{"id": 1, "doc_path": "doc.md", "chunk_id": 0}],
        },
    )
    assert result.stats.read_nonempty is True
    assert "可读正文" in result.observation["content"]
    assert "火灾发生" in result.observation["content"]
    assert "清洗后" not in result.observation["content"]
    assert "没有可展开正文" not in result.observation["content"]
    assert "没有更多可读内容" not in result.observation["content"]


def test_retrieval_quality_distinguishes_weak_keyword_hit():
    query = "题目：OFD格式电子发票必须提供OFD格式源文件。\n选项：\nA. 对\nB. 错"
    strong = "OFD格式电子发票必须提供OFD格式源文件。"
    weak = "TRIZ OFD DFMEA 正向解决技术冲突的方案。"
    assert retrieval_quality(query, strong) == "strong"
    assert retrieval_quality(query, weak) in {"weak", "none"}


def test_answer_mode_discourages_more_search(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cleanroom.md").write_text(
        "# 无尘室规范\n\n禁止在无尘服、手套、电脑、机台、键盘、文件袋、工具、墙上等写字、乱涂乱画。",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 5})
    meta = {
        "query": "题目：手套、无尘服、电脑、设备、工具等禁止写字乱涂乱画\n选项：\nA. 对\nB. 错",
        "expected_answer": "[bool] A",
        "num_turns": 0,
        "max_turns": 5,
    }
    first = runner.process_turn(
        [{"role": "assistant", "content": "<search>手套 无尘服 电脑 设备 工具 禁止 写字 乱涂乱画</search>"}],
        meta,
    )
    assert first.metadata is not None
    assert first.metadata.get("answer_mode") is not True
    read = runner.process_turn(
        [{"role": "assistant", "content": "<read>1</read>"}],
        first.metadata,
    )
    assert read.metadata is not None
    assert read.metadata["answer_mode"] is True
    second = runner.process_turn(
        [{"role": "assistant", "content": "<search>无尘室 禁止 涂鸦</search>"}],
        read.metadata,
    )
    assert second.terminated is False
    assert second.stats.answer_mode_search is True
    assert "只输出 boxed" in second.observation["content"]
    assert second.next_stop_strings == ["</search>", "</read>"]


def test_no_evidence_after_search_forces_conservative_boxed_fallback():
    runner = QARunner({"docs_dir": "missing", "max_turns": 4, "max_search_attempts": 1})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>安全卫生 注意事项</search>"}],
        {
            "query": (
                "题目：正确的安全卫生注意事项\n"
                "选项：\n"
                "A. 接触化学品应用大量水冲洗20分钟以上\n"
                "B. 清洁机台的粘IPA的无尘布应丢弃红色易燃性垃圾桶\n"
                "C. 触摸未知液体\n"
                "D. 闻到特殊气味不管不顾"
            ),
            "expected_answer": "[multiple] A,B",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert result.metadata is not None
    assert result.metadata["answer_mode"] is True
    assert "没有找到直接相关资料" in result.observation["content"]
    assert "不要继续 search" in result.observation["content"]
    assert "安全卫生" in result.observation["content"]
    assert "保守判断" in result.observation["content"]
    assert "只输出 boxed" in result.observation["content"]


def test_weak_search_feedback_teaches_query_rewrite():
    runner = QARunner({"docs_dir": "missing", "max_turns": 4, "max_search_attempts": 2})
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>私自 by pass 安全设施 处罚</search>"}],
        {
            "query": "题目：私自by pass安全设施，对现场无影响或造成虚惊事件的，季度绩效<B。",
            "expected_answer": "[bool] A",
            "num_turns": 0,
            "max_turns": 4,
        },
    )
    assert result.metadata is not None
    assert result.metadata.get("answer_mode") is not True
    assert "不要重复上一轮" in result.observation["content"]
    assert "删掉具体事件细节" in result.observation["content"]
    assert "核心对象/专有名词" in result.observation["content"]
    assert "合写英文术语" in result.observation["content"]


def test_runner_allows_pre_search_draft_but_penalizes_trailing_text(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "invoice.md").write_text("# OFD 发票\n\nOFD格式电子发票必须提供OFD格式源文件。", encoding="utf-8")
    runner = QARunner({"docs_dir": str(docs), "max_turns": 3})
    meta = {"expected_answer": "[bool] A", "num_turns": 0, "max_turns": 3}
    clean = runner.process_turn(
        [{"role": "assistant", "content": "<search>OFD格式电子发票 源文件</search>"}],
        dict(meta),
    )
    with_draft = runner.process_turn(
        [{"role": "assistant", "content": "我先查一下。<search>OFD格式电子发票 源文件</search>"}],
        dict(meta),
    )
    trailing = runner.process_turn(
        [{"role": "assistant", "content": "<search>OFD格式电子发票 源文件</search>我猜答案是A"}],
        dict(meta),
    )
    assert with_draft.stats.extra_action_text is False
    assert with_draft.reward == clean.reward
    assert trailing.stats.extra_action_text is True
    assert trailing.reward < clean.reward


def test_runner_invalid_then_max_turns():
    runner = QARunner({"docs_dir": "missing", "max_turns": 1})
    meta = {"expected_answer": "[single] A", "num_turns": 0, "max_turns": 1}
    result = runner.process_turn(
        [{"role": "assistant", "content": "我不知道"}],
        meta,
    )
    assert result.terminated is True
    assert result.reward == -1.0
    assert result.metadata is None


def test_validation_split_has_unshaped_rewards(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("资料：A", encoding="utf-8")
    runner = QARunner({"docs_dir": str(docs), "max_turns": 3})
    search_meta = {"expected_answer": "[single] A", "num_turns": 0, "max_turns": 3, "split": "validation"}
    search_result = runner.process_turn(
        [{"role": "assistant", "content": "<search>资料</search>"}],
        search_meta,
    )
    assert search_result.reward == 0.0

    answer_result = runner.process_turn(
        [{"role": "assistant", "content": r"\boxed{B}"}],
        {"expected_answer": "[single] A", "num_turns": 0, "max_turns": 3, "split": "validation"},
    )
    assert answer_result.reward == 0.0


def test_total_retrieval_budget_is_enforced(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "long.md").write_text("# 长文档\n\n" + "控制图分析永久排除Sample。" * 80, encoding="utf-8")
    runner = QARunner(
        {
            "docs_dir": str(docs),
            "max_turns": 4,
            "max_retrieval_tokens_per_turn": 40,
            "max_total_retrieval_tokens": 20,
        }
    )
    meta = {"expected_answer": "[single] A", "num_turns": 0, "max_turns": 4}
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>控制图分析 Sample</search>"}],
        meta,
    )
    assert result.metadata is not None
    assert result.metadata["retrieval_tokens_used"] <= 24


def test_require_docs_fails_fast_for_missing_docs():
    with pytest.raises(FileNotFoundError, match="no searchable"):
        QARunner({"docs_dir": "missing", "require_docs": True})


def test_environment_rejects_mismatched_batches():
    env = QAEnvironment({"docs_dir": "missing"})
    with pytest.raises(ValueError, match="length mismatch"):
        env.step([[{"role": "assistant", "content": r"\boxed{A}"}]], [])
