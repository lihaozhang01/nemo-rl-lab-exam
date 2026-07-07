from __future__ import annotations

from pathlib import Path

import pytest

from common.environments.qa_retrieval_env import (
    QAEnvironment,
    QARunner,
    parse_action,
    parse_expected_answer,
)
from common.environments.search_utils import SimpleSearchEngine
from common.rewards.qa_reward import qa_rule_reward_fn


def test_parse_expected_answer():
    parsed = parse_expected_answer("[multiple] A,B,C")
    assert parsed.answer_type == "multiple"
    assert parsed.gold_answer == "A,B,C"


def test_parse_boxed_takes_precedence_over_search():
    action = parse_action(r"<search>控制图分析</search> 现在确定 \boxed{A}")
    assert action.action == "answer"
    assert action.boxed_answers == ["A"]


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


def test_runner_search_feedback(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "invoice.md").write_text(
        "# OFD 发票\n\nOFD格式电子发票必须提供OFD格式源文件。",
        encoding="utf-8",
    )
    runner = QARunner({"docs_dir": str(docs), "max_turns": 3})
    meta = {"expected_answer": "[bool] A", "num_turns": 0, "max_turns": 3}
    result = runner.process_turn(
        [{"role": "assistant", "content": "<search>OFD格式电子发票 源文件</search>"}],
        meta,
    )
    assert result.terminated is False
    assert result.reward > 0
    assert result.next_stop_strings == ["</search>"]
    assert "OFD格式电子发票" in result.observation["content"]
    assert result.metadata is not None
    assert result.answer is None


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
