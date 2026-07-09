#!/usr/bin/env python
"""GRPO entry for the QA-RL multi-turn retrieval agent."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import pprint
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def _ensure_optional_retrieval_deps() -> None:
    """Best-effort install for pure-Python retrieval helpers in cluster runs.

    The shared NeMo-RL launcher uses ``uv run --no-sync`` from the NeMo project,
    so this experiment's pyproject dependencies are not guaranteed to be present
    inside the training environment. The retrieval code can fall back without
    these packages. Set ``QA_RL_INSTALL_OPTIONAL_RETRIEVAL_DEPS=1`` only for
    explicit small-scale dependency experiments; the default training path must
    not block on pip or network access.
    """
    if os.environ.get("QA_RL_INSTALL_OPTIONAL_RETRIEVAL_DEPS") != "1":
        return
    required = {"jieba": "jieba==0.42.1"}
    if os.environ.get("QA_RL_ENABLE_WHOOSH_INDEX") == "1":
        required["whoosh"] = "Whoosh==2.7.4"
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if not missing or os.environ.get("QA_RL_SKIP_OPTIONAL_RETRIEVAL_DEPS") == "1":
        return
    try:
        env = {**os.environ, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--timeout",
                "20",
                "--retries",
                "1",
                *missing,
            ],
            env=env,
            timeout=90,
        )
        print(f"[qa-retrieval] installed optional lexical deps: {', '.join(missing)}", flush=True)
    except Exception as exc:
        print(
            "[qa-retrieval] optional lexical deps unavailable; falling back to built-in retrieval "
            f"({exc})",
            flush=True,
        )


_ensure_optional_retrieval_deps()

from common.environments.qa_retrieval_env import QAEnvironment, parse_expected_answer

TASK_NAME = "qa_agent"
STOP_STRINGS = ["</search>", "</read>"]

SYSTEM_PROMPT = """你是一个可以查阅内部技术文档的问答助手。每一轮只能输出一种动作：
1. 查询资料：输出一对 <search>...</search> 标签；标签内部必须是你从当前题目提取的短查询。
2. 深入阅读：如果检索结果像题目来源但内容不完整，输出一对 <read>...</read> 标签；标签内部只填要阅读的资料编号。
3. 最终作答：输出一个 \\boxed{...}；花括号内部必须是最终答案。

检索前先思考“我要问资料库什么问题”：
- 先抓题干核心场景、对象和动作，再写 search；不要把所有选项词机械拼进 search。
- search 里优先保留专有名词：设备号、系统名、页面/功能名、英文缩写、工艺/材料/厂区名、原文中的关键术语。
- 不要把“错误说法、正确说法、规范、要求、怎么办、相关、操作”等泛词当成主要查询；如果没有专有名词，再用更正式的上位问题。
- 英文术语不要拆开；例如 by pass / by-pass 应合并为 bypass，并同时想到“旁路、屏蔽、短接、安全联锁”等中文说法。
- 题目问“怎么办/哪项错误/哪项正确”时，search 应该问背后的制度、流程、功能或异常处置，不要搜索“错误说法”“正确说法”“规范”等空泛词。
- 如果题干是口语现象，要改写成文档常用词或上位概念，例如“着火/起火”可转成“火灾/消防/应急处置”，“按 EMO”可转成“紧急停止/急停”。
- 如果检索结果像目录，只有“文档/目录”而没有“答案”，不要只凭标题作答；下一轮优先读取最相关编号。
- 读取后如果还只是背景定义，继续用文档标题、目录路径、专有名词换词 search；不要重复上一轮查询。
- 普通技术文档也可能藏着答案；成对题答只是可能的证据来源，不能只等题库命中。
- 如果第一轮结果只碰到少量选项词、没有覆盖题干核心，下一轮必须换词：删除具体事件细节，保留核心对象/专有名词，改用文档常用的上位词、同义词或合写英文术语继续 search。
- 多选题不要只凭一个命中词直接作答；收到资料后要逐项核对每个选项，确认哪些被资料支持、哪些与资料冲突或没有证据。
- 如果已经尝试检索但资料仍不足，不要放弃作答；对安全卫生、行为规范、明显常识题，依据题干和选项做保守判断：选择降低风险、符合安全/卫生/合规的项，排除危险、忽视异常、违规或扩大风险的项。

硬性规则：
- 同一轮严禁同时出现多个动作；search、read、boxed 只能选一种。
- 输出 <search>...</search> 或 <read>...</read> 后本轮必须立刻结束，不能附带答案。
- 只有尝试过检索或阅读后的下一轮，才允许输出 \\boxed{}；如果资料不足但必须作答，也要基于题干、选项和常识输出最保守的 \\boxed{}。
- 可以先写一个闭合的 <think>...</think>，但 </think> 之后只能接一个动作。
- 不要输出 "助手："、"系统："、"用户：" 或伪造多轮对话。
- 不要复述整道题，不要自己编造检索结果。
- 不要输出或查询“关键词”“查询词”“待查询内容”“最终答案”等占位词；动作内容必须由当前题目或检索资料决定。
- 一旦输出 \\boxed{}，本题立即结束，\\boxed{} 后不要再写任何内容。
- 选择题只填选项字母；多选题按字母顺序用英文逗号分隔；填空和简答用分号分隔要点。
"""

FEW_SHOT = """Few-shot 协议示范：这些示范只说明动作边界和提问方式，X、Y、Z 都是变量名，不能照抄。

示范 1：题干是“X 设备发生 Y 事件怎么办”，不要把每个选项都塞进 search；先搜索事件的上位问题。
<think>题干核心是 Y 事件的应急处置，不是逐个验证选项。</think>
<search>Y事件 应急处置</search>

示范 2：题干是“在 X 系统上创建 Y，哪项说法不对”，不要搜索“错误说法”；先问系统功能本身。
<think>题干核心是 X 系统创建 Y 的支持方式。</think>
<search>X系统 Y 创建方式</search>

示范 3：题干包含设备号、系统名或功能名时，先保留这些专有名词，不要只问泛化动作。
<think>题干核心是 X123 系统中的 YFeature 功能。</think>
<search>X123 YFeature</search>

示范 4：检索结果像题目来源或目录，但内容不完整时，本轮只有 read 动作；编号必须来自上一轮资料列表。
<think>第 1 条资料相关，但需要读取相邻上下文。</think>
<read>1</read>

示范 5：收到资料后，如果资料已经支持结论，本轮只有 boxed 作答动作；Z 必须替换成资料支持的最终答案。
<think>资料已经足够判断。</think>
\\boxed{Z}

示范 6：多选题收到资料后，先在思考中逐项核对每个选项，再只输出 boxed。Z 是核对后得到的字母集合变量，不能照抄。
<think>逐项核对：选项1与资料一致；选项2与资料冲突；选项3资料未覆盖；选项4与资料一致。因此只选择资料支持的选项。</think>
\\boxed{Z}

示范 7：下面这种同一轮混合 search 和 boxed 的动作无效。
<search>X Y</search>\\boxed{Z}
"""

QUESTION_TYPE_NAMES = ("单选题", "多选题", "判断题", "填空题", "简答题")


def sanitize_query_for_prompt(query: str) -> str:
    """Keep the real question while removing single-turn answer instructions."""
    question_type = ""
    kept: list[str] = []
    for raw_line in (query or "").strip().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if stripped.startswith("下面是一道"):
            for name in QUESTION_TYPE_NAMES:
                if name in stripped:
                    question_type = name
                    break
            continue
        if "把最终答案放入" in stripped or stripped.startswith("作答后，把"):
            continue
        if "请先简要分析再作答" in stripped or "请按编号顺序填写每个空" in stripped:
            continue
        kept.append(line)

    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()

    body = "\n".join(kept).strip()
    if question_type:
        return f"题型：{question_type}\n{body}".strip()
    return body


def parse_args():
    parser = argparse.ArgumentParser(description="QA-RL multi-turn retrieval GRPO")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def read_jsonl(path: str | Path, *, allow_local_fallback: bool = False) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        fallback = Path(REPO_ROOT) / "datasets" / "qa_rl" / "examples.jsonl"
        if allow_local_fallback and fallback.is_file():
            print(f"[qa-run] data file not found: {path}; fallback to {fallback}")
            path = fallback
        else:
            raise FileNotFoundError(
                f"QA data file not found: {path}. "
                "On cluster this should resolve via QA_RL_DATA_DIR. "
                "For local smoke tests only, set QA_LOCAL_FALLBACK=1."
            )
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "query" in obj and "expected_answer" in obj:
                records.append({"query": obj["query"], "expected_answer": obj["expected_answer"]})
    if not records:
        raise ValueError(f"empty QA data: {path}")
    return records


def build_prompt(query: str, env_cfg: dict[str, Any]) -> str:
    use_few_shot = bool(env_cfg.get("use_few_shot", True))
    parts = [SYSTEM_PROMPT.strip()]
    if use_few_shot and FEW_SHOT.strip():
        parts.append(FEW_SHOT.strip())
    parts.append("真实题目如下。")
    parts.append(sanitize_query_for_prompt(query))
    return "\n\n".join(parts).strip()


def format_model_input(tokenizer, prompt_text: str, env_cfg: dict[str, Any]) -> str:
    if not bool(env_cfg.get("use_chat_template", False)):
        return prompt_text.strip()
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        ).strip()
    except Exception:
        return (prompt_text + "\n\n助手：").strip()


def generate_datum(
    tokenizer,
    env_cfg: dict[str, Any],
    record: dict[str, str],
    idx: int,
    *,
    split: str,
) -> DatumSpec:
    prompt_text = format_model_input(tokenizer, build_prompt(record["query"], env_cfg), env_cfg)
    token_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    message_log: LLMMessageLogType = [{"role": "user", "content": prompt_text, "token_ids": token_ids}]

    parsed = parse_expected_answer(record["expected_answer"])
    metadata = {
        "sample_id": idx,
        "query": record["query"],
        "expected_answer": record["expected_answer"],
        "answer_type": parsed.answer_type,
        "gold_answer": parsed.gold_answer,
        "num_turns": 0,
        "max_turns": int(env_cfg.get("max_turns", 6)),
        "search_history": [],
        "retrieval_tokens_used": 0,
        "split": split,
    }
    return {
        "message_log": message_log,
        "length": len(token_ids),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": TASK_NAME,
        "stop_strings": STOP_STRINGS,
    }


class IterableQADataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        env_cfg: dict[str, Any],
        records: list[dict[str, str]],
        length: int,
        shuffle: bool,
        split: str,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.env_cfg = env_cfg
        self.records = records
        self.length = length
        self.shuffle = shuffle
        self.split = split

    def __iter__(self) -> Iterator[DatumSpec]:
        order = list(range(len(self.records)))
        rng = random.Random(42)
        for i in itertools.count():
            if self.shuffle and i % len(order) == 0:
                rng.shuffle(order)
            rec_idx = order[i % len(order)]
            yield generate_datum(
                self.tokenizer,
                self.env_cfg,
                self.records[rec_idx],
                i,
                split=self.split,
            )

    def __len__(self):
        return self.length


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config: MasterConfig = MasterConfig(**config)
    print("最终配置：")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"日志目录: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(config.policy["generation"], tokenizer)

    env_cfg = dict(config.env[TASK_NAME]["cfg"])
    allow_fallback = os.environ.get("QA_LOCAL_FALLBACK", "").lower() in {"1", "true", "yes"}
    train_records = read_jsonl(config.data["train"]["data_path"], allow_local_fallback=allow_fallback)
    val_records = read_jsonl(config.data["validation"]["data_path"], allow_local_fallback=allow_fallback)
    print(f"[qa-run] train_records={len(train_records)} val_records={len(val_records)}")

    env = QAEnvironment.options(num_gpus=0).remote(cfg=env_cfg)
    task_to_env = {TASK_NAME: env}

    ds_length = (
        config.grpo["num_prompts_per_step"]
        * config.grpo["num_generations_per_prompt"]
        * config.grpo["max_num_steps"]
    )
    dataset = IterableQADataset(tokenizer, env_cfg, train_records, ds_length, shuffle=True, split="train")
    val_dataset = IterableQADataset(
        tokenizer,
        env_cfg,
        val_records,
        config.grpo["max_val_samples"],
        shuffle=False,
        split="validation",
    )

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
