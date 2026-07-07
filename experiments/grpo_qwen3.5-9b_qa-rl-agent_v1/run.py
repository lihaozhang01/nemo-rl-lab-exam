#!/usr/bin/env python
"""GRPO entry for the QA-RL multi-turn retrieval agent."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pprint
import random
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

from common.environments.qa_retrieval_env import QAEnvironment, parse_expected_answer

TASK_NAME = "qa_agent"
STOP_STRINGS = ["</search>"]

SYSTEM_PROMPT = """你是一个可以查阅内部技术文档的问答助手。请严格遵守协议：
1. 如果需要查询资料，可以先用一句短草稿说明要查什么，然后必须输出完整检索标签：<search>关键词</search>
2. 系统返回检索结果后，你可以继续搜索，也可以作答。
3. 当你确定最终答案后，必须输出：\\boxed{答案}
4. 一旦输出 \\boxed{}，本题立即结束。不要在 \\boxed{} 后继续输出内容。
5. 选择题只填选项字母；多选题按字母顺序用英文逗号分隔；填空和简答用分号分隔要点。
6. 输出 <search>...</search> 后，本轮立即结束；不要预测检索结果、不要继续推演答案。
"""

FEW_SHOT = """示例一：
题目：示例设备通过什么系统与示例区域连接？
助手：需要查设备、区域和连接系统。<search>示例设备与示例区域连接系统</search>
系统：
### 检索结果返回：
资料显示：示例设备通过 Alpha Link 系统与示例区域连接。
助手：资料说明连接方式是 Alpha Link。 \\boxed{Alpha Link}

示例二：
题目：示例流程必须先完成记录归档。A. 对 B. 错
助手：需要查流程开始前的归档要求。<search>示例流程;记录归档要求</search>
系统：
### 检索结果返回：
资料显示：示例流程开始前必须先完成记录归档。
助手：资料明确说明该说法正确。 \\boxed{A}
"""


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
    if use_few_shot:
        parts.append(FEW_SHOT.strip())
    parts.append("现在请回答下面的真实题目。")
    parts.append(query.strip())
    return "\n\n".join(parts).strip()


def apply_prompt_template(tokenizer, prompt_text: str) -> str:
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
    prompt_text = apply_prompt_template(tokenizer, build_prompt(record["query"], env_cfg))
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
