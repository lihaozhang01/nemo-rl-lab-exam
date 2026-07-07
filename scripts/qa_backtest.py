#!/usr/bin/env python
"""Local QA-RL replay and output inspection utilities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.environments.qa_backtest import (  # noqa: E402
    episode_to_jsonable,
    gold_script,
    inspect_message_logs,
    iter_jsonl_objects,
    read_qa_records,
    read_scripted_outputs,
    render_inspect_dsl,
    render_replay_dsl,
    replay_episode,
    summarize_episodes,
)
from common.environments.qa_retrieval_env import QARunner  # noqa: E402


def _print_metrics(metrics: dict[str, float]) -> None:
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def replay_cmd(args: argparse.Namespace) -> int:
    records = read_qa_records(args.data, limit=args.limit)
    scripts = read_scripted_outputs(args.script) if args.script else {}
    runner = QARunner(
        {
            "docs_dir": args.docs,
            "max_turns": args.max_turns,
            "search_top_k": args.search_top_k,
            "max_retrieval_tokens_per_turn": args.max_retrieval_tokens_per_turn,
            "max_total_retrieval_tokens": args.max_total_retrieval_tokens,
        }
    )
    episodes = []
    for idx, record in enumerate(records):
        outputs = scripts.get(idx)
        if outputs is None:
            outputs = gold_script(record, with_search=args.strategy == "gold-search")
        episodes.append(
            replay_episode(
                runner,
                record,
                outputs,
                sample_id=idx,
                split=args.split,
                max_turns=args.max_turns,
            )
        )

    metrics = summarize_episodes(episodes)
    if args.format == "dsl":
        print(render_replay_dsl(episodes, metrics))
    else:
        _print_metrics(metrics)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for episode in episodes:
                f.write(json.dumps(episode_to_jsonable(episode), ensure_ascii=False) + "\n")
        print(f"wrote: {out}")
    return 0


def inspect_cmd(args: argparse.Namespace) -> int:
    metrics, examples = inspect_message_logs(iter_jsonl_objects(args.files), max_examples=args.max_examples)
    if args.format == "dsl":
        print(render_inspect_dsl(metrics, examples))
    else:
        _print_metrics(metrics)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    replay = sub.add_parser("replay", help="Replay scripted outputs through QARunner")
    replay.add_argument("--data", default=str(REPO_ROOT / "datasets" / "qa_rl" / "examples.jsonl"))
    replay.add_argument("--docs", default=str(REPO_ROOT / "datasets" / "qa_rl"))
    replay.add_argument("--script", help="JSONL with {sample_id, outputs}")
    replay.add_argument("--strategy", choices=["gold-search", "gold-direct"], default="gold-search")
    replay.add_argument("--split", choices=["train", "validation"], default="validation")
    replay.add_argument("--limit", type=int)
    replay.add_argument("--max-turns", type=int, default=6)
    replay.add_argument("--search-top-k", type=int, default=4)
    replay.add_argument("--max-retrieval-tokens-per-turn", type=int, default=240)
    replay.add_argument("--max-total-retrieval-tokens", type=int, default=560)
    replay.add_argument("--format", choices=["dsl", "text"], default="dsl")
    replay.add_argument("--json-out")
    replay.set_defaults(func=replay_cmd)

    inspect = sub.add_parser("inspect", help="Summarize assistant outputs from validation JSONL logs")
    inspect.add_argument("files", nargs="+")
    inspect.add_argument("--max-examples", type=int, default=5)
    inspect.add_argument("--format", choices=["dsl", "text"], default="dsl")
    inspect.set_defaults(func=inspect_cmd)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
