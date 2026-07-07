from __future__ import annotations

import json
from pathlib import Path

from common.environments.qa_backtest import (
    inspect_message_logs,
    iter_jsonl_objects,
    read_qa_records,
    render_inspect_dsl,
    render_replay_dsl,
    replay_episode,
    summarize_episodes,
    summarize_message_logs,
)
from common.environments.qa_retrieval_env import QARunner


def test_replay_episode_and_summary(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "invoice.md").write_text("# OFD 发票\n\nOFD格式电子发票必须提供OFD格式源文件。", encoding="utf-8")
    runner = QARunner({"docs_dir": str(docs), "max_turns": 4})
    record = {
        "query": "题目：OFD格式电子发票必须提供OFD格式源文件。\n选项：A. 对\nB. 错",
        "expected_answer": "[bool] A",
    }
    episode = replay_episode(
        runner,
        record,
        ["需要查发票源文件要求。<search>OFD格式电子发票 源文件</search>", r"\boxed{A}"],
        split="validation",
    )
    metrics = summarize_episodes([episode])
    assert episode.terminated is True
    assert episode.final_answer == "A"
    assert metrics["accuracy"] == 1.0
    assert metrics["valid_search_rate"] > 0
    dsl = render_replay_dsl([episode], metrics)
    assert dsl.startswith("qa_replay v=1")
    assert "metrics {" in dsl
    assert "episode id=0" in dsl
    assert "turn n=1" in dsl
    assert "assistant:" in dsl


def test_read_qa_records_limit(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "expected_answer": "[single] A"}, ensure_ascii=False),
                json.dumps({"query": "q2", "expected_answer": "[single] B"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    assert len(read_qa_records(data, limit=1)) == 1


def test_summarize_message_logs(tmp_path: Path):
    log = tmp_path / "val.jsonl"
    log.write_text(
        json.dumps(
            {
                "message_log": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "需要查。<search>OFD 发票</search>"},
                    {"role": "environment", "content": "result"},
                    {"role": "assistant", "content": r"\boxed{A}"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = summarize_message_logs(iter_jsonl_objects([log]))
    assert metrics["assistant_messages"] == 2
    assert metrics["search_message_rate"] == 0.5
    assert metrics["boxed_message_rate"] == 0.5
    dsl = render_inspect_dsl(metrics)
    assert dsl.startswith("qa_inspect v=1")
    assert "assistant_messages=2.0000" in dsl


def test_inspect_dsl_includes_bad_output_examples(tmp_path: Path):
    log = tmp_path / "bad_val.jsonl"
    log.write_text(
        json.dumps(
            {
                "message_log": [
                    {"role": "assistant", "content": "我还没想好"},
                    {"role": "assistant", "content": "<search>OFD 发票</search>我猜是A"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics, examples = inspect_message_logs(iter_jsonl_objects([log]), max_examples=2)
    assert metrics["invalid_message_rate"] == 0.5
    assert metrics["trailing_search_text_rate"] == 0.5
    assert metrics["error_no_action_rate"] == 0.5
    dsl = render_inspect_dsl(metrics, examples)
    assert "example n=1 kind=invalid" in dsl
    assert "example n=2 kind=trailing_search_text" in dsl
    assert "assistant:" in dsl
