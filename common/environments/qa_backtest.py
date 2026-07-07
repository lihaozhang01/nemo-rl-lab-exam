"""Local replay/backtest helpers for the QA-RL environment.

These utilities intentionally avoid NeMo-RL/Ray dependencies. They let us test
the environment state machine with scripted model outputs and inspect validation
JSONL logs after a cluster run.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from common.environments.qa_retrieval_env import QARunner, QATurnStats, parse_action, parse_expected_answer


@dataclass(frozen=True)
class ReplayTurn:
    assistant: str
    observation: str
    reward: float
    terminated: bool
    answer: str | None
    stats: QATurnStats


@dataclass(frozen=True)
class ReplayEpisode:
    sample_id: int
    expected_answer: str
    turns: list[ReplayTurn]
    final_answer: str | None
    final_reward: float
    terminated: bool


@dataclass(frozen=True)
class InspectExample:
    kind: str
    action: str
    error: str
    assistant: str


def read_qa_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "query" in obj and "expected_answer" in obj:
                records.append({"query": obj["query"], "expected_answer": obj["expected_answer"]})
            if limit is not None and len(records) >= limit:
                break
    return records


def read_scripted_outputs(path: str | Path) -> dict[int, list[str]]:
    """Read JSONL scripts: {"sample_id": 0, "outputs": ["...", "..."]}."""
    scripts: dict[int, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            scripts[int(obj["sample_id"])] = [str(x) for x in obj.get("outputs", [])]
    return scripts


def _question_keywords(query: str, max_chars: int = 48) -> str:
    text = re.sub(r"选项：.*", "", query, flags=re.S)
    text = re.sub(r"下面是一道.*?\n", "", text)
    text = re.sub(r"把最终答案.*?\n", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[-max_chars:] if len(text) > max_chars else text


def gold_script(record: dict[str, str], *, with_search: bool = True) -> list[str]:
    parsed = parse_expected_answer(record["expected_answer"])
    if not with_search:
        return [rf"\boxed{{{parsed.gold_answer}}}"]
    keywords = _question_keywords(record["query"])
    return [f"需要查题干中的关键术语。<search>{keywords}</search>", rf"\boxed{{{parsed.gold_answer}}}"]


def replay_episode(
    runner: QARunner,
    record: dict[str, str],
    outputs: list[str],
    *,
    sample_id: int = 0,
    split: str = "validation",
    max_turns: int = 6,
) -> ReplayEpisode:
    parsed = parse_expected_answer(record["expected_answer"])
    metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "query": record["query"],
        "expected_answer": record["expected_answer"],
        "answer_type": parsed.answer_type,
        "gold_answer": parsed.gold_answer,
        "num_turns": 0,
        "max_turns": max_turns,
        "search_history": [],
        "retrieval_tokens_used": 0,
        "split": split,
    }
    message_log: list[dict[str, str]] = [{"role": "user", "content": record["query"]}]
    turns: list[ReplayTurn] = []
    final_answer: str | None = None
    final_reward = 0.0
    terminated = False

    for output in outputs:
        message_log.append({"role": "assistant", "content": output})
        step = runner.process_turn(message_log, metadata)
        turns.append(
            ReplayTurn(
                assistant=output,
                observation=step.observation["content"],
                reward=step.reward,
                terminated=step.terminated,
                answer=step.answer,
                stats=step.stats,
            )
        )
        final_reward += step.reward
        if step.answer is not None:
            final_answer = step.answer
        if step.terminated:
            terminated = True
            break
        metadata = dict(step.metadata or {})
        message_log.append(step.observation)

    return ReplayEpisode(
        sample_id=sample_id,
        expected_answer=record["expected_answer"],
        turns=turns,
        final_answer=final_answer,
        final_reward=final_reward,
        terminated=terminated,
    )


def summarize_episodes(episodes: Iterable[ReplayEpisode]) -> dict[str, float]:
    episodes = list(episodes)
    if not episodes:
        return {}
    total_turns = sum(len(ep.turns) for ep in episodes)
    final_count = sum(1 for ep in episodes if ep.final_answer is not None)
    correct = sum(1 for ep in episodes if ep.final_reward >= 0.999)
    searches = sum(1 for ep in episodes for turn in ep.turns if turn.stats.valid_search)
    nonempty = sum(1 for ep in episodes for turn in ep.turns if turn.stats.search_nonempty)
    extra = sum(1 for ep in episodes for turn in ep.turns if turn.stats.extra_action_text)
    format_errors = sum(1 for ep in episodes for turn in ep.turns if turn.stats.format_error)
    return {
        "episodes": float(len(episodes)),
        "accuracy": correct / len(episodes),
        "boxed_rate": final_count / len(episodes),
        "avg_turns": total_turns / len(episodes),
        "valid_search_rate": searches / max(total_turns, 1),
        "search_nonempty_rate": nonempty / max(searches, 1),
        "extra_action_text_rate": extra / max(total_turns, 1),
        "format_error_rate": format_errors / max(total_turns, 1),
    }


def episode_to_jsonable(ep: ReplayEpisode) -> dict[str, Any]:
    return {
        "sample_id": ep.sample_id,
        "expected_answer": ep.expected_answer,
        "final_answer": ep.final_answer,
        "final_reward": ep.final_reward,
        "terminated": ep.terminated,
        "turns": [
            {
                "assistant": turn.assistant,
                "observation": turn.observation,
                "reward": turn.reward,
                "terminated": turn.terminated,
                "answer": turn.answer,
                "stats": asdict(turn.stats),
            }
            for turn in ep.turns
        ],
    }


def _dsl_quote(value: Any, *, max_chars: int = 220) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return f'"{text}"'


def _dsl_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return _dsl_quote(value)


def render_metrics_dsl(metrics: dict[str, float]) -> str:
    body = " ".join(f"{key}={_dsl_value(metrics[key])}" for key in sorted(metrics))
    return f"metrics {{ {body} }}"


def _turn_flags(stats: QATurnStats) -> list[str]:
    flags = [stats.action]
    if stats.valid_search:
        flags.append("valid_search")
    if stats.search_nonempty:
        flags.append("nonempty")
    if stats.extra_action_text:
        flags.append("trailing_text")
    if stats.format_error:
        flags.append("format_error")
    if stats.boxed:
        flags.append("boxed")
    if stats.final_correct:
        flags.append("correct")
    if stats.timeout:
        flags.append("timeout")
    return flags


def render_replay_dsl(episodes: Iterable[ReplayEpisode], metrics: dict[str, float] | None = None) -> str:
    episodes = list(episodes)
    metrics = metrics if metrics is not None else summarize_episodes(episodes)
    lines = ["qa_replay v=1", render_metrics_dsl(metrics)]
    for ep in episodes:
        lines.append(
            "episode "
            f"id={ep.sample_id} expected={_dsl_quote(ep.expected_answer)} "
            f"final={_dsl_quote(ep.final_answer)} reward={ep.final_reward:.4f} "
            f"terminated={_dsl_value(ep.terminated)} {{"
        )
        for idx, turn in enumerate(ep.turns, 1):
            flags = ",".join(_turn_flags(turn.stats))
            lines.append(
                "  turn "
                f"n={idx} reward={turn.reward:.4f} done={_dsl_value(turn.terminated)} "
                f"flags=[{flags}] answer={_dsl_quote(turn.answer)}"
            )
            lines.append(f"    assistant: {_dsl_quote(turn.assistant)}")
            if turn.observation:
                lines.append(f"    observation: {_dsl_quote(turn.observation)}")
        lines.append("}")
    return "\n".join(lines)


def render_inspect_dsl(metrics: dict[str, float], examples: Iterable[InspectExample] | None = None) -> str:
    lines = ["qa_inspect v=1", render_metrics_dsl(metrics)]
    for idx, example in enumerate(examples or [], 1):
        lines.append(
            "example "
            f"n={idx} kind={example.kind} action={example.action} "
            f"error={_dsl_quote(example.error)} {{"
        )
        lines.append(f"  assistant: {_dsl_quote(example.assistant)}")
        lines.append("}")
    return "\n".join(lines)


def iter_jsonl_objects(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _flatten_message_logs(obj: Any) -> Iterable[list[dict[str, Any]]]:
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) and "role" in x for x in obj):
            yield obj
        else:
            for item in obj:
                yield from _flatten_message_logs(item)
    elif isinstance(obj, dict):
        for key in ("message_log", "messages", "conversation"):
            if key in obj:
                yield from _flatten_message_logs(obj[key])
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _flatten_message_logs(value)


def inspect_message_logs(
    records: Iterable[dict[str, Any]], *, max_examples: int = 5
) -> tuple[dict[str, float], list[InspectExample]]:
    assistant_messages = 0
    search_messages = 0
    boxed_messages = 0
    invalid_messages = 0
    trailing_search_text = 0
    error_counts = {key: 0 for key in ("no_action", "malformed_search", "empty_search", "unclosed_think")}
    examples: list[InspectExample] = []
    for record in records:
        seen_logs: set[int] = set()
        for log in _flatten_message_logs(record):
            log_id = id(log)
            if log_id in seen_logs:
                continue
            seen_logs.add(log_id)
            for msg in log:
                if msg.get("role") != "assistant":
                    continue
                assistant_messages += 1
                content = str(msg.get("content", ""))
                action = parse_action(content)
                search_messages += int(action.action == "search")
                boxed_messages += int(action.action == "answer")
                invalid_messages += int(action.action == "invalid")
                trailing_search_text += int(action.has_extra_text)
                if action.action == "invalid" and action.error in error_counts:
                    error_counts[action.error] += 1
                if max_examples > 0 and len(examples) < max_examples:
                    if action.action == "invalid":
                        examples.append(
                            InspectExample(
                                kind="invalid",
                                action=action.action,
                                error=action.error,
                                assistant=content,
                            )
                        )
                    elif action.has_extra_text:
                        examples.append(
                            InspectExample(
                                kind="trailing_search_text",
                                action=action.action,
                                error="",
                                assistant=content,
                            )
                        )
    denom = max(assistant_messages, 1)
    metrics = {
        "assistant_messages": float(assistant_messages),
        "search_message_rate": search_messages / denom,
        "boxed_message_rate": boxed_messages / denom,
        "invalid_message_rate": invalid_messages / denom,
        "trailing_search_text_rate": trailing_search_text / denom,
    }
    for error, count in error_counts.items():
        metrics[f"error_{error}_rate"] = count / denom
    return metrics, examples


def summarize_message_logs(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    metrics, _examples = inspect_message_logs(records, max_examples=0)
    return metrics
