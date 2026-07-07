"""Multi-turn retrieval QA environment for the QA-RL exam task."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

try:  # Cluster runtime.
    import ray  # type: ignore
except Exception:  # Local unit tests do not install Ray.
    ray = None

try:  # Cluster runtime.
    import torch  # type: ignore
except Exception:  # Local unit tests do not install PyTorch.
    torch = None

try:  # Cluster runtime.
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn  # type: ignore
except Exception:  # Local unit tests do not install NeMo-RL.
    class EnvironmentInterface:  # type: ignore
        def __class_getitem__(cls, _item):
            return cls

    @dataclass
    class EnvironmentReturn:  # type: ignore
        observations: list[dict[str, str]]
        metadata: list[dict[str, Any] | None]
        next_stop_strings: list[list[str] | None]
        rewards: Any
        terminateds: Any
        answers: list[str | None] | None

from common.environments.search_utils import (
    SimpleSearchEngine,
    clean_query,
    estimate_tokens,
    format_hits,
    split_search_intents,
)
from common.rewards.qa_reward import extract_boxed, qa_rule_reward_fn

BOXED_START_RE = re.compile(r"\\boxed\s*\{")
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
SEARCH_STOP_STRINGS = ["</search>"]


class QAMetadata(TypedDict, total=False):
    sample_id: int
    query: str
    expected_answer: str
    answer_type: str
    gold_answer: str
    num_turns: int
    max_turns: int
    search_history: list[str]
    retrieval_tokens_used: int
    split: str


@dataclass(frozen=True)
class ParsedExpected:
    answer_type: str
    gold_answer: str


@dataclass(frozen=True)
class ActionParseResult:
    action: str
    boxed_answers: list[str]
    search_queries: list[str]
    error: str = ""
    has_extra_text: bool = False


@dataclass(frozen=True)
class QATurnStats:
    action: str
    valid_search: bool = False
    search_nonempty: bool = False
    repeated_queries: int = 0
    empty_queries: int = 0
    boxed: bool = False
    final_correct: bool = False
    answer_type: str = "unknown"
    format_error: bool = False
    timeout: bool = False
    extra_action_text: bool = False


@dataclass(frozen=True)
class QAStepResult:
    observation: dict[str, str]
    reward: float
    terminated: bool
    next_stop_strings: list[str] | None
    metadata: QAMetadata | None
    answer: str | None
    stats: QATurnStats


class QAMetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_steps = 0
        self.valid_searches = 0
        self.nonempty_searches = 0
        self.repeated_queries = 0
        self.empty_queries = 0
        self.boxed_finals = 0
        self.correct_finals = 0
        self.format_errors = 0
        self.timeouts = 0
        self.extra_action_texts = 0

    def update(self, stats: QATurnStats) -> None:
        self.total_steps += 1
        self.valid_searches += int(stats.valid_search)
        self.nonempty_searches += int(stats.search_nonempty)
        self.repeated_queries += stats.repeated_queries
        self.empty_queries += stats.empty_queries
        self.boxed_finals += int(stats.boxed)
        self.correct_finals += int(stats.final_correct)
        self.format_errors += int(stats.format_error)
        self.timeouts += int(stats.timeout)
        self.extra_action_texts += int(stats.extra_action_text)

    def snapshot(self, *, reset: bool = False) -> dict[str, float]:
        denom = max(self.total_steps, 1)
        boxed_denom = max(self.boxed_finals, 1)
        out = {
            "qa/valid_search_rate": self.valid_searches / denom,
            "qa/search_nonempty_rate": self.nonempty_searches / max(self.valid_searches, 1),
            "qa/boxed_format_rate": self.boxed_finals / denom,
            "qa/final_correct_rate": self.correct_finals / boxed_denom,
            "qa/format_error_rate": self.format_errors / denom,
            "qa/timeout_rate": self.timeouts / denom,
            "qa/extra_action_text_rate": self.extra_action_texts / denom,
            "qa/repeated_queries": float(self.repeated_queries),
            "qa/empty_queries": float(self.empty_queries),
        }
        if reset:
            self.reset()
        return out


def parse_expected_answer(raw: str) -> ParsedExpected:
    match = re.match(r"^\s*\[([A-Za-z]+)\]\s*(.*?)\s*$", raw or "")
    if not match:
        return ParsedExpected("unknown", (raw or "").strip())
    return ParsedExpected(match.group(1).lower(), match.group(2).strip())


def _find_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    i = 0
    while True:
        match = BOXED_START_RE.search(text, i)
        if not match:
            break
        pos = match.end()
        depth = 1
        chars: list[str] = []
        while pos < len(text):
            ch = text[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    answers.append("".join(chars).strip())
                    pos += 1
                    break
                chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
        i = max(pos, match.end())
    return answers


def _strip_think_blocks(text: str) -> tuple[str, str]:
    lower = text.lower()
    if "<think>" in lower and "</think>" not in lower:
        return text, "unclosed_think"
    return THINK_RE.sub("", text), ""


def parse_action(text: str, *, max_queries: int = 3, max_query_chars: int = 64) -> ActionParseResult:
    outside, think_error = _strip_think_blocks(text or "")
    if think_error:
        return ActionParseResult("invalid", [], [], think_error)

    boxed = _find_boxed_answers(outside)
    if boxed:
        return ActionParseResult("answer", boxed, [])

    matches = list(SEARCH_RE.finditer(outside))
    raw_searches = [m.group(1) for m in matches]
    if raw_searches:
        trailing_text = outside[matches[-1].end() :].strip()
        has_extra_text = bool(trailing_text)
        queries: list[str] = []
        for raw in raw_searches:
            queries.extend(split_search_intents(raw, max_queries=max_queries, max_chars=max_query_chars))
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            key = clean_query(query, max_chars=max_query_chars).casefold()
            if key and key not in seen:
                seen.add(key)
                deduped.append(query)
            if len(deduped) >= max_queries:
                break
        if deduped:
            return ActionParseResult("search", [], deduped, has_extra_text=has_extra_text)
        return ActionParseResult("invalid", [], [], "empty_search")

    lower = outside.lower()
    if "<search>" in lower or "</search>" in lower:
        return ActionParseResult("invalid", [], [], "malformed_search")
    return ActionParseResult("invalid", [], [], "no_action")


class QARewardPolicy:
    def __init__(self, phase: str = "curriculum"):
        self.phase = phase

    def phase_for_turn(self, metadata: QAMetadata) -> str:
        phase = self.phase
        if phase in {"a", "phase_a", "protocol"}:
            return "a"
        if phase in {"b", "phase_b", "retrieval"}:
            return "b"
        if phase in {"c", "phase_c", "accuracy"}:
            return "c"
        turn = int(metadata.get("num_turns", 0))
        if turn < 2:
            return "a"
        if turn < 5:
            return "b"
        return "c"

    @staticmethod
    def is_validation(metadata: QAMetadata) -> bool:
        return str(metadata.get("split", "train")).lower() in {"val", "valid", "validation"}

    def search_reward(self, stats: QATurnStats, metadata: QAMetadata) -> float:
        if self.is_validation(metadata):
            return 0.0
        phase = self.phase_for_turn(metadata)
        if phase == "a":
            reward = 0.04 + (0.02 if stats.search_nonempty else 0.0)
        elif phase == "b":
            reward = 0.01 - 0.02 * stats.repeated_queries - 0.02 * stats.empty_queries
        else:
            reward = -0.005
        if stats.extra_action_text:
            reward -= 0.02
        return max(min(reward, 0.2), -0.2)

    def format_error_reward(self, error: str, metadata: QAMetadata) -> float:
        if self.is_validation(metadata):
            return 0.0
        return -0.05 if error in {"malformed_search", "empty_search", "unclosed_think"} else -0.1

    def timeout_reward(self, metadata: QAMetadata) -> float:
        return 0.0 if self.is_validation(metadata) else -1.0


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    history: list[str]
    tokens_used: int
    stats: QATurnStats


class RetrievalSession:
    def __init__(
        self,
        search_engine: SimpleSearchEngine,
        *,
        search_top_k: int,
        max_retrieval_tokens_per_turn: int,
        max_total_retrieval_tokens: int,
    ):
        self.search_engine = search_engine
        self.search_top_k = search_top_k
        self.max_retrieval_tokens_per_turn = max_retrieval_tokens_per_turn
        self.max_total_retrieval_tokens = max_total_retrieval_tokens

    def run(self, queries: list[str], metadata: QAMetadata) -> RetrievalResult:
        history = list(metadata.get("search_history", []))
        history_folded = {h.casefold() for h in history}
        used = int(metadata.get("retrieval_tokens_used", 0))
        remaining = max(self.max_total_retrieval_tokens - used, 0)
        if remaining <= 0:
            stats = QATurnStats(action="search", valid_search=True, empty_queries=len(queries))
            return RetrievalResult("", history, used, stats)

        all_hits = []
        repeated = 0
        empty = 0
        for query in queries:
            key = query.casefold()
            if key in history_folded:
                repeated += 1
                continue
            hits = list(self.search_engine.search(query, top_k=self.search_top_k))
            if not hits:
                empty += 1
            all_hits.extend(hits)
            history.append(query)
            history_folded.add(key)

        budget = min(self.max_retrieval_tokens_per_turn, remaining)
        content = format_hits(all_hits, max_tokens=budget)
        tokens_now = estimate_tokens(content)
        stats = QATurnStats(
            action="search",
            valid_search=True,
            search_nonempty=bool(content),
            repeated_queries=repeated,
            empty_queries=empty,
            extra_action_text=bool(metadata.get("_last_action_extra_text", False)),
        )
        return RetrievalResult(content, history, used + tokens_now, stats)


class QARunner:
    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.max_queries_per_turn = int(self.cfg.get("max_queries_per_turn", 3))
        self.max_query_chars = int(self.cfg.get("max_query_chars", 64))
        docs_dir = self.cfg.get("docs_dir", "/data/docs")
        self.search_engine = SimpleSearchEngine(docs_dir)
        if bool(self.cfg.get("require_docs", False)) and not self.search_engine.chunks:
            raise FileNotFoundError(
                f"no searchable markdown/txt docs found in docs_dir={docs_dir!r}; "
                "check that /data/docs is mounted or disable env.qa_agent.cfg.require_docs for local smoke tests"
            )
        self.reward_policy = QARewardPolicy(str(self.cfg.get("reward_phase", "curriculum")))
        self.retrieval = RetrievalSession(
            self.search_engine,
            search_top_k=int(self.cfg.get("search_top_k", 4)),
            max_retrieval_tokens_per_turn=int(self.cfg.get("max_retrieval_tokens_per_turn", 240)),
            max_total_retrieval_tokens=int(self.cfg.get("max_total_retrieval_tokens", 560)),
        )

    def _last_assistant_content(self, message_log: list[dict[str, Any]]) -> str:
        for msg in reversed(message_log or []):
            if msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
        return ""

    def _format_feedback(self, queries: list[str], results_text: str) -> str:
        joined = "；".join(queries)
        if results_text:
            return (
                f"\n### 检索结果返回：\n关于“{joined}”，资料中有这些相关内容：\n\n"
                f"{results_text}\n\n请根据这些资料继续思考；如果还不够确定，可以换一个更具体的关键词继续检索。\n"
            )
        return (
            f"\n### 检索结果返回：\n没有找到和“{joined}”直接相关的内容。\n\n"
            "可以想一想题干里有没有更具体的设备名、系统名、英文缩写或工艺名，再换一个关键词检索。\n"
        )

    def _format_error(self) -> str:
        return (
            "\n你的上一轮输出格式不完整，我没有执行检索。\n"
            "如果要查资料，请输出完整的 <search>关键词</search>。\n"
            "如果已经确定答案，请输出 \\boxed{答案}。\n"
        )

    def _max_turns_reached(self, metadata: QAMetadata) -> bool:
        return int(metadata.get("num_turns", 0)) >= int(metadata.get("max_turns", self.cfg.get("max_turns", 6)))

    def process_turn(
        self,
        message_log: list[dict[str, Any]],
        metadata: QAMetadata,
    ) -> QAStepResult:
        metadata = dict(metadata or {})
        metadata.setdefault("search_history", [])
        metadata.setdefault("retrieval_tokens_used", 0)
        metadata["num_turns"] = int(metadata.get("num_turns", 0)) + 1

        assistant_text = self._last_assistant_content(message_log)
        action = parse_action(
            assistant_text,
            max_queries=self.max_queries_per_turn,
            max_query_chars=self.max_query_chars,
        )

        if action.action == "answer":
            expected_answer = str(metadata.get("expected_answer", ""))
            parsed = parse_expected_answer(expected_answer)
            answer_type = str(metadata.get("answer_type") or parsed.answer_type)
            final_reward = float(
                qa_rule_reward_fn(
                    [str(metadata.get("query", ""))],
                    [assistant_text],
                    [expected_answer],
                )[0]
            )
            prediction = extract_boxed(assistant_text) or action.boxed_answers[-1]
            metadata["grade_reason"] = "common.rewards.qa_reward.qa_rule_reward_fn"
            metadata["prediction"] = prediction
            metadata["gold_answer"] = str(metadata.get("gold_answer") or parsed.gold_answer)
            stats = QATurnStats(
                action="answer",
                boxed=True,
                final_correct=final_reward >= 0.999,
                answer_type=answer_type,
            )
            return QAStepResult(
                observation={"role": "environment", "content": ""},
                reward=float(final_reward),
                terminated=True,
                next_stop_strings=None,
                metadata=None,
                answer=prediction,
                stats=stats,
            )

        if action.action == "search":
            metadata["_last_action_extra_text"] = action.has_extra_text
            retrieval = self.retrieval.run(action.search_queries, metadata)
            metadata.pop("_last_action_extra_text", None)
            metadata["search_history"] = retrieval.history
            metadata["retrieval_tokens_used"] = retrieval.tokens_used
            reward = self.reward_policy.search_reward(retrieval.stats, metadata)
            if self._max_turns_reached(metadata):
                stats = QATurnStats(action="timeout", timeout=True)
                return QAStepResult(
                    observation={"role": "environment", "content": "\n已达到最大轮数，本题未完成作答。\n"},
                    reward=self.reward_policy.timeout_reward(metadata),
                    terminated=True,
                    next_stop_strings=None,
                    metadata=None,
                    answer=None,
                    stats=stats,
                )
            return QAStepResult(
                observation={"role": "environment", "content": self._format_feedback(action.search_queries, retrieval.content)},
                reward=float(reward),
                terminated=False,
                next_stop_strings=SEARCH_STOP_STRINGS,
                metadata=metadata,
                answer=None,
                stats=retrieval.stats,
            )

        reward = self.reward_policy.format_error_reward(action.error, metadata)
        if self._max_turns_reached(metadata):
            stats = QATurnStats(action="timeout", timeout=True, format_error=True)
            return QAStepResult(
                observation={"role": "environment", "content": "\n已达到最大轮数，本题未完成作答。\n"},
                reward=self.reward_policy.timeout_reward(metadata),
                terminated=True,
                next_stop_strings=None,
                metadata=None,
                answer=None,
                stats=stats,
            )
        stats = QATurnStats(action="invalid", format_error=True)
        return QAStepResult(
            observation={"role": "environment", "content": self._format_error()},
            reward=float(reward),
            terminated=False,
            next_stop_strings=SEARCH_STOP_STRINGS,
            metadata=metadata,
            answer=None,
            stats=stats,
        )


class _QAEnvironment(EnvironmentInterface[QAMetadata]):  # type: ignore[misc]
    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.runner = QARunner(cfg or {})
        self.metrics = QAMetricsTracker()

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[QAMetadata],
    ) -> EnvironmentReturn:
        if len(message_log_batch) != len(metadata):
            raise ValueError(f"message_log_batch/metadata length mismatch: {len(message_log_batch)} != {len(metadata)}")
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata, strict=False)
        ]
        observations, rewards, terminateds, stops, next_metadata, answers = [], [], [], [], [], []
        for result in results:
            self.metrics.update(result.stats)
            observations.append(result.observation)
            rewards.append(result.reward)
            terminateds.append(result.terminated)
            stops.append(result.next_stop_strings)
            next_metadata.append(result.metadata)
            answers.append(result.answer)
        rewards_obj = torch.tensor(rewards, dtype=torch.float32) if torch is not None else rewards
        terminateds_obj = torch.tensor(terminateds, dtype=torch.bool) if torch is not None else terminateds
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=stops,
            rewards=rewards_obj,
            terminateds=terminateds_obj,
            answers=answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(self, batch):
        qa_metrics = self.metrics.snapshot(reset=True)
        if torch is None:
            return batch, {"accuracy": 0.0, **qa_metrics}
        final_rewards = batch.get("total_reward", torch.tensor([0.0] * len(batch["idx"])))
        accuracy = ((final_rewards >= 0.9).float().mean().item()) if len(final_rewards) > 0 else 0.0
        avg_reward = final_rewards.float().mean().item() if len(final_rewards) > 0 else 0.0
        return batch, {"accuracy": accuracy, "avg_total_reward": avg_reward, **qa_metrics}

    def metrics_snapshot(self, reset: bool = False) -> dict[str, float]:
        return self.metrics.snapshot(reset=reset)


if ray is not None:  # pragma: no cover - exercised in cluster runtime.
    QAEnvironment = ray.remote(_QAEnvironment)
else:
    QAEnvironment = _QAEnvironment
