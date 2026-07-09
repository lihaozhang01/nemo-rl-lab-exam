"""Multi-turn retrieval QA environment for the QA-RL exam task."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
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
    DocumentSectionHit,
    EN_TOKEN_RE,
    SearchHit,
    SimpleSearchEngine,
    clean_query,
    clean_retrieval_snippet,
    count_evidence_score,
    cjk_ngrams,
    duplicate_text_key,
    estimate_tokens,
    format_structured_read_context,
    format_question_bank_item_text,
    is_quantity_query,
    is_near_duplicate_text,
    _looks_like_toc,
    normalize_text,
    split_search_intents,
    trim_to_token_budget,
)
from common.rewards.qa_reward import extract_boxed, qa_rule_reward_fn

BOXED_START_RE = re.compile(r"\\boxed\s*\{")
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
READ_RE = re.compile(r"<read>(.*?)</read>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ACTION_STOP_STRINGS = ["</search>", "</read>"]
SEARCH_STOP_STRINGS = ACTION_STOP_STRINGS
WEAK_SEARCH_STOP_STRINGS = ACTION_STOP_STRINGS

ALPHA_STUTTER_FRAGMENT_RE = re.compile(r"([A-Za-z])\1{2,}")
RESULT_ITEM_INLINE_BOUNDARY_RE = re.compile(r"([^\n])(\d{1,2}\.\s*(?:文档|资料|试卷|题目)[:：])")
RESULT_ITEM_START_RE = re.compile(r"^\s*\d{1,2}\.\s*(?:文档|资料|试卷|题目)[:：]")
RESULT_GUIDANCE_START_RE = re.compile(
    r"^\s*(?:下一步|没有找到|读取失败|动作无效|已有资料|已达到最大轮数|已经读过)"
)
QUESTION_PREFIX_RE = re.compile(r"题目[:：]\s*(.*?)(?:\n\s*选项[:：]|\Z)", re.DOTALL)
OPTION_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$", re.MULTILINE)
COMPACT_RE = re.compile(r"[\W_]+", re.UNICODE)
GENERIC_DOC_TITLE_RE = re.compile(
    r"^(?:page|slide)?\s*\d+$|^(?:doc|docs|document|manual|index|content|contents|内部文档片段|description)$",
    re.IGNORECASE,
)
LEADING_DOC_NUMBER_RE = re.compile(
    r"^\s*(?:[【\[]\s*\d+(?:\.\d+){0,5}\s*[】\]]\s*|"
    r"\d+(?:\.\d+){1,5}(?:-\d+)?(?:\s+|[-_、．)]\s*))"
)
APPROVAL_QUERY_RE = re.compile(r"审批|批准|审核|签核|签字|许可|作业票|到谁|谁可以|谁才可以|负责人|工程师", re.IGNORECASE)
APPROVAL_EVIDENCE_TERMS = {
    "审批",
    "批准",
    "审核",
    "签核",
    "签字",
    "许可",
    "EHS",
    "工程师",
    "部门负责人",
    "负责人",
    "主管",
}
NOISE_TERMS = {
    "下面",
    "一道",
    "单选",
    "多选",
    "判断",
    "填空",
    "简答",
    "题目",
    "选项",
    "答案",
    "正确",
    "错误",
    "说法",
    "作答",
    "分析",
    "唯一",
    "所有",
    "应该",
    "怎么",
    "么办",
    "怎么办",
    "操作",
    "通知",
    "相关",
    "支持",
    "clean",
    "room",
    "page",
}


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
    answer_mode: bool
    search_attempts: int
    read_attempts: int
    read_history: list[str]
    last_search_hits: list[dict[str, Any]]
    evidence_quality: str
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
    mixed_action: bool = False
    read_ids: tuple[int, ...] = ()


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
    answer_mode_search: bool = False
    valid_read: bool = False
    read_nonempty: bool = False
    missing_reads: int = 0
    evidence_quality: str = "unknown"


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
        self.answer_mode_searches = 0
        self.valid_reads = 0
        self.nonempty_reads = 0
        self.missing_reads = 0

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
        self.answer_mode_searches += int(stats.answer_mode_search)
        self.valid_reads += int(stats.valid_read)
        self.nonempty_reads += int(stats.read_nonempty)
        self.missing_reads += stats.missing_reads

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
            "qa/answer_mode_search_rate": self.answer_mode_searches / denom,
            "qa/valid_read_rate": self.valid_reads / denom,
            "qa/read_nonempty_rate": self.nonempty_reads / max(self.valid_reads, 1),
            "qa/missing_reads": float(self.missing_reads),
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


def _question_text(query: str) -> str:
    match = QUESTION_PREFIX_RE.search(query or "")
    if match:
        return normalize_text(match.group(1))
    return normalize_text(query)


def _option_texts(query: str) -> list[str]:
    return [normalize_text(m.group(2)) for m in OPTION_RE.finditer(query or "")]


def _visible_answer_type(query: str) -> str:
    text = normalize_text(query)
    if "多选题" in text or "选出所有正确" in text:
        return "multiple"
    if "判断题" in text or "判断下面说法是否正确" in text:
        return "bool"
    if "填空题" in text:
        return "fill"
    if "简答题" in text:
        return "short"
    if "单选题" in text or "选出唯一正确" in text:
        return "single"
    return ""


def _signal_terms(text: str) -> set[str]:
    text = normalize_text(text)
    terms = {t.casefold() for t in EN_TOKEN_RE.findall(text) if len(t) >= 2}
    terms.update(g for g in cjk_ngrams(text, min_n=2, max_n=4) if len(g) >= 2)
    if any(term in text for term in ("着火", "起火", "火灾", "火警")):
        terms.update({"着火", "起火", "火灾", "火警", "消防", "灭火", "应急", "应急处置"})
    if "emo" in text.casefold() or any(term in text for term in ("紧急停止", "急停")):
        terms.update({"emo", "紧急停止", "急停"})
    return {t for t in terms if t and t not in NOISE_TERMS}


def _compact(text: str) -> str:
    return COMPACT_RE.sub("", normalize_text(text).casefold())


def _option_specific_hit_count(query: str, retrieval_text: str) -> int:
    retrieval_compact = _compact(retrieval_text)
    retrieval_terms = _signal_terms(retrieval_text)
    count = 0
    for option in _option_texts(query):
        option_compact = _compact(option)
        if len(option_compact) >= 2 and option_compact in retrieval_compact:
            count += 1
            continue
        option_terms = _signal_terms(option)
        strong_terms = {
            term
            for term in option_terms
            if term
            and term not in NOISE_TERMS
            and (len(term) >= 3 or re.fullmatch(r"[A-Z]{2,}", term, flags=re.IGNORECASE))
        }
        if strong_terms and strong_terms & retrieval_terms:
            count += 1
    return count


def _needs_answer_bearing_evidence(query: str) -> bool:
    question = _question_text(query)
    return bool(APPROVAL_QUERY_RE.search(question))


def _has_answer_bearing_evidence(query: str, retrieval_text: str) -> bool:
    if not _needs_answer_bearing_evidence(query):
        return True
    retrieval_compact = _compact(retrieval_text)
    has_approval_term = any(_compact(term) in retrieval_compact for term in APPROVAL_EVIDENCE_TERMS)
    return has_approval_term and _option_specific_hit_count(query, retrieval_text) > 0


def _is_question_bank_lookup_query(query: str) -> bool:
    """Avoid letting a single split keyword pull in a loosely related exam item."""
    terms = _signal_terms(query)
    if len(terms) < 2:
        return False
    compact = _compact(query)
    if len(compact) >= 10:
        return True
    strong_terms = [
        term
        for term in terms
        if len(term) >= 3 or re.fullmatch(r"[A-Z]{2,}\d*", term, flags=re.IGNORECASE)
    ]
    return len(strong_terms) >= 2


def _is_exam_like_source(path: str, title: str = "") -> bool:
    source = f"{Path(path).stem} {title}"
    return any(marker in source for marker in ("试卷", "试题", "考试", "考核"))


def _is_generic_doc_title(title: str) -> bool:
    title = normalize_text(title)
    if not title:
        return True
    if GENERIC_DOC_TITLE_RE.fullmatch(title):
        return True
    if len(title) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z]", title):
        return True
    return False


def _strip_leading_doc_number(title: str) -> str:
    title = normalize_text(title)
    for _ in range(3):
        stripped = LEADING_DOC_NUMBER_RE.sub("", title, count=1).strip(" -_、．)）")
        if stripped == title:
            break
        title = stripped
    return title


def _canonical_doc_title(title: str) -> str:
    title = normalize_text(title)
    stripped = _strip_leading_doc_number(title)
    if stripped and not _is_generic_doc_title(stripped) and len(_compact(stripped)) >= 4:
        return stripped
    return title


def _same_canonical_title(left: str, right: str) -> bool:
    left_key = _compact(_canonical_doc_title(left))
    right_key = _compact(_canonical_doc_title(right))
    return bool(left_key and right_key and left_key == right_key)


def _prefer_catalog_title(current: str, candidate: str) -> str:
    current = normalize_text(current)
    candidate = normalize_text(candidate)
    if not candidate:
        return current
    if not current or _is_generic_doc_title(current):
        return candidate
    if _same_canonical_title(current, candidate):
        current_clean = _canonical_doc_title(current)
        candidate_clean = _canonical_doc_title(candidate)
        current_has_prefix = _compact(current_clean) != _compact(current)
        candidate_has_prefix = _compact(candidate_clean) != _compact(candidate)
        if current_has_prefix and not candidate_has_prefix:
            return candidate
        if candidate_has_prefix and not current_has_prefix:
            return current
        if len(candidate_clean) < len(current_clean):
            return candidate_clean
    return current


def _document_group_key(doc_path: str, title: str) -> str:
    canonical_title = _canonical_doc_title(title)
    compact_title = _compact(canonical_title)
    if compact_title and not _is_generic_doc_title(canonical_title) and len(compact_title) >= 4:
        return f"title:{compact_title}"
    return f"path:{doc_path}"


def _catalog_path_label(title: str, section_path: tuple[str, ...], fallback: str = "") -> str:
    title = normalize_text(title)
    parts = [normalize_text(part) for part in section_path if normalize_text(part)]
    if parts and title and _same_canonical_title(parts[0], title):
        parts[0] = _prefer_catalog_title(parts[0], title)
    if title and (not parts or not _same_canonical_title(parts[0], title)):
        parts.insert(0, title)
    if not parts and fallback:
        parts = [normalize_text(fallback)]
    return " > ".join(part for part in parts if part)


def _doc_title_and_section(hit: SearchHit) -> tuple[str, str]:
    doc_title = normalize_text(Path(hit.chunk.doc_path).stem)
    heading = normalize_text(hit.chunk.heading)
    if (
        doc_title
        and heading
        and _compact(doc_title) == _compact(heading)
        and len(heading) > len(doc_title)
        and not _is_generic_doc_title(heading)
    ):
        title = heading
        section = ""
    elif _is_generic_doc_title(doc_title) and not _is_generic_doc_title(heading):
        title = heading
        section = ""
    else:
        title = doc_title if not _is_generic_doc_title(doc_title) else "内部文档"
        section = "" if _is_generic_doc_title(heading) or heading == title else heading
    return title, section


def _format_catalog_results(
    queries: list[str],
    catalog_hits: list[DocumentSectionHit],
    *,
    max_tokens: int,
    max_results: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Return document titles and multi-level section paths, not body snippets."""
    groups: dict[str, dict[str, Any]] = {}
    for hit in catalog_hits:
        section = hit.section
        title = normalize_text(section.doc_title or Path(section.doc_path).stem)
        group_key = _document_group_key(section.doc_path, title)
        group = groups.setdefault(
            group_key,
            {
                "title": title,
                "score": 0.0,
                "matched": set(),
                "sections": [],
                "hits": [],
            },
        )
        group["title"] = _prefer_catalog_title(str(group["title"]), title)
        path = tuple(normalize_text(part) for part in section.section_path if normalize_text(part))
        section_label = _catalog_path_label(str(group["title"]), path, section.section_title)
        matched = set(hit.matched_terms)
        group["score"] = max(float(group["score"]), hit.score)
        group["matched"].update(matched)
        group["hits"].append((len(matched), len(path), hit.score, hit, section_label))

    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (len(group["matched"]), float(group["score"]), len(group["hits"])),
        reverse=True,
    )

    remaining = max_tokens
    parts: list[str] = []
    refs: list[dict[str, Any]] = []
    seen_visible_text_keys: list[str] = []
    next_id = 1
    item_cap = min(max(max_tokens // 2, 120), 260)
    for group in ranked_groups:
        if not group["hits"]:
            continue
        best_hits = sorted(group["hits"], key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, _, best_hit, best_section = best_hits[0]
        title = normalize_text(str(group["title"]))
        lines = [f"文档：{title}"]
        if best_section:
            lines.append(f"目录：{best_section}")
        text = "\n".join(lines).strip()
        if is_near_duplicate_text(text, seen_visible_text_keys):
            continue
        prefix = f"{next_id}. "
        if remaining <= estimate_tokens(prefix) + 8:
            break
        body_budget = min(max(remaining - estimate_tokens(prefix), 8), item_cap)
        if estimate_tokens(text) > body_budget:
            text = trim_to_token_budget(text, body_budget)
        numbered = f"{prefix}{text}"
        parts.append(numbered)
        refs.append(
            {
                "id": next_id,
                "doc_path": best_hit.section.doc_path,
                "chunk_id": best_hit.section.chunk_id,
                "catalog_entry": True,
                "doc_title": title,
                "section": best_section,
                "section_path": list(best_hit.section.section_path),
            }
        )
        seen_visible_text_keys.append(duplicate_text_key(text))
        remaining -= estimate_tokens(numbered)
        next_id += 1
        if len(refs) >= max_results:
            break
    return "\n".join(parts), refs


def _clean_visible_feedback_text(text: str) -> str:
    text = RESULT_ITEM_INLINE_BOUNDARY_RE.sub(r"\1\n\2", text or "")
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if len(ALPHA_STUTTER_FRAGMENT_RE.findall(line)) >= 5:
            line = ALPHA_STUTTER_FRAGMENT_RE.sub("", line)
            line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)
    return _dedupe_repeated_result_items("\n".join(lines)).strip()


def _dedupe_repeated_result_items(text: str) -> str:
    out: list[str] = []
    current: list[str] = []
    seen_item_keys: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        item_text = "\n".join(current).strip()
        key_text = re.sub(r"^\s*\d{1,2}\.\s*", "", item_text, count=1)
        if not is_near_duplicate_text(key_text, seen_item_keys, threshold=0.98):
            out.extend(current)
            seen_item_keys.append(duplicate_text_key(key_text))
        current = []

    for line in (text or "").splitlines():
        if RESULT_ITEM_START_RE.match(line):
            flush_current()
            current = [line]
        elif current and line.strip() and not RESULT_GUIDANCE_START_RE.match(line):
            current.append(line)
        else:
            flush_current()
            out.append(line)
    flush_current()
    return "\n".join(out)


def _looks_like_question_only_retrieval(text: str) -> bool:
    raw_text = text or ""
    normalized = normalize_text(raw_text)
    if "题目：" in raw_text and "答案：" in raw_text:
        return False
    source_titles = re.findall(r"(?:^|\n)\d+\.\s*(?:资料|试卷)[:：]\s*([^\n]+)", raw_text)
    if source_titles and any(not any(marker in title for marker in ("试卷", "试题", "考试", "考核")) for title in source_titles):
        return False
    text = normalized
    has_exam_source = any(marker in text for marker in ("试卷", "试题", "考试", "考核"))
    has_question_shape = any(marker in text for marker in ("单选题", "多选题", "判断题", "填空题", "简答题", " A.", " B.", "( )", "（ ）"))
    return has_exam_source and has_question_shape


def _looks_like_catalog_retrieval(text: str) -> bool:
    raw = text or ""
    normalized = normalize_text(raw)
    has_catalog = (("文档：" in raw or "文档:" in normalized) and ("目录：" in raw or "目录:" in normalized))
    return has_catalog and not ("题目：" in raw and "答案：" in raw) and "读取对象" not in normalized


def retrieval_quality(query: str, retrieval_text: str) -> str:
    """Classify retrieval relevance without using expected answers."""
    if not retrieval_text.strip():
        return "none"
    if _looks_like_catalog_retrieval(retrieval_text):
        return "weak"
    if _looks_like_question_only_retrieval(retrieval_text):
        return "weak"
    question_text = _question_text(query)
    has_question_bank_answer = "题目：" in retrieval_text and "答案：" in retrieval_text
    question_compact = _compact(question_text)
    retrieval_compact = _compact(retrieval_text)
    if len(question_compact) >= 8 and question_compact in retrieval_compact:
        return "strong"

    retrieval_terms = _signal_terms(retrieval_text)
    if not retrieval_terms:
        return "weak"

    question_terms = _signal_terms(question_text)
    option_term_sets = [terms for terms in (_signal_terms(opt) for opt in _option_texts(query)) if terms]
    question_hits = question_terms & retrieval_terms
    option_hits = [terms & retrieval_terms for terms in option_term_sets if terms]
    option_hit_count = sum(1 for hits in option_hits if hits)
    option_specific_hit_count = _option_specific_hit_count(query, retrieval_text)

    if is_quantity_query(question_text) and not has_question_bank_answer:
        count_score = count_evidence_score(question_text, text=retrieval_text)
        if count_score < 5.0:
            return "weak" if question_hits else "none"
    if not has_question_bank_answer and not _has_answer_bearing_evidence(query, retrieval_text):
        return "weak" if question_hits or option_hit_count else "none"

    # Strong means the returned text looks like it covers the actual question,
    # and for choice questions it also contains at least one option-specific clue.
    if len(question_hits) >= 2 and (not option_term_sets or option_specific_hit_count > 0):
        return "strong"
    if question_hits and option_specific_hit_count >= 2:
        return "strong"
    if len(question_hits) >= 1 or any(option_hits):
        return "weak"
    return "none"


def _format_ranked_search_results(
    queries: list[str],
    bank_hits: list[Any],
    catalog_hits: list[DocumentSectionHit],
    *,
    max_tokens: int,
    max_results: int = 6,
    max_doc_snippets: int = 2,
) -> tuple[str, list[dict[str, Any]]]:
    remaining = max_tokens
    parts: list[str] = []
    refs: list[dict[str, Any]] = []
    next_id = 1
    seen_visible_text_keys: list[str] = []

    query_terms: set[str] = set()
    for query in queries:
        query_terms.update(_signal_terms(query))

    def relevance_score(text: str, base_score: float = 0.0) -> tuple[int, float, int]:
        text_terms = _signal_terms(text)
        matched = query_terms & text_terms if query_terms else set()
        return (len(matched), base_score, estimate_tokens(text))

    def append_part(text: str, ref: dict[str, Any], *, max_item_tokens: int | None = None) -> bool:
        nonlocal remaining, next_id
        text = text.strip()
        if not text:
            return False
        if is_near_duplicate_text(text, seen_visible_text_keys):
            return False
        prefix = f"{next_id}. "
        if remaining <= estimate_tokens(prefix) + 8:
            return False
        body_budget = max(remaining - estimate_tokens(prefix), 8)
        if max_item_tokens is not None:
            body_budget = min(body_budget, max(max_item_tokens, 8))
        if estimate_tokens(prefix + text) > remaining or estimate_tokens(text) > body_budget:
            text = trim_to_token_budget(text, body_budget)
        numbered = f"{prefix}{text}"
        parts.append(numbered)
        refs.append({"id": next_id, **ref})
        seen_visible_text_keys.append(duplicate_text_key(text))
        remaining -= estimate_tokens(numbered)
        next_id += 1
        return True

    candidates: list[tuple[tuple[int, float, int], str, dict[str, Any]]] = []
    seen_bank: set[tuple[str, str, int]] = set()
    for hit in bank_hits:
        item = hit.item
        bank_key = (item.question_path, item.answer_path, item.ordinal)
        if bank_key in seen_bank:
            continue
        seen_bank.add(bank_key)
        text = format_question_bank_item_text(item)
        candidates.append(
            (
                relevance_score(f"{item.question} {item.answer_detail or item.answer}", hit.score),
                text,
                {
                "doc_path": item.answer_path or item.question_path,
                "chunk_id": 0,
                "question_bank": True,
                "question_path": item.question_path,
                "answer_path": item.answer_path,
                "question": normalize_text(item.question),
                "answer": normalize_text(item.answer),
                "answer_detail": item.answer_detail,
                "same_file": item.answer_path == item.question_path,
                },
            )
        )

    if not candidates:
        return _format_catalog_results(
            queries,
            catalog_hits,
            max_tokens=max_tokens,
            max_results=max_results,
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    ranked = list(enumerate(candidates))
    ordered_indices = [idx for idx, _ in ranked]

    item_cap = min(max(max_tokens // 2, 160), 320)
    doc_counts: dict[str, int] = {}
    for idx in ordered_indices:
        _, text, ref = candidates[idx]
        if not ref.get("question_bank"):
            doc_key = str(ref.get("doc_path", ""))
            doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            if doc_counts[doc_key] > max_doc_snippets:
                continue
        if not append_part(text, ref, max_item_tokens=item_cap):
            break
        if len(refs) >= max_results:
            break
    return "\n".join(parts), refs


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
    # Qwen-family base models often start an unclosed <think>. Closed blocks are
    # hidden from action parsing, but an unclosed tag should not make us miss a
    # later valid search/boxed action.
    without_closed = THINK_RE.sub("", text)
    without_tags = re.sub(r"</?think>", "", without_closed, flags=re.IGNORECASE)
    return without_tags, ""


def parse_action(text: str, *, max_queries: int = 3, max_query_chars: int = 64) -> ActionParseResult:
    outside, think_error = _strip_think_blocks(text or "")
    if think_error:
        return ActionParseResult("invalid", [], [], think_error)

    boxed = _find_boxed_answers(outside)
    matches = list(SEARCH_RE.finditer(outside))
    raw_searches = [m.group(1) for m in matches]
    read_matches = list(READ_RE.finditer(outside))
    raw_reads = [m.group(1) for m in read_matches]
    action_count = int(bool(boxed)) + int(bool(raw_searches)) + int(bool(raw_reads))
    if action_count > 1:
        return ActionParseResult("invalid", boxed, [], "mixed_action", mixed_action=True)
    if boxed:
        return ActionParseResult("answer", boxed, [])

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

    if raw_reads:
        trailing_text = outside[read_matches[-1].end() :].strip()
        read_ids: list[int] = []
        seen_ids: set[int] = set()
        for raw in raw_reads:
            for token in re.findall(r"\d+", raw):
                read_id = int(token)
                if read_id > 0 and read_id not in seen_ids:
                    seen_ids.add(read_id)
                    read_ids.append(read_id)
        if read_ids:
            return ActionParseResult("read", [], [], has_extra_text=bool(trailing_text), read_ids=tuple(read_ids[:2]))
        return ActionParseResult("invalid", [], [], "empty_read")

    lower = outside.lower()
    if "<search>" in lower or "</search>" in lower:
        return ActionParseResult("invalid", [], [], "malformed_search")
    if "<read>" in lower or "</read>" in lower:
        return ActionParseResult("invalid", [], [], "malformed_read")
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
        if stats.answer_mode_search:
            return -0.2
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

    def read_reward(self, stats: QATurnStats, metadata: QAMetadata) -> float:
        if self.is_validation(metadata):
            return 0.0
        reward = 0.02 if stats.read_nonempty else -0.03
        reward -= 0.02 * stats.missing_reads
        if stats.extra_action_text:
            reward -= 0.02
        return max(min(reward, 0.1), -0.1)

    def format_error_reward(self, error: str, metadata: QAMetadata) -> float:
        if self.is_validation(metadata):
            return 0.0
        if error == "mixed_action":
            return -0.2
        return -0.05 if error in {"malformed_search", "malformed_read", "empty_search", "empty_read", "unclosed_think"} else -0.1

    def timeout_reward(self, metadata: QAMetadata) -> float:
        return 0.0 if self.is_validation(metadata) else -1.0


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    history: list[str]
    tokens_used: int
    stats: QATurnStats
    hit_refs: list[dict[str, Any]] | None = None
    read_history: list[str] | None = None


class RetrievalSession:
    def __init__(
        self,
        search_engine: SimpleSearchEngine,
        *,
        search_top_k: int,
        max_retrieval_tokens_per_turn: int,
        max_total_retrieval_tokens: int,
        max_read_tokens_per_turn: int,
        read_radius: int,
    ):
        self.search_engine = search_engine
        self.search_top_k = search_top_k
        self.max_retrieval_tokens_per_turn = max_retrieval_tokens_per_turn
        self.max_total_retrieval_tokens = max_total_retrieval_tokens
        self.max_read_tokens_per_turn = max_read_tokens_per_turn
        self.read_radius = read_radius

    def run(self, queries: list[str], metadata: QAMetadata) -> RetrievalResult:
        history = list(metadata.get("search_history", []))
        def history_key(raw: str) -> str:
            return _compact(clean_query(raw, max_chars=128))

        history_folded = {history_key(h) for h in history}
        used = int(metadata.get("retrieval_tokens_used", 0))
        remaining = max(self.max_total_retrieval_tokens - used, 0)
        if remaining <= 0:
            stats = QATurnStats(action="search", valid_search=True, empty_queries=len(queries))
            return RetrievalResult("", history, used, stats, hit_refs=[], read_history=list(metadata.get("read_history", [])))

        repeated = 0
        empty = 0
        fresh_queries: list[str] = []
        for query in queries:
            key = history_key(query)
            if key in history_folded:
                repeated += 1
                continue
            fresh_queries.append(query)
            history.append(query)
            history_folded.add(key)

        visible_answer_type = _visible_answer_type(str(metadata.get("query", "")))
        answer_type = visible_answer_type or str(metadata.get("answer_type") or "")
        bank_queries = [query for query in fresh_queries if _is_question_bank_lookup_query(query)]
        seen_bank_queries: set[str] = set()
        bank_hits = []
        for bank_query in bank_queries:
            bank_query = normalize_text(bank_query)
            key = bank_query.casefold()
            if not bank_query or key in seen_bank_queries:
                continue
            seen_bank_queries.add(key)
            bank_hits.extend(self.search_engine.search_question_bank(bank_query, answer_type=answer_type, top_k=3))

        all_catalog_hits = []
        if not bank_hits:
            for query in fresh_queries:
                hits = list(self.search_engine.search_catalog(query, top_k=self.search_top_k))
                if not hits:
                    empty += 1
                all_catalog_hits.extend(hits)

        deduped_catalog_hits = []
        seen_hit_keys: set[tuple[str, tuple[str, ...]]] = set()
        for hit in all_catalog_hits:
            hit_key = (hit.section.doc_path, hit.section.section_path)
            if hit_key in seen_hit_keys:
                continue
            seen_hit_keys.add(hit_key)
            deduped_catalog_hits.append(hit)

        budget = min(self.max_retrieval_tokens_per_turn, remaining)
        content, hit_refs = _format_ranked_search_results(
            fresh_queries,
            bank_hits,
            deduped_catalog_hits,
            max_tokens=budget,
        )
        tokens_now = estimate_tokens(content)
        stats = QATurnStats(
            action="search",
            valid_search=True,
            search_nonempty=bool(content),
            repeated_queries=repeated,
            empty_queries=empty,
            extra_action_text=bool(metadata.get("_last_action_extra_text", False)),
        )
        return RetrievalResult(content, history, used + tokens_now, stats, hit_refs=hit_refs, read_history=list(metadata.get("read_history", [])))

    def read(self, read_ids: tuple[int, ...], metadata: QAMetadata) -> RetrievalResult:
        history = list(metadata.get("search_history", []))
        read_history = list(metadata.get("read_history", []))
        used = int(metadata.get("retrieval_tokens_used", 0))
        remaining = max(self.max_total_retrieval_tokens - used, 0)
        if remaining <= 0:
            stats = QATurnStats(action="read", valid_read=True, missing_reads=len(read_ids))
            content = (
                "读取失败：本题检索上下文预算已用完。\n"
                "下一步：根据已有资料输出 boxed，或用更短、更核心的问题重新 search。"
            )
            return RetrievalResult(
                content,
                history,
                used,
                stats,
                hit_refs=list(metadata.get("last_search_hits", [])),
                read_history=read_history,
            )

        hit_refs = list(metadata.get("last_search_hits", []))
        ref_by_id = {int(ref.get("id", 0)): ref for ref in hit_refs if isinstance(ref, dict)}
        parts: list[str] = []
        missing = 0
        for read_id in read_ids:
            ref = ref_by_id.get(read_id)
            if not ref:
                missing += 1
                parts.append(
                    "\n".join(
                        [
                            f"读取失败：上一轮资料列表里没有编号 {read_id}。",
                            "下一步：只能读取上一轮实际出现的编号，或重新 search。",
                        ]
                    )
                )
                continue
            if bool(ref.get("question_bank", False)):
                parts.append(
                    "\n".join(
                        [
                            f"读取对象：上一轮第 {read_id} 条资料",
                            f"题目：{normalize_text(str(ref.get('question', '')))}",
                            f"答案：{str(ref.get('answer_detail') or normalize_text(str(ref.get('answer', ''))))}",
                        ]
                    )
                )
                read_key = f"question_bank#{read_id}"
                if read_key not in read_history:
                    read_history.append(read_key)
                continue
            doc_path = str(ref.get("doc_path", ""))
            chunk_id = int(ref.get("chunk_id", -1))
            read_key = f"{doc_path}#{chunk_id}"
            if read_key not in read_history:
                read_history.append(read_key)
            read_query = " ".join(
                [
                    str(metadata.get("query", "")),
                    " ".join(str(q) for q in history[-3:]),
                    str(ref.get("doc_title", "")),
                    str(ref.get("section", "")),
                    " ".join(str(part) for part in ref.get("section_path", []) if part),
                ]
            )
            context = ""
            fallback_plans = (
                (self.read_radius, 3),
                (max(self.read_radius + 2, 4), 6),
                (max(self.read_radius + 6, 8), 10),
            )
            for radius, top_k in fallback_plans:
                chunks = self.search_engine.read_relevant_context(
                    doc_path,
                    chunk_id,
                    query=read_query,
                    radius=radius,
                    top_k=top_k,
                    prefer_relevant=bool(ref.get("catalog_entry", False)),
                )
                context = format_structured_read_context(
                    chunks,
                    center_chunk_id=chunk_id,
                    source_id=read_id,
                    neighbor_radius=radius,
                    max_tokens=self.max_read_tokens_per_turn,
                )
                if context:
                    break
            if context:
                parts.append(context)
            else:
                missing += 1
                parts.append(
                    "\n".join(
                        [
                            f"读取对象：上一轮第 {read_id} 条资料",
                            "读取失败：这条资料没有足够正文。",
                            "下一步：换一个资料编号，或用更核心的问题重新 search。",
                        ]
                    )
                )

        budget = min(self.max_read_tokens_per_turn, remaining)
        content = "\n\n".join(part for part in parts if part)
        if estimate_tokens(content) > budget:
            content = trim_to_token_budget(content, budget)
        tokens_now = estimate_tokens(content)
        stats = QATurnStats(
            action="read",
            valid_read=True,
            read_nonempty=bool(content),
            missing_reads=missing,
            extra_action_text=bool(metadata.get("_last_action_extra_text", False)),
        )
        return RetrievalResult(content, history, used + tokens_now, stats, hit_refs=hit_refs, read_history=read_history)


class QARunner:
    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.max_queries_per_turn = int(self.cfg.get("max_queries_per_turn", 3))
        self.max_query_chars = int(self.cfg.get("max_query_chars", 64))
        self.max_search_attempts = int(self.cfg.get("max_search_attempts", 2))
        self.max_read_attempts = int(self.cfg.get("max_read_attempts", 2))
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
            max_read_tokens_per_turn=int(self.cfg.get("max_read_tokens_per_turn", 320)),
            read_radius=int(self.cfg.get("read_radius", 1)),
        )

    def _last_assistant_content(self, message_log: list[dict[str, Any]]) -> str:
        for msg in reversed(message_log or []):
            if msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
        return ""

    def _format_feedback(
        self,
        queries: list[str],
        results_text: str,
        quality: str,
        *,
        force_answer: bool = False,
    ) -> str:
        results_text = _clean_visible_feedback_text(results_text)
        if results_text:
            if force_answer:
                guidance = (
                    "下一步：不要继续 search。若资料不足，依据题干、选项和常识做保守判断，只输出 boxed。\n"
                )
            elif _looks_like_catalog_retrieval(results_text):
                guidance = "下一步：如果目录相关就只输出 <read>编号</read>，打开该文档章节继续阅读。\n"
            elif quality == "strong":
                guidance = "下一步：证据足够就只输出 boxed；不足就只输出 <read>编号</read>。\n"
            else:
                guidance = (
                    "下一步：如果编号相关就只输出 <read>编号</read>；否则不要重复上一轮，"
                    "删掉具体事件细节，保留核心对象/专有名词，换上位词、同义词或合写英文术语继续 search。\n"
                )
            return f"\n{results_text}\n\n{guidance}"
        if force_answer:
            return (
                "\n没有找到直接相关资料。\n\n"
                "下一步：不要继续 search。若这是安全卫生、行为规范或明显常识题，"
                "依据题干和选项做保守判断，只输出 boxed。\n"
            )
        return (
            "\n没有找到直接相关资料。\n\n"
            "下一步：只输出一个动作；不要重复上一轮，删掉具体事件细节，"
            "保留核心对象/专有名词，换文档常用词、同义词或合写英文术语继续 search；"
            "若已经必须作答则输出 boxed。\n"
        )

    def _format_read_feedback(self, results_text: str, *, quality: str = "", force_answer: bool = False) -> str:
        results_text = _clean_visible_feedback_text(results_text)
        if results_text:
            if force_answer or quality == "strong":
                guidance = "下一步：只输出 boxed。\n"
            else:
                guidance = (
                    "下一步：若资料已能支持结论就只输出 boxed；若仍缺关键结论，"
                    "只输出 <read>其他编号</read>，或用资料标题、相关章节、专有名词继续 search。\n"
                )
            return (
                f"\n{results_text}\n\n"
                f"{guidance}"
            )
        return (
            "\n没有读到更多相关内容。\n\n"
            "下一步：只输出一个动作，继续 search 或输出 boxed。\n"
        )

    def _format_answer_mode_reminder(self) -> str:
        return (
            "\n已有资料足够直接，不要继续换关键词搜索。\n"
            "若资料不足，依据题干、选项和常识做保守判断；否则依据资料判断。下一步只输出 boxed。\n"
        )

    def _format_error(self, error: str) -> str:
        if error == "mixed_action":
            detail = (
                "动作无效：同一轮同时出现多个动作，本轮没有执行检索、阅读或评分。\n"
                "原因：search、read、boxed 会触发不同步骤，不能合并。\n"
                "下一步：只输出一个动作，例如 <search>火灾 应急处置</search> 或 \\boxed{A,C}。\n"
            )
        elif error == "empty_search":
            detail = (
                "动作无效：search 为空，或只是“关键词/查询词”等占位词，本轮没有执行检索。\n"
                "下一步：把题干里的对象、场景、动作写成短查询，例如 <search>Carrier FOUP 创建方式</search>。\n"
            )
        elif error == "empty_read":
            detail = (
                "动作无效：read 中没有资料编号，本轮没有深入阅读。\n"
                "下一步：只读取上一轮资料列表里的编号，例如 <read>1</read>。\n"
            )
        elif error == "malformed_search":
            detail = (
                "动作无效：search 标签不完整，本轮没有执行检索。\n"
                "下一步：只输出一个闭合标签，例如 <search>控制图 排除 Sample</search>。\n"
            )
        elif error == "malformed_read":
            detail = (
                "动作无效：read 标签不完整，本轮没有深入阅读。\n"
                "下一步：只输出一个闭合标签，例如 <read>1</read>。\n"
            )
        elif error == "no_action":
            detail = (
                "动作无效：没有完整的 search、read 或 boxed，本轮没有执行任何动作。\n"
                "原因：解释、提示词、复述题目、伪造资料都不是有效动作。\n"
                "下一步：只输出一个动作；要查资料用 search，要读编号用 read，要作答用 boxed。\n"
            )
        else:
            detail = (
                "动作无效：输出不符合动作协议，本轮没有执行。\n"
                "下一步：search、read 或 boxed 只能选一种，并且标签必须闭合。\n"
            )
        return f"\n{detail}"

    def _max_turns_reached(self, metadata: QAMetadata) -> bool:
        return int(metadata.get("num_turns", 0)) >= int(metadata.get("max_turns", self.cfg.get("max_turns", 6)))

    def process_turn(
        self,
        message_log: list[dict[str, Any]],
        metadata: QAMetadata,
    ) -> QAStepResult:
        metadata = dict(metadata or {})
        metadata.setdefault("search_history", [])
        metadata.setdefault("read_history", [])
        metadata.setdefault("last_search_hits", [])
        metadata.setdefault("retrieval_tokens_used", 0)
        metadata.setdefault("search_attempts", 0)
        metadata.setdefault("read_attempts", 0)
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
            if bool(metadata.get("answer_mode", False)):
                stats = QATurnStats(action="search", valid_search=True, answer_mode_search=True)
                return QAStepResult(
                    observation={"role": "environment", "content": self._format_answer_mode_reminder()},
                    reward=self.reward_policy.search_reward(stats, metadata),
                    terminated=False,
                    next_stop_strings=ACTION_STOP_STRINGS,
                    metadata=metadata,
                    answer=None,
                    stats=stats,
                )

            metadata["_last_action_extra_text"] = action.has_extra_text
            retrieval = self.retrieval.run(action.search_queries, metadata)
            metadata.pop("_last_action_extra_text", None)
            metadata["search_history"] = retrieval.history
            metadata["retrieval_tokens_used"] = retrieval.tokens_used
            metadata["last_search_hits"] = retrieval.hit_refs or []
            metadata["search_attempts"] = int(metadata.get("search_attempts", 0)) + 1
            quality = retrieval_quality(str(metadata.get("query", "")), retrieval.content)
            metadata["evidence_quality"] = quality
            can_read_weak_hits = (
                quality == "weak"
                and bool(retrieval.hit_refs)
                and int(metadata.get("read_attempts", 0)) < self.max_read_attempts
            )
            if quality == "strong" or (
                int(metadata.get("search_attempts", 0)) >= self.max_search_attempts and not can_read_weak_hits
            ):
                metadata["answer_mode"] = True
            stats = QATurnStats(
                action=retrieval.stats.action,
                valid_search=retrieval.stats.valid_search,
                search_nonempty=retrieval.stats.search_nonempty,
                repeated_queries=retrieval.stats.repeated_queries,
                empty_queries=retrieval.stats.empty_queries,
                extra_action_text=retrieval.stats.extra_action_text,
                evidence_quality=quality,
            )
            reward = self.reward_policy.search_reward(stats, metadata)
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
                observation={
                    "role": "environment",
                    "content": self._format_feedback(
                        action.search_queries,
                        retrieval.content,
                        quality,
                        force_answer=bool(metadata.get("answer_mode", False)),
                    ),
                },
                reward=float(reward),
                terminated=False,
                next_stop_strings=ACTION_STOP_STRINGS,
                metadata=metadata,
                answer=None,
                stats=stats,
            )

        if action.action == "read":
            if int(metadata.get("read_attempts", 0)) >= self.max_read_attempts:
                stats = QATurnStats(action="read", valid_read=True, missing_reads=len(action.read_ids))
                return QAStepResult(
                    observation={
                        "role": "environment",
                        "content": "\n已经读过足够多的上下文。请根据已有资料输出 boxed 作答动作。\n",
                    },
                    reward=self.reward_policy.read_reward(stats, metadata),
                    terminated=False,
                    next_stop_strings=ACTION_STOP_STRINGS,
                    metadata=metadata,
                    answer=None,
                    stats=stats,
                )
            metadata["_last_action_extra_text"] = action.has_extra_text
            retrieval = self.retrieval.read(action.read_ids, metadata)
            metadata.pop("_last_action_extra_text", None)
            metadata["search_history"] = retrieval.history
            metadata["read_history"] = retrieval.read_history or list(metadata.get("read_history", []))
            metadata["retrieval_tokens_used"] = retrieval.tokens_used
            metadata["last_search_hits"] = retrieval.hit_refs or list(metadata.get("last_search_hits", []))
            metadata["read_attempts"] = int(metadata.get("read_attempts", 0)) + 1
            read_quality = retrieval_quality(str(metadata.get("query", "")), retrieval.content)
            metadata["evidence_quality"] = read_quality
            force_answer_after_read = bool(retrieval.content) and (
                read_quality == "strong"
                or int(metadata.get("read_attempts", 0)) >= self.max_read_attempts
            )
            metadata["answer_mode"] = force_answer_after_read
            read_stats = QATurnStats(
                action=retrieval.stats.action,
                valid_read=retrieval.stats.valid_read,
                read_nonempty=retrieval.stats.read_nonempty,
                missing_reads=retrieval.stats.missing_reads,
                extra_action_text=retrieval.stats.extra_action_text,
                evidence_quality=read_quality,
            )
            reward = self.reward_policy.read_reward(read_stats, metadata)
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
                observation={
                    "role": "environment",
                    "content": self._format_read_feedback(
                        retrieval.content,
                        quality=read_quality,
                        force_answer=force_answer_after_read,
                    ),
                },
                reward=float(reward),
                terminated=False,
                next_stop_strings=ACTION_STOP_STRINGS,
                metadata=metadata,
                answer=None,
                stats=read_stats,
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
            observation={"role": "environment", "content": self._format_error(action.error)},
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
