"""Lightweight mixed Chinese/English retrieval for the QA-RL agent.

The exam environment may not have extra search packages installed, so this file
uses only the standard library. It indexes markdown/text chunks with substring
matches, English tokens, Chinese character n-grams, and heading/path features.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

try:  # Optional lexical-search stack. The environment falls back cleanly without it.
    import jieba as _jieba
except Exception:  # pragma: no cover - exercised only when optional deps are absent.
    _jieba = None

try:  # pragma: no cover - import availability is environment-dependent.
    from whoosh import fields as _whoosh_fields
    from whoosh import scoring as _whoosh_scoring
    from whoosh.analysis import LowercaseFilter as _WhooshLowercaseFilter
    from whoosh.analysis import RegexTokenizer as _WhooshRegexTokenizer
    from whoosh.filedb.filestore import FileStorage as _WhooshFileStorage
    from whoosh.qparser import MultifieldParser as _WhooshMultifieldParser
    from whoosh.qparser import OrGroup as _WhooshOrGroup
except Exception:  # pragma: no cover - exercised only when optional deps are absent.
    _whoosh_fields = None
    _whoosh_scoring = None
    _WhooshLowercaseFilter = None
    _WhooshRegexTokenizer = None
    _WhooshFileStorage = None
    _WhooshMultifieldParser = None
    _WhooshOrGroup = None

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
QUESTION_FILE_RE = re.compile(r"试题|试卷|考核|考试", re.IGNORECASE)
ANSWER_FILE_RE = re.compile(r"答案|参考答案|答案版", re.IGNORECASE)
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]*|\d+(?:\.\d+)?")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
BRACKET_SECTION_HEADING_RE = re.compile(r"^\s*[【\[]\s*(\d+(?:\.\d+){0,5})\s*[】\]]\s*(.{1,140}?)\s*$")
PLAIN_DOTTED_SECTION_HEADING_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,3}){1,5})(?:\s+|[、．]\s*|\.\s+)(.{1,140}?)\s*$"
)
PLACEHOLDER_QUERIES = {
    "关键词",
    "关键字",
    "检索词",
    "查询词",
    "待查询内容",
    "真实查询词",
    "题目关键词",
    "自然短语",
    "query",
    "keyword",
    "keywords",
    "x1",
    "x2",
    "x1 x2",
    "y",
}
ENV_FEEDBACK_QUERY_MARKERS = (
    "检索结果返回",
    "深入阅读返回",
    "协议反馈",
    "上一轮动作无效",
    "正确示例",
    "不要复述资料",
    "不要输出",
    "我没有执行检索",
    "本轮没有执行检索",
    "下一轮只输出",
    "下一轮不要复述",
    "boxed",
)
CONVERSION_FAILURE_MARKERS = (
    "转换失败清单",
    "FileNotFoundError",
    "No such file or directory",
    "/data/word/",
    "/data/ppt/",
)
QUERY_EXPANSIONS = (
    (("着火", "起火", "火警"), ("火灾", "消防", "灭火", "应急处置")),
    (("火灾",), ("着火", "起火", "消防", "灭火", "应急处置")),
    (("emo", "emergency off", "紧急停止"), ("EMO", "急停", "紧急停止")),
    (("bypass", "by pass", "by-pass"), ("bypass", "旁路", "屏蔽", "短接", "安全设施", "安全联锁", "处罚", "绩效", "警告")),
    (("安全设施", "安全联锁", "安全装置"), ("bypass", "旁路", "屏蔽", "短接", "处罚", "绩效", "警告", "虚惊事件")),
    (("带薪年假", "年假", "年休假", "带薪休假"), ("带薪年假", "年假", "年休假", "带薪年休假", "年度休假", "假期有效期", "休假有效期", "员工假期")),
    (("电子发票", "ofd发票", "ofd格式"), ("OFD", "电子发票", "OFD格式", "发票源文件")),
)
QUERY_CONCEPT_ALIASES = (
    (("着火", "起火", "火警", "火灾"), ("着火", "起火", "火警", "火灾", "消防", "灭火")),
    (("emo", "emergency off", "紧急停止", "急停"), ("EMO", "emergency off", "紧急停止", "急停")),
    (("bypass", "by pass", "by-pass"), ("bypass", "by pass", "by-pass", "旁路", "屏蔽", "短接")),
    (("安全设施", "安全联锁", "安全装置"), ("安全设施", "安全联锁", "安全装置")),
    (("带薪年假", "年假", "年休假", "带薪休假"), ("带薪年假", "年假", "年休假", "带薪年休假", "年度休假", "休假", "假期")),
    (("电子发票", "ofd发票", "ofd格式"), ("电子发票", "OFD格式", "发票源文件")),
)
DOMAIN_ANCHORS = (
    (("带薪年假", "年假", "年休假", "带薪休假"), ("带薪年假", "年假", "年休假", "带薪年休假", "年度休假", "休假", "假期", "员工假期")),
    (("着火", "起火", "火警", "火灾"), ("着火", "起火", "火警", "火灾", "消防", "灭火", "应急")),
    (("bypass", "by pass", "by-pass"), ("bypass", "旁路", "屏蔽", "短接", "安全设施", "安全联锁", "安全装置")),
    (("电子发票", "ofd发票", "ofd格式"), ("电子发票", "OFD格式", "发票源文件")),
)
GENERIC_SEARCH_TERMS = {
    "一个",
    "一些",
    "以下",
    "以上",
    "下面",
    "不能",
    "不要",
    "多少",
    "为什么",
    "什么",
    "使用",
    "全部",
    "其他",
    "关于",
    "判断",
    "功能",
    "单选",
    "可以",
    "哪个",
    "哪些",
    "回答",
    "多选",
    "如何",
    "对应",
    "应该",
    "是什么",
    "怎么",
    "怎么办",
    "怎样",
    "几个",
    "几种",
    "情形",
    "所有",
    "支持",
    "操作",
    "是否",
    "有关",
    "正确",
    "流程",
    "答案",
    "相关",
    "规范",
    "要求",
    "解决",
    "说明",
    "请问",
    "资料",
    "选择",
    "通知",
    "错误",
    "错误说法",
    "正确说法",
    "题干",
    "题目",
    "选项",
    "the",
    "and",
    "for",
    "with",
    "what",
    "which",
    "how",
    "many",
    "there",
    "are",
    "is",
    "each",
    "per",
    "number",
    "count",
    "counts",
    "quantity",
    "qty",
}
WEAK_STANDALONE_QUERY_TERMS = {
    "有效",
    "效期",
    "功能",
    "有效期",
    "事项",
    "注意事项",
    "流程",
    "操作",
    "处理",
    "方式",
    "管理",
    "规范",
    "要求",
    "安全注意",
    "安全注意事项",
    "说法",
    "情况",
    "内容",
    "文件",
    "源文",
    "源文件",
}
STANDALONE_FALLBACK_TERMS = {
    "bypass",
}
JIEBA_DOMAIN_TERMS = {
    "OFD格式",
    "电子发票",
    "发票源文件",
    "登高作业",
    "脚手架",
    "带薪年假",
    "带薪年休假",
    "年休假",
    "有效期",
    "控制图",
    "永久排除",
    "安全设施",
    "安全联锁",
    "应急处置",
    "紧急停止",
    "急停",
    "灭火器",
    "疏散集合点",
}
MANDATORY_DOMAIN_TERMS = {
    "OFD格式",
    "OFD",
    "电子发票",
    "发票源文件",
    "带薪年假",
    "带薪年休假",
    "年休假",
    "登高作业",
    "脚手架",
    "安全设施",
    "安全联锁",
}
GENERIC_SEARCH_PHRASES = (
    "错误说法",
    "正确说法",
    "哪个说法",
    "哪些说法",
    "以下哪个",
    "以下哪些",
    "应该怎么",
    "怎么处理",
    "怎么办",
)
PROPER_CODE_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9_+.#/-]*\d+[A-Z0-9_+.#/-]*|"
    r"[A-Z]{2,}(?:/[A-Z]{2,})+|"
    r"[A-Z]{2,}[A-Z0-9_+.#/-]*)\b"
)
TOC_TITLE_RE = re.compile(r"目录|contents|table\s+of\s+contents", re.IGNORECASE)
TOC_EXACT_TITLE_RE = re.compile(r"^(?:目录|contents|table\s+of\s+contents)$", re.IGNORECASE)
TOC_ENTRY_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+)\s*[、.．]?\s+"
    r"(.+?)(?:\s*\.{2,}\s*\d+|\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)
UI_OCR_NOISE_WORDS = {
    "选择时间",
    "时间段",
    "查询",
    "导出",
    "最近",
    "当前小时",
    "本月",
    "本年",
    "清空",
    "确定",
    "首页",
    "选择标签",
}
CONTROL_PANEL_OCR_TERMS = {
    "atm",
    "auto-zero",
    "cap",
    "clng",
    "cooling",
    "cp-op",
    "depo",
    "dn",
    "edit",
    "elb",
    "elv",
    "enb",
    "exh",
    "mfc",
    "manual",
    "not",
    "off",
    "on",
    "open",
    "opn",
    "press",
    "proc-enb",
    "process",
    "pump",
    "recipe",
    "tube",
    "valve",
    "vent",
}
QUANTITY_QUERY_RE = re.compile(
    r"\bhow\s+many\b|\bnumber\s+of\b|\bquantity\s+of\b|\bcounts?\b|\bqty\b|多少|几个|几种|数量",
    re.IGNORECASE,
)
QUANTITY_NUMBER_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"single|double|triple)\b|[一二三四五六七八九十]+个?",
    re.IGNORECASE,
)
QUANTITY_STOP_TERMS = {
    "how",
    "many",
    "number",
    "count",
    "counts",
    "quantity",
    "qty",
    "there",
    "are",
    "is",
    "for",
    "each",
    "per",
    "of",
    "the",
    "a",
    "an",
    "and",
    "or",
    "in",
    "on",
    "to",
    "do",
    "does",
}
EQUIPMENT_CODE_RUN_RE = re.compile(r"(?:\b[A-Z]{1,4}\d{3,}[A-Z]?\b[\s,，;；]*){6,}")
REPEATED_ALPHA_TOKEN_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_+-]{1,15})\b(?:\s+\1\b){3,}",
    re.IGNORECASE,
)
REPEATED_ALPHA_PAIR_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_+-]{1,15})\s+([A-Za-z][A-Za-z0-9_+-]{1,15})\b"
    r"(?:\s+\1\s+\2\b){3,}",
    re.IGNORECASE,
)
ALPHA_STUTTER_RE = re.compile(r"([A-Za-z])\1{2,}$")
ALPHA_STUTTER_FRAGMENT_RE = re.compile(r"([A-Za-z])\1{2,}")


@dataclass(frozen=True)
class SearchChunk:
    doc_path: str
    heading: str
    chunk_id: int
    text: str
    doc_title: str = ""
    heading_level: int = 0
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    chunk: SearchChunk
    score: float
    query: str
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSection:
    doc_path: str
    doc_title: str
    section_title: str
    section_path: tuple[str, ...]
    heading_level: int
    chunk_id: int
    search_text: str
    feature_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSectionHit:
    section: DocumentSection
    score: float
    query: str
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoreMatchStats:
    total: int
    matched: int
    english_total: int
    english_matched: int
    cjk_total: int
    cjk_matched: int
    matched_units: tuple[str, ...]


@dataclass(frozen=True)
class QueryProfile:
    raw: str
    normalized: str
    lexical_terms: tuple[str, ...]
    keywords: tuple[str, ...]
    proper_terms: tuple[str, ...]
    mandatory_terms: tuple[str, ...]
    concept_groups: tuple[tuple[str, ...], ...]
    english_phrases: tuple[str, ...]
    quantity_required: tuple[str, ...]
    high_value_terms: tuple[str, ...]
    is_quantity: bool


@dataclass(frozen=True)
class QuestionBankItem:
    qtype: str
    ordinal: int
    question: str
    answer: str
    question_path: str
    answer_path: str
    score_key: str
    options: tuple[tuple[str, str], ...] = ()
    answer_detail: str = ""


@dataclass(frozen=True)
class QuestionBankHit:
    item: QuestionBankItem
    score: float


@dataclass(frozen=True)
class AnswerGroup:
    answers_by_ordinal: dict[int, str]
    source: str


def normalize_text(text: str) -> str:
    """Normalize full-width forms and whitespace without changing semantics."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_search_terms(text: str) -> str:
    """Normalize searchable aliases without changing display text elsewhere."""
    text = normalize_text(text)
    text = re.sub(r"\bby\s*[- ]\s*pass\b", "bypass", text, flags=re.IGNORECASE)
    return text


def clean_query(query: str, max_chars: int = 64) -> str:
    query = normalize_search_terms(query)
    query = re.sub(r"[<>`*_#]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip(" \t\r\n,，;；、。.")
    lowered = query.casefold()
    if any(marker.casefold() in lowered for marker in ENV_FEEDBACK_QUERY_MARKERS):
        return ""
    query = query[:max_chars].strip()
    if query.casefold() in PLACEHOLDER_QUERIES:
        return ""
    return query


def split_search_intents(raw: str, *, max_queries: int = 3, max_chars: int = 64) -> list[str]:
    """Return primary natural query plus conservative intent splits."""
    primary = clean_query(raw, max_chars=max_chars)
    if not primary:
        return []

    candidates = [primary]
    strong_parts = re.split(r"[;；\n]+", primary)
    if len(strong_parts) > 1:
        candidates.extend(strong_parts)
    else:
        candidates.extend(re.split(r"[,，、]+", primary))

    # Add acronym/term fallbacks from long mixed-language queries.
    english_terms = EN_TOKEN_RE.findall(primary)
    has_cjk = bool(CJK_RE.search(primary))
    for term in english_terms:
        term_n = normalize_search_terms(term)
        if _is_generic_search_term(term_n):
            continue
        is_code_like = bool(PROPER_CODE_RE.fullmatch(term_n) and (any(ch.isdigit() for ch in term_n) or "/" in term_n))
        is_long_acronym = bool(re.fullmatch(r"[A-Z]{4,}", term_n))
        if (
            term_n.casefold() in STANDALONE_FALLBACK_TERMS
            or is_code_like
            or is_long_acronym
            or len(term_n) >= 4
        ):
            candidates.append(term_n)

    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        q = clean_query(cand, max_chars=max_chars)
        key = q.casefold()
        if len(q) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_queries:
            break
    return out


def cjk_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    chars = [ch for ch in text if CJK_RE.match(ch)]
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(chars) >= n:
            grams.extend("".join(chars[i : i + n]) for i in range(len(chars) - n + 1))
    return grams


def feature_tokens(text: str, *, heading: str = "", path: str = "") -> list[str]:
    text_n = normalize_search_terms(text)
    heading_n = normalize_search_terms(heading)
    path_n = normalize_search_terms(Path(path).stem)

    tokens: list[str] = []
    tokens.extend(t.casefold() for t in EN_TOKEN_RE.findall(text_n))
    tokens.extend(cjk_ngrams(text_n))

    # Heading/path features are duplicated to provide a mild prior.
    heading_tokens = [t.casefold() for t in EN_TOKEN_RE.findall(heading_n)] + cjk_ngrams(heading_n)
    path_tokens = [t.casefold() for t in EN_TOKEN_RE.findall(path_n)] + cjk_ngrams(path_n)
    tokens.extend(heading_tokens * 2)
    tokens.extend(path_tokens)
    return [t for t in tokens if t]


def query_expansion_tokens(query: str) -> list[str]:
    """Add formal/colloquial aliases to improve recall without leaking answers."""
    text = normalize_search_terms(query).casefold()
    compact = _compact_for_match(query)
    tokens: list[str] = []
    for triggers, aliases in QUERY_EXPANSIONS:
        if any(trigger.casefold() in text or _compact_for_match(trigger) in compact for trigger in triggers):
            for alias in aliases:
                tokens.extend(feature_tokens(alias))
    return tokens


def query_keywords(query: str) -> tuple[str, ...]:
    """Human-readable query terms used for hit explanations and tie-breaking."""
    query_n = clean_query(query, max_chars=96)
    if not query_n:
        return tuple()
    terms: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\s,，、;；:：()（）\[\]【】]+", query_n):
        part = part.strip(" \t\r\n.。?？!！")
        if not part:
            continue
        found = EN_TOKEN_RE.findall(part)
        found.extend(m.group(0) for m in re.finditer(r"[\u4e00-\u9fff]{2,}", part))
        if not found and len(part) >= 2:
            found.append(part)
        for term in found:
            key = normalize_text(term).casefold()
            if key and key not in seen and not _is_generic_search_term(term):
                seen.add(key)
                terms.append(term)
    return tuple(terms)


def _is_generic_search_term(term: str) -> bool:
    key = normalize_text(term).casefold()
    if not key:
        return True
    if key in GENERIC_SEARCH_TERMS:
        return True
    compact = re.sub(r"[\W_]+", "", key)
    return any(phrase in compact for phrase in GENERIC_SEARCH_PHRASES)


def _is_proper_english_term(term: str) -> bool:
    if _is_generic_search_term(term):
        return False
    if len(term) < 2:
        return False
    return bool(PROPER_CODE_RE.fullmatch(term) or any(ch.isupper() or ch.isdigit() for ch in term) or len(term) >= 4)


def proper_query_terms(query: str) -> tuple[str, ...]:
    """Terms that should dominate retrieval: equipment IDs, acronyms, systems, functions."""
    query_n = clean_query(query, max_chars=128)
    if not query_n:
        return tuple()
    terms: list[str] = []
    seen: set[str] = set()

    for term in EN_TOKEN_RE.findall(query_n):
        if not _is_proper_english_term(term):
            continue
        key = normalize_text(term).casefold()
        if key not in seen:
            seen.add(key)
            terms.append(term)

    for term in query_keywords(query_n):
        key = normalize_text(term).casefold()
        if key in seen or _is_generic_search_term(term):
            continue
        if CJK_RE.search(term) and len(term) >= 3:
            seen.add(key)
            terms.append(term)
    return tuple(terms)


def mandatory_query_terms(query: str) -> tuple[str, ...]:
    """High-signal anchors that should not be replaced by generic words."""
    query_n = clean_query(query, max_chars=128)
    if not query_n:
        return tuple()
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = normalize_search_terms(term).strip()
        key = _compact_for_match(term)
        if not key or key in seen or _is_generic_search_term(term):
            return
        seen.add(key)
        terms.append(term)

    for term in EN_TOKEN_RE.findall(query_n):
        if PROPER_CODE_RE.fullmatch(term):
            add(term)

    compact_query = _compact_for_match(query_n)
    for term in MANDATORY_DOMAIN_TERMS:
        if _compact_for_match(term) in compact_query:
            add(term)
    for triggers, aliases in DOMAIN_ANCHORS:
        if any(_compact_for_match(trigger) in compact_query for trigger in triggers):
            for alias in aliases:
                add(alias)
    return tuple(terms)


def is_quantity_query(query: str) -> bool:
    return bool(QUANTITY_QUERY_RE.search(normalize_search_terms(query or "")))


def _english_content_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in EN_TOKEN_RE.findall(normalize_search_terms(text)):
        key = normalize_text(term).casefold()
        if not key or key in seen or key in QUANTITY_STOP_TERMS or _is_generic_search_term(key):
            continue
        if len(key) < 2:
            continue
        seen.add(key)
        terms.append(key)
    return terms


def quantity_required_terms(query: str) -> tuple[str, ...]:
    """Objects being counted in an English quantity question."""
    query_n = normalize_search_terms(query)
    match = re.search(
        r"\bhow\s+many\s+(.+?)(?:\s+(?:are|is|for|in|on|does|do|can|should)\b|[?？]|$)",
        query_n,
        flags=re.IGNORECASE,
    )
    if match:
        terms = _english_content_terms(match.group(1))
        if terms:
            return _dedupe_ordered(terms)
    return _dedupe_ordered(
        term
        for term in _english_content_terms(query_n)
        if term not in {"process", "module", "pm"}
    )


def english_query_phrases(query: str) -> tuple[str, ...]:
    terms = _english_content_terms(query)
    phrases: list[str] = []
    for n in (3, 2):
        for i in range(0, max(len(terms) - n + 1, 0)):
            phrase = " ".join(terms[i : i + n])
            if phrase:
                phrases.append(phrase)
    return _dedupe_ordered(phrases)


def _high_value_query_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    terms.extend(proper_query_terms(query))
    terms.extend(query_keywords(query))
    if is_quantity_query(query):
        terms.extend(quantity_required_terms(query))
    terms.extend(query_domain_anchors(query))
    weak_compacts = {_compact_for_match(term) for term in WEAK_STANDALONE_QUERY_TERMS}
    return _dedupe_ordered(
        term
        for term in terms
        if term and not _is_generic_search_term(term) and _compact_for_match(term) not in weak_compacts
    )


def build_query_profile(query: str) -> QueryProfile:
    query_n = clean_query(query, max_chars=128)
    quantity = is_quantity_query(query_n)
    lexical = list(lexical_tokens(query_n, for_query=True))
    lexical.extend(query_expansion_tokens(query_n))
    quantity_required = quantity_required_terms(query_n) if quantity else tuple()
    for term in quantity_required:
        lexical.extend(lexical_tokens(term, for_query=False))
    phrases = english_query_phrases(query_n)
    for phrase in phrases:
        lexical.extend(lexical_tokens(phrase, for_query=False))
    return QueryProfile(
        raw=query,
        normalized=query_n,
        lexical_terms=_dedupe_ordered(lexical),
        keywords=query_keywords(query_n),
        proper_terms=proper_query_terms(query_n),
        mandatory_terms=mandatory_query_terms(query_n),
        concept_groups=_query_concept_groups(query_n),
        english_phrases=phrases,
        quantity_required=quantity_required,
        high_value_terms=_high_value_query_terms(query_n),
        is_quantity=quantity,
    )


def _term_present_in_text(term: str, *, text: str, heading: str = "", path: str = "") -> bool:
    raw_haystack = normalize_search_terms(" ".join([text, heading, Path(path).stem]))
    haystack = raw_haystack.casefold()
    if not raw_haystack.strip():
        return False
    term_raw = normalize_search_terms(term).strip()
    term_n = term_raw.casefold()
    if not term_raw:
        return False
    if re.fullmatch(r"[A-Z]{2,}", term_raw):
        for token in EN_TOKEN_RE.findall(raw_haystack):
            parts = [part for part in re.split(r"[/_+.#-]+", token) if part]
            if len(term_raw) <= 3:
                if any(part == term_raw for part in parts):
                    return True
            elif any(part.casefold() == term_n for part in parts):
                return True
        return False
    if re.fullmatch(r"[A-Za-z0-9_+.#/-]+", term_raw):
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_+.#/-]){re.escape(term_n)}(?:s|es)?(?![A-Za-z0-9_+.#/-])",
                haystack,
            )
        )
    return _compact_for_match(term_n) in _compact_for_match(haystack)


def high_value_match_terms(
    profile: QueryProfile,
    *,
    text: str,
    heading: str = "",
    path: str = "",
) -> tuple[str, ...]:
    if not profile.high_value_terms:
        return tuple()
    if not normalize_search_terms(" ".join([text, heading, Path(path).stem])).strip():
        return tuple()
    matched: list[str] = []
    for term in profile.high_value_terms:
        if _term_present_in_text(term, text=text, heading=heading, path=path):
            matched.append(term)
    return _dedupe_ordered(matched)


def phrase_match_score(profile: QueryProfile, *, text: str, heading: str = "", path: str = "") -> float:
    if not profile.english_phrases:
        return 0.0
    body = normalize_search_terms(text).casefold()
    title = normalize_search_terms(" ".join([heading, Path(path).stem])).casefold()
    score = 0.0
    for phrase in profile.english_phrases:
        phrase_n = normalize_search_terms(phrase).casefold()
        if not phrase_n:
            continue
        if phrase_n in body:
            score += 1.8
        if phrase_n in title:
            score += 2.4
    return score


def count_evidence_score(query: str, *, text: str, heading: str = "", path: str = "") -> float:
    """Score whether text looks like evidence for a quantity/count question."""
    if not is_quantity_query(query):
        return 0.0
    haystack = normalize_search_terms(" ".join([text, heading, Path(path).stem])).casefold()
    if not haystack.strip():
        return 0.0
    compact = _compact_for_match(haystack)
    required = quantity_required_terms(query)
    def has_required_term(term: str) -> bool:
        term_n = normalize_search_terms(term).casefold()
        if re.fullmatch(r"[a-z0-9_+.#/-]+", term_n):
            return bool(re.search(rf"\b{re.escape(term_n)}s?\b", haystack, flags=re.IGNORECASE))
        return _compact_for_match(term_n) in compact

    matched_required = [term for term in required if has_required_term(term)]
    if required and len(matched_required) < len(required):
        return 0.0

    score = 1.4 * len(matched_required)
    phrases = english_query_phrases(query)
    for phrase in phrases:
        if phrase in haystack:
            score += 1.6
    if "process module" in normalize_search_terms(query).casefold():
        if "process module" in haystack or re.search(r"\bpm\b", haystack):
            score += 2.0
        else:
            score -= 2.0
    if QUANTITY_NUMBER_RE.search(haystack):
        score += 0.8
    relation_hits = 0
    for term in matched_required:
        term_re = re.escape(term)
        if re.search(rf"(?:{term_re}.{{0,80}}{QUANTITY_NUMBER_RE.pattern})|(?:{QUANTITY_NUMBER_RE.pattern}.{{0,80}}{term_re})", haystack, flags=re.IGNORECASE):
            score += 0.7
            relation_hits += 1
    has_unit_context = any(marker in haystack for marker in ("each", "per ", "for each", "每个", "每一"))
    has_count_context = any(
        marker in haystack
        for marker in (
            " has ",
            " have ",
            " contains ",
            " equipped ",
            " configuration",
            " count",
            " number",
            "数量",
            "配置",
            "几个",
        )
    )
    if has_unit_context:
        score += 0.4
    if has_count_context:
        score += 0.8
    if required and relation_hits < len(required) and not (has_unit_context or has_count_context):
        score *= 0.45
    return max(score, 0.0)


def proper_term_match_count_from_terms(
    terms: Iterable[str],
    *,
    text: str,
    heading: str = "",
    path: str = "",
) -> int:
    haystack = normalize_text(" ".join([text, heading, Path(path).stem])).casefold()
    haystack_compact = re.sub(r"[\W_]+", "", haystack)
    count = 0
    for term in terms:
        needle = normalize_text(term).casefold()
        if not needle:
            continue
        if CJK_RE.search(needle):
            needle = re.sub(r"[\W_]+", "", needle)
            if needle and needle in haystack_compact:
                count += 1
        elif _term_present_in_text(term, text=text, heading=heading, path=path):
                count += 1
    return count


def proper_term_match_count(query: str, *, text: str, heading: str = "", path: str = "") -> int:
    return proper_term_match_count_from_terms(proper_query_terms(query), text=text, heading=heading, path=path)


_JIEBA_READY = False


def _ensure_jieba_ready() -> None:
    global _JIEBA_READY
    if _JIEBA_READY or _jieba is None:
        return
    for term in JIEBA_DOMAIN_TERMS:
        _jieba.add_word(term, freq=200000)
    for _, aliases in QUERY_EXPANSIONS:
        for alias in aliases:
            if CJK_RE.search(alias):
                _jieba.add_word(alias, freq=120000)
    _JIEBA_READY = True


def _dedupe_ordered(items: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = normalize_search_terms(item).strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


def lexical_tokens(text: str, *, for_query: bool = False) -> tuple[str, ...]:
    """Tokenize text for fielded lexical search.

    Whoosh receives whitespace-separated tokens, so Chinese segmentation happens
    here. We still keep English IDs/acronyms verbatim because equipment IDs and
    system names are often the strongest retrieval signal.
    """
    text_n = normalize_search_terms(text)
    tokens: list[str] = [term.casefold() for term in EN_TOKEN_RE.findall(text_n)]
    _ensure_jieba_ready()
    for segment in re.findall(r"[\u4e00-\u9fff]+", text_n):
        if _jieba is not None:
            tokens.extend(str(tok).strip() for tok in _jieba.cut_for_search(segment))
        else:
            tokens.extend(cjk_ngrams(segment, min_n=2, max_n=4))
    if for_query:
        tokens.extend(query_keywords(text_n))
        for proper in proper_query_terms(text_n):
            tokens.append(proper)
        compact = _compact_for_match(text_n)
        for triggers, aliases in QUERY_EXPANSIONS:
            if any(_compact_for_match(trigger) in compact for trigger in triggers):
                for alias in aliases:
                    tokens.extend(lexical_tokens(alias, for_query=False))
    cleaned: list[str] = []
    for token in tokens:
        token = normalize_search_terms(str(token)).strip(" \t\r\n,，;；、。.?？!！")
        if not token:
            continue
        if len(token) == 1 and CJK_RE.search(token):
            continue
        if for_query and _is_generic_search_term(token):
            continue
        cleaned.append(token.casefold())
    return _dedupe_ordered(cleaned)


def _lexical_document_text(*parts: str) -> str:
    tokens: list[str] = []
    for part in parts:
        tokens.extend(lexical_tokens(part, for_query=False))
    return " ".join(_dedupe_ordered(tokens))


def _query_concept_groups(query: str) -> tuple[tuple[str, ...], ...]:
    query_n = clean_query(query, max_chars=128)
    if not query_n:
        return tuple()
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    compact_query = _compact_for_match(query_n)
    expanded_alias_compacts: set[str] = set()
    for triggers, aliases in QUERY_EXPANSIONS:
        if any(_compact_for_match(trigger) in compact_query for trigger in triggers):
            expanded_alias_compacts.update(_compact_for_match(trigger) for trigger in triggers)
            expanded_alias_compacts.update(_compact_for_match(alias) for alias in aliases)

    def add_group(aliases: Iterable[str]) -> None:
        cleaned: list[str] = []
        for alias in aliases:
            alias = normalize_search_terms(str(alias)).strip()
            if not alias or _is_generic_search_term(alias):
                continue
            if _compact_for_match(alias) in {_compact_for_match(term) for term in WEAK_STANDALONE_QUERY_TERMS}:
                continue
            cleaned.append(alias)
        group = _dedupe_ordered(cleaned)
        if not group:
            return
        key = "|".join(_compact_for_match(term) for term in group)
        if key and key not in seen:
            seen.add(key)
            groups.append(group)

    for term in proper_query_terms(query_n):
        if _compact_for_match(term) in expanded_alias_compacts:
            continue
        add_group((term,))
    for token in lexical_tokens(query_n, for_query=False):
        if token in WEAK_STANDALONE_QUERY_TERMS:
            continue
        if _compact_for_match(token) in expanded_alias_compacts:
            continue
        if len(token) < 2:
            continue
        add_group((token,))

    for triggers, aliases in QUERY_EXPANSIONS:
        if any(_compact_for_match(trigger) in compact_query for trigger in triggers):
            add_group(aliases)

    for term in query_keywords(query_n):
        if CJK_RE.search(term) and len(term) >= 4 and not groups:
            add_group((term,))
    return tuple(groups)


def _core_match_stats_for_groups(
    groups: tuple[tuple[str, ...], ...],
    *,
    text: str,
    heading: str = "",
    path: str = "",
) -> CoreMatchStats:
    if not groups:
        return CoreMatchStats(0, 0, 0, 0, 0, 0, tuple())
    haystack = normalize_search_terms(" ".join([text, heading, Path(path).stem])).casefold()
    haystack_compact = _compact_for_match(haystack)
    matched_units: list[str] = []
    english_total = english_matched = cjk_total = cjk_matched = 0
    matched = 0
    for group in groups:
        is_cjk = any(CJK_RE.search(alias) for alias in group)
        if is_cjk:
            cjk_total += 1
        else:
            english_total += 1
        group_matched = False
        for alias in group:
            alias_n = normalize_search_terms(alias).casefold()
            alias_compact = _compact_for_match(alias_n)
            if not alias_compact:
                continue
            if CJK_RE.search(alias_n):
                group_matched = alias_compact in haystack_compact
            else:
                group_matched = _term_present_in_text(alias, text=text, heading=heading, path=path)
            if group_matched:
                matched_units.append(alias)
                break
        if group_matched:
            matched += 1
            if is_cjk:
                cjk_matched += 1
            else:
                english_matched += 1
    return CoreMatchStats(
        total=len(groups),
        matched=matched,
        english_total=english_total,
        english_matched=english_matched,
        cjk_total=cjk_total,
        cjk_matched=cjk_matched,
        matched_units=tuple(matched_units),
    )


def core_match_stats(query: str, *, text: str, heading: str = "", path: str = "") -> CoreMatchStats:
    return _core_match_stats_for_groups(_query_concept_groups(query), text=text, heading=heading, path=path)


def passes_core_relevance(query: str, *, text: str, heading: str = "", path: str = "") -> bool:
    stats = core_match_stats(query, text=text, heading=heading, path=path)
    if stats.total == 0:
        return True
    return stats.matched > 0


def _looks_like_toc(text: str, *, heading: str = "") -> bool:
    text_n = normalize_text(text)
    heading_n = normalize_text(heading)
    if TOC_TITLE_RE.search(heading_n) or TOC_TITLE_RE.search(text_n[:160]):
        return True
    toc_like = 0
    for line in text.splitlines()[:80]:
        if TOC_ENTRY_RE.match(normalize_text(line)):
            toc_like += 1
    return toc_like >= 4


def _is_toc_title(text: str) -> bool:
    title = normalize_text(text).strip(" \t\r\n#:-:：-—")
    return bool(TOC_EXACT_TITLE_RE.fullmatch(title))


def _section_path_has_toc(section_path: Iterable[str]) -> bool:
    return any(_is_toc_title(part) for part in section_path)


def _toc_entries(text: str) -> tuple[str, ...]:
    entries: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines()[:120]:
        line = normalize_text(raw_line)
        if not line or TOC_TITLE_RE.fullmatch(line):
            continue
        match = TOC_ENTRY_RE.match(line)
        if match:
            title = normalize_text(match.group(1))
        else:
            title = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", line).strip()
        title = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+)\s*[、.．]?\s*", "", title)
        title = re.sub(r"\s+\d{1,4}\s*$", "", title).strip()
        if len(title) < 2 or _is_generic_search_term(title):
            continue
        key = title.casefold()
        if key not in seen:
            seen.add(key)
            entries.append(title)
    return tuple(entries)


def _text_has_any_query_term(text: str, terms: Iterable[str]) -> bool:
    haystack = normalize_text(text).casefold()
    haystack_compact = re.sub(r"[\W_]+", "", haystack)
    for term in terms:
        needle = normalize_text(term).casefold()
        if not needle:
            continue
        if CJK_RE.search(needle):
            needle_compact = re.sub(r"[\W_]+", "", needle)
            if needle_compact and needle_compact in haystack_compact:
                return True
        elif needle in haystack:
            return True
    return False


def query_domain_anchors(query: str) -> tuple[str, ...]:
    compact_query = _compact_for_match(query)
    anchors: list[str] = []
    seen: set[str] = set()
    for triggers, aliases in DOMAIN_ANCHORS:
        if not any(_compact_for_match(trigger) in compact_query for trigger in triggers):
            continue
        for alias in aliases:
            key = _compact_for_match(alias)
            if key and key not in seen:
                seen.add(key)
                anchors.append(alias)
    return tuple(anchors)


def domain_anchor_match_count(query: str, *, text: str, heading: str = "", path: str = "") -> int:
    anchors = query_domain_anchors(query)
    if not anchors:
        return 0
    count = sum(1 for anchor in anchors if _term_present_in_text(anchor, text=text, heading=heading, path=path))
    if count:
        return count
    query_compact = _compact_for_match(query)
    haystack_compact = _compact_for_match(" ".join([text, heading, Path(path).stem]))
    invoice_triggered = any(
        _compact_for_match(trigger) in query_compact
        for trigger in ("电子发票", "OFD发票", "OFD格式", "发票源文件")
    )
    if invoice_triggered and "发票" in haystack_compact and "源文件" in haystack_compact:
        return 1
    return 0


def matched_query_keywords(query: str, *, text: str, heading: str = "", path: str = "") -> tuple[str, ...]:
    haystack = normalize_text(" ".join([text, heading, Path(path).stem])).casefold()
    matched: list[str] = []
    for term in query_keywords(query):
        if normalize_text(term).casefold() in haystack:
            matched.append(term)
    return tuple(matched)


def estimate_tokens(text: str) -> int:
    """A cheap mixed-language token estimate for retrieval budgeting."""
    text = normalize_text(text)
    if not text:
        return 0
    cjk = len(CJK_RE.findall(text))
    other = max(len(text) - cjk, 0)
    return cjk + math.ceil(other / 4)


def trim_to_token_budget(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    chars: list[str] = []
    budget = 0.0
    for ch in text:
        budget += 1.0 if CJK_RE.match(ch) else 0.25
        if budget > max_tokens:
            break
        chars.append(ch)
    return "".join(chars).rstrip() + "..."


def duplicate_text_key(text: str, *, max_chars: int = 320) -> str:
    """Compact cleaned text used to suppress overlapping OCR/PDF chunks."""
    cleaned = clean_retrieval_snippet(text) or normalize_text(text)
    key = re.sub(r"[\W_]+", "", normalize_text(cleaned).casefold())
    return key[:max_chars]


def is_near_duplicate_text(text: str, seen_keys: Iterable[str], *, threshold: float = 0.92) -> bool:
    key = duplicate_text_key(text)
    if not key:
        return True
    for old in seen_keys:
        if not old:
            continue
        if key == old:
            return True
        shorter, longer = (key, old) if len(key) <= len(old) else (old, key)
        if len(shorter) >= 56 and shorter in longer:
            return True
        prefix_len = min(len(shorter), 180)
        if prefix_len >= 72 and key[:prefix_len] == old[:prefix_len]:
            return True
        if min(len(key), len(old)) >= 80 and SequenceMatcher(None, key, old).ratio() >= threshold:
            return True
    return False


def _looks_like_conversion_failure_manifest(text: str) -> bool:
    text_n = unicodedata.normalize("NFKC", text or "")
    if "转换失败清单" in text_n and (
        "FileNotFoundError" in text_n or "No such file or directory" in text_n
    ):
        return True
    failure_hits = text_n.count("FileNotFoundError") + text_n.count("No such file or directory")
    path_hits = text_n.count("/data/word/") + text_n.count("/data/ppt/")
    return failure_hits >= 3 and path_hits >= 3


def _looks_like_ui_ocr_noise(line: str) -> bool:
    line_n = normalize_text(line)
    if not line_n:
        return True
    if len(line_n) <= 1:
        return True
    lowered = line_n.casefold()
    if "there is no text in the image" in lowered or "no text detected" in lowered:
        return True
    if lowered.startswith(("environment:", "assistant:", "user:", "system:")):
        return True
    if "FileNotFoundError" in line_n or "No such file or directory" in line_n:
        return True
    if "转换失败清单" in line_n and "原因" in line_n:
        return True
    if re.fullmatch(r"#+\s*notes\s*:?", lowered):
        return True
    if re.search(r"\bcpu\s*:", lowered) and re.search(r"\boos\s*:", lowered):
        return True
    if len(re.findall(r"\b[A-Z]\d{3,}\b", line_n)) >= 8:
        return True
    if len(re.findall(r"\b(?:Module|Fab)\s*-?\d{1,3}(?:-\d{1,3})?\b", line_n, flags=re.IGNORECASE)) >= 8:
        return True
    tokens = re.findall(r"[A-Za-z0-9_+-]{2,}", line_n)
    token_keys = [token.casefold() for token in tokens]
    if len(tokens) >= 30 and len(set(tokens)) / max(len(tokens), 1) < 0.35:
        return True
    if len(token_keys) >= 12:
        pairs = Counter(zip(token_keys, token_keys[1:], strict=False))
        if pairs and pairs.most_common(1)[0][1] >= 6:
            return True
    cjk = len(CJK_RE.findall(line_n))
    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", line_n)
    if len(alpha_tokens) >= 18 and cjk <= 4 and not re.search(r"[。！？；;,.，、]", line_n):
        return True
    control_hits = sum(1 for token in token_keys if token in CONTROL_PANEL_OCR_TERMS)
    short_panel_tokens = sum(
        1
        for token in tokens
        if re.fullmatch(r"[A-Z]{1,5}\d{0,2}|[A-Z]{1,3}-[A-Z0-9]{1,5}", token)
    )
    length = max(len(line_n), 1)
    if len(line_n) > 160 and control_hits >= 8 and cjk / length < 0.25:
        return True
    if len(line_n) > 220 and short_panel_tokens >= 40 and cjk / length < 0.2:
        return True
    ui_hits = sum(1 for word in UI_OCR_NOISE_WORDS if word in line_n)
    date_hits = len(re.findall(r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2})?|\d{1,2}:\d{2}(?::\d{2})?", line_n))
    digits = len(re.findall(r"\d", line_n))
    if ui_hits >= 3 and (date_hits >= 2 or digits >= 30):
        return True
    if len(line_n) > 140 and digits / length > 0.38 and cjk / length < 0.42:
        return True
    return False


def clean_extracted_text(text: str) -> str:
    """Lightly clean PDF/OCR conversion artifacts while preserving useful OCR text."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\\n", "\n")
    if _looks_like_conversion_failure_manifest(text):
        return ""
    text = re.sub(r"<!--\s*(?:Slide|Page)\s+number\s*:\s*\d+\s*-->", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\*?\[\s*Image OCR\s*\]\*?", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\*?\[\s*End OCR\s*\]\*?", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"There is no text in the image\.?", " ", text, flags=re.IGNORECASE)
    text = text.replace("\u3000", " ")

    out_lines: list[str] = []
    recent_short_lines: list[str] = []
    blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if not blank:
                out_lines.append("")
            blank = True
            continue
        if _looks_like_ui_ocr_noise(line):
            continue
        line_key = line.casefold()
        if len(line) <= 40 and line_key in recent_short_lines:
            continue
        out_lines.append(line)
        if len(line) <= 40:
            recent_short_lines.append(line_key)
            recent_short_lines = recent_short_lines[-6:]
        blank = False
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_panelish_token(token: str) -> bool:
    key = token.casefold()
    if key in CONTROL_PANEL_OCR_TERMS:
        return True
    if re.fullmatch(r"[A-Z]{1,5}\d{0,3}", token):
        return True
    return bool(re.fullmatch(r"[A-Z]{1,4}-[A-Z0-9]{1,8}", token))


def _drop_repeated_panel_match(match: re.Match[str]) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", match.group(0))
    if tokens and all(_is_panelish_token(token) for token in tokens[:2]):
        return " "
    return match.group(0)


def _strip_dense_panel_segment(line: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?", line)
    if len(tokens) < 28:
        return line
    panelish = sum(1 for token in tokens if _is_panelish_token(token))
    cjk = len(CJK_RE.findall(line))
    if panelish >= 18 and panelish / max(len(tokens), 1) >= 0.62 and cjk <= 12:
        return ""
    return line


def _token_key(token: str) -> str:
    token = token.strip(" \t\r\n,，;；.。:：()（）[]【】<>《》\"'")
    return normalize_text(token).casefold()


def _looks_like_cjk_phrase(token: str) -> bool:
    cjk = "".join(CJK_RE.findall(token))
    return len(cjk) >= 4


def _drop_alpha_stutter_tokens(line: str) -> str:
    if len(ALPHA_STUTTER_FRAGMENT_RE.findall(line)) >= 5:
        line = ALPHA_STUTTER_FRAGMENT_RE.sub("", line)
    tokens = line.split()
    stutter_indexes = {
        idx
        for idx, token in enumerate(tokens)
        if ALPHA_STUTTER_RE.fullmatch(_token_key(token) or "")
    }
    if len(stutter_indexes) < 5:
        return line
    return " ".join(token for idx, token in enumerate(tokens) if idx not in stutter_indexes)


def _collapse_repeated_space_tokens(line: str) -> str:
    tokens = line.split()
    if len(tokens) < 2:
        return line
    out: list[str] = []
    prev_key = ""
    for token in tokens:
        key = _token_key(token)
        collapsible = _looks_like_cjk_phrase(key) or _is_panelish_token(key.upper())
        if key and key == prev_key and collapsible:
            continue
        out.append(token)
        prev_key = key
    return " ".join(out)


def clean_retrieval_snippet(text: str) -> str:
    """Clean text immediately before it is shown to the model."""
    text = clean_extracted_text(text)
    if not text:
        return ""
    text = EQUIPMENT_CODE_RUN_RE.sub(" ", text)
    text = REPEATED_ALPHA_PAIR_RE.sub(_drop_repeated_panel_match, text)
    text = REPEATED_ALPHA_TOKEN_RE.sub(_drop_repeated_panel_match, text)

    out_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        line = _drop_alpha_stutter_tokens(line)
        line = _collapse_repeated_space_tokens(line)
        if not line:
            continue
        if _looks_like_ui_ocr_noise(line):
            continue
        line = _strip_dense_panel_segment(line)
        if line:
            out_lines.append(line)
    cleaned = "\n".join(out_lines)
    cleaned = clean_extracted_text(cleaned)
    return normalize_text(cleaned)


def _source_label(hit: SearchHit) -> str:
    title = hit.chunk.heading or Path(hit.chunk.doc_path).stem
    title = normalize_text(title)
    if re.fullmatch(r"(?:page|slide)\s*\d+", title, flags=re.IGNORECASE):
        return "内部文档片段"
    return title or "内部文档片段"


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]
    out = []
    start = 0
    stride = max(max_chars - 120, max_chars // 2)
    while start < len(paragraph):
        piece = paragraph[start : start + max_chars].strip()
        if piece:
            out.append(piece)
        start += stride
    return out


def _valid_numbered_heading_title(number: str, title_text: str) -> bool:
    title_text = normalize_text(title_text)
    if not title_text or title_text[0].isdigit() or title_text[0] in "%‰+-=<>":
        return False
    first_part = number.split(".", 1)[0]
    if first_part == "0":
        return False
    cjk_count = len(CJK_RE.findall(title_text))
    alpha_count = len(re.findall(r"[A-Za-z]", title_text))
    digit_count = len(re.findall(r"\d", title_text))
    if cjk_count < 2 and alpha_count < 3:
        return False
    if digit_count / max(len(title_text), 1) > 0.35:
        return False
    numeric_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", title_text)
    if len(numeric_tokens) >= 4:
        return False
    if re.search(r"[<>]=?|=", title_text) and len(numeric_tokens) >= 2:
        return False
    return True


def _numbered_section_heading(line: str) -> tuple[int, str] | None:
    line = normalize_text(line)
    match = BRACKET_SECTION_HEADING_RE.match(line)
    if match:
        number = match.group(1)
        title_text = normalize_text(match.group(2))
        if not _valid_numbered_heading_title(number, title_text):
            return None
        title = normalize_text(f"【{number}】{title_text}")
        if not title or _looks_like_ui_ocr_noise(title):
            return None
        return len(number.split(".")), title
    match = PLAIN_DOTTED_SECTION_HEADING_RE.match(line)
    if not match:
        return None
    number = match.group(1)
    title_text = normalize_text(match.group(2))
    if not _valid_numbered_heading_title(number, title_text):
        return None
    title = normalize_text(f"【{match.group(1)}】{match.group(2)}")
    if not title or _looks_like_ui_ocr_noise(title):
        return None
    return len(match.group(1).split(".")), title


def _catalog_text_sample(text: str, *, max_chars: int = 900) -> str:
    """Cheap hidden text sample for catalog ranking; full cleaning happens on output/read."""
    return normalize_text((text or "")[:max_chars])


def _emit_chunk(
    chunks: list[SearchChunk],
    doc_path: str,
    heading: str,
    text: str,
    max_chars: int,
    *,
    doc_title: str = "",
    heading_level: int = 0,
    section_path: tuple[str, ...] = (),
) -> None:
    text = text.strip()
    if not text:
        return
    for part in _split_long_paragraph(text, max_chars):
        chunks.append(
            SearchChunk(
                doc_path,
                heading,
                len(chunks),
                part,
                doc_title=doc_title,
                heading_level=heading_level,
                section_path=section_path,
            )
        )


def chunk_markdown(path: Path, *, max_chars: int = 2600, min_chars: int = 650) -> list[SearchChunk]:
    text = clean_extracted_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    rel = str(path)
    doc_title = normalize_text(path.stem)
    heading = ""
    heading_level = 0
    heading_stack: list[str] = []
    chunks: list[SearchChunk] = []
    buf: list[str] = []
    doc_title_from_heading = False

    def flush() -> None:
        if not buf:
            return
        raw = "\n\n".join(buf).strip()
        buf.clear()
        if not raw:
            return
        section_path = tuple(item for item in heading_stack if item)
        _emit_chunk(
            chunks,
            rel,
            heading,
            raw,
            max_chars,
            doc_title=doc_title,
            heading_level=heading_level,
            section_path=section_path,
        )

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            heading_level = len(m.group(1))
            heading = normalize_text(m.group(2))
            if heading_level == 1 and heading and not _is_toc_title(heading) and not doc_title_from_heading:
                doc_title = heading
                doc_title_from_heading = True
            if len(heading_stack) < heading_level:
                heading_stack.extend([""] * (heading_level - len(heading_stack)))
            heading_stack[heading_level - 1] = heading
            del heading_stack[heading_level:]
            continue
        numbered_heading = _numbered_section_heading(line)
        if numbered_heading:
            flush()
            heading_level, heading = numbered_heading
            if len(heading_stack) < heading_level:
                heading_stack.extend([""] * (heading_level - len(heading_stack)))
            heading_stack[heading_level - 1] = heading
            del heading_stack[heading_level:]
            continue
        line = line.strip()
        if not line:
            continue
        current_len = sum(len(part) for part in buf)
        if buf and current_len + len(line) > max_chars:
            flush()
        buf.append(line)
        if sum(len(part) for part in buf) >= min_chars and re.search(r"[。.!?！？；;]$", line):
            flush()
    flush()
    return chunks


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _is_assessment_source_path(path: str | Path) -> bool:
    text = str(path)
    return bool(QUESTION_FILE_RE.search(text) or ANSWER_FILE_RE.search(text))


def load_doc_chunks(
    docs_dir: str | Path,
    *,
    max_chars: int = 2600,
    exclude_paths: Iterable[str | Path] | None = None,
) -> list[SearchChunk]:
    root = Path(docs_dir)
    if not root.exists():
        return []
    excluded = {_path_key(path) for path in (exclude_paths or [])}
    chunks: list[SearchChunk] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES):
        if _path_key(path) in excluded:
            continue
        if _is_assessment_source_path(path.relative_to(root)):
            continue
        try:
            chunks.extend(chunk_markdown(path, max_chars=max_chars))
        except OSError:
            continue
    return chunks


SECTION_RE = re.compile(r"(单选题?|多选题?|判断题?|填空题?|简答题?|选择题)")
QUESTION_NUM_RE = re.compile(r"^\s*(\d+)\s*[.、．]\s*(.+?)\s*$")
OPTION_START_RE = re.compile(r"\s+[A-Z]\s*[.．、]\s*")
OPTION_PAIR_RE = re.compile(r"([A-Z])\s*[.．、]\s*(.*?)(?=\s+[A-Z]\s*[.．、]\s*|$)")
ANSWER_PAIR_RE = re.compile(
    r"(\d+)\s*[:：.、]\s*"
    r"([A-Z](?:\s+[A-Z]){0,8}|[A-Z]{1,8}|[√×VXx对错正确错误]+)"
    r"(?=\s*(?:\d+\s*[:：.、]|[;；,，/]|$|\*|\)|）))"
)
INLINE_ANSWER_RE = re.compile(
    r"[（(]\s*([A-Z](?:\s*[,，、/;；]?\s*[A-Z]){0,7}|[√×VXx对错正确错误])\s*[）)]"
)
PLACEHOLDER_QUESTION_RE = re.compile(r"^[\s。.\-—_…·,，、;；:：!?！？()（）【】\[\]]+$")
CHINESE_SECTION_NUMS = "一二三四五六七八九十"
CHINESE_SECTION_RE = re.compile(rf"(?:^|\s)([{CHINESE_SECTION_NUMS}]+)\s*[、.．]\s*(.*?)(?=(?:\s+[{CHINESE_SECTION_NUMS}]+\s*[、.．])|$)", re.DOTALL)
ARABIC_SECTION_RE = re.compile(rf"(?:^|\s)(\d+)\s*[.、．]\s*(.*?)(?=(?:\s+\d+\s*[.、．]\s*)|(?:\s+[{CHINESE_SECTION_NUMS}]\s*[、.．])|$)", re.DOTALL)


def _strip_markdown_noise(text: str) -> str:
    text = text.replace("**", " ")
    text = re.sub(r"\*\[Image OCR\].*?\[End OCR\]\*", " ", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    return text


def _answer_tail(text: str) -> str:
    clean = _strip_markdown_noise(text)
    tail_pos = max(clean.rfind("参考答案"), clean.rfind("答案"))
    return clean[tail_pos:] if tail_pos >= 0 else clean


def _chinese_section_index(label: str) -> int | None:
    values = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    label = normalize_text(label)
    if label in values:
        return values[label]
    if label.startswith("十") and len(label) == 2 and label[1] in values:
        return 10 + values[label[1]]
    if label.endswith("十") and len(label) == 2 and label[0] in values:
        return values[label[0]] * 10
    if len(label) == 3 and label[1] == "十" and label[0] in values and label[2] in values:
        return values[label[0]] * 10 + values[label[2]]
    return None


def _section_type(text: str) -> str | None:
    match = SECTION_RE.search(text)
    if not match:
        return None
    label = match.group(1)
    if "多选" in label:
        return "multiple"
    if "判断" in label:
        return "bool"
    if "填空" in label:
        return "fill"
    if "简答" in label:
        return "short"
    if "单选" in label or "选择" in label:
        return "single"
    return None


def _compact_for_match(text: str) -> str:
    text = normalize_text(text).casefold()
    text = re.sub(r"【\d+】|\[\d+\]|\(\s*\)|（\s*）|_", "", text)
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def _question_core(text: str) -> str:
    text = normalize_text(text)
    text = OPTION_START_RE.split(text, maxsplit=1)[0]
    text = INLINE_ANSWER_RE.sub("（ ）", text)
    text = re.sub(r"^\d+\s*[.、．]\s*", "", text)
    return text.strip()


def _choice_options(text: str, qtype: str | None = None) -> tuple[tuple[str, str], ...]:
    options: list[tuple[str, str]] = []
    for letter, option_text in OPTION_PAIR_RE.findall(normalize_text(text)):
        option_text = INLINE_ANSWER_RE.sub("", option_text)
        option_text = normalize_text(option_text).strip(" ;；,，")
        if option_text:
            options.append((letter.upper(), option_text))
    if not options and qtype == "bool":
        return (("A", "对"), ("B", "错"))
    return tuple(options)


def _is_valid_question_bank_question(question: str) -> bool:
    question = normalize_text(question)
    if not question:
        return False
    if PLACEHOLDER_QUESTION_RE.fullmatch(question):
        return False
    compact = _compact_for_match(question)
    if len(compact) < 2:
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", compact))


def _visible_answer(answer: str, options: tuple[tuple[str, str], ...], qtype: str) -> str:
    if qtype not in {"single", "multiple", "bool"}:
        return normalize_text(answer)
    option_map = {letter.upper(): text for letter, text in options}
    details: list[str] = []
    for letter in re.findall(r"[A-Z]", normalize_text(answer).upper()):
        option_text = option_map.get(letter)
        if option_text:
            details.append(option_text)
    return "；".join(details) if details else normalize_text(answer)


def _strip_inline_answer_marks(text: str) -> str:
    return normalize_text(INLINE_ANSWER_RE.sub("（ ）", text))


def _answer_key_for_pair(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"参考答案|答案版|答案|试题版|试题|试卷|考核试卷|考试", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[-_ .（）()【】\[\]&]+", "", stem)
    return stem.casefold()


def _normalize_choice_answer(ans: str, qtype: str) -> str:
    ans = normalize_text(ans).strip("* ")
    ans = ans.replace("，", ",").replace("、", ",").replace(" ", ",")
    if qtype == "bool":
        folded = ans.casefold()
        if any(mark in folded for mark in ("√", "v", "对", "正确")):
            return "A"
        if any(mark in folded for mark in ("×", "x", "错", "错误")):
            return "B"
    letters = re.findall(r"[A-Z]", ans.upper())
    if qtype in {"single", "bool"}:
        return letters[0] if letters else ans
    if qtype == "multiple" and letters:
        return ",".join(letters)
    return ans


def _clean_answer_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r"^(?:答案|参考答案)\s*[:：]?\s*", "", line)
    line = re.sub(r"^\d+\s*[.、．]\s*", "", line)
    return line.strip(" *;；:：")


def _parse_question_items(text: str, path: str) -> tuple[list[QuestionBankItem], list[str]]:
    text = _strip_markdown_noise(text)
    lines = [normalize_text(line) for line in text.splitlines()]
    # Some converted exam files keep several numbered questions on one line.
    expanded: list[str] = []
    for line in lines:
        line = re.sub(r"\s+(?=\d+\s*[.、．]\s*[\u4e00-\u9fffA-Za-z])", "\n", line)
        expanded.extend(part.strip() for part in line.splitlines() if part.strip())

    items: list[QuestionBankItem] = []
    qtype_order: list[str] = []
    qtype_counts: Counter[str] = Counter()
    current_qtype: str | None = None
    pending_num: int | None = None
    pending_parts: list[str] = []

    def flush() -> None:
        nonlocal pending_num, pending_parts
        if current_qtype and pending_num is not None and pending_parts:
            raw_question = " ".join(pending_parts)
            question = _question_core(raw_question)
            if question and _section_type(question) is None and _is_valid_question_bank_question(question):
                qtype_counts[current_qtype] += 1
                items.append(
                    QuestionBankItem(
                        qtype=current_qtype,
                        ordinal=qtype_counts[current_qtype],
                        question=question,
                        answer="",
                        question_path=path,
                        answer_path="",
                        score_key=_compact_for_match(raw_question),
                        options=_choice_options(raw_question, current_qtype),
                    )
                )
        pending_num = None
        pending_parts = []

    for line in expanded:
        maybe_qtype = _section_type(line)
        numbered = QUESTION_NUM_RE.match(line)
        # Header lines such as "1. 填空题" set the section but are not questions.
        if maybe_qtype and (not numbered or _compact_for_match(numbered.group(2)) in {"单选题", "多选题", "判断题", "填空题", "简答题", "选择题"}):
            flush()
            current_qtype = maybe_qtype
            if maybe_qtype not in qtype_order:
                qtype_order.append(maybe_qtype)
            continue
        if maybe_qtype and re.search(r"每[题空]|题型|一[、.．]|二[、.．]|三[、.．]", line):
            flush()
            current_qtype = maybe_qtype
            if maybe_qtype not in qtype_order:
                qtype_order.append(maybe_qtype)
            continue
        if numbered:
            flush()
            pending_num = int(numbered.group(1))
            pending_parts = [numbered.group(2)]
            continue
        if pending_num is not None:
            pending_parts.append(line)
    flush()
    return items, qtype_order


def _inline_answer_type(answer: str, current_qtype: str | None) -> str:
    if current_qtype and current_qtype != "fill":
        return current_qtype
    normalized = normalize_text(answer)
    if re.search(r"[√×VXx对错正确错误]", normalized):
        return "bool"
    letters = re.findall(r"[A-Z]", normalized.upper())
    if len(letters) > 1:
        return "multiple"
    return "single"


def _parse_inline_answer_items(text: str, path: str) -> list[QuestionBankItem]:
    text = _strip_markdown_noise(text)
    lines = [normalize_text(line) for line in text.splitlines()]
    expanded: list[str] = []
    for line in lines:
        line = re.sub(r"\s+(?=\d+\s*[.、．]\s*[\u4e00-\u9fffA-Za-z])", "\n", line)
        expanded.extend(part.strip() for part in line.splitlines() if part.strip())

    items: list[QuestionBankItem] = []
    qtype_counts: Counter[str] = Counter()
    current_qtype: str | None = None
    pending_num: int | None = None
    pending_parts: list[str] = []

    def flush() -> None:
        nonlocal pending_num, pending_parts
        if pending_num is None or not pending_parts:
            pending_num = None
            pending_parts = []
            return
        body = " ".join(pending_parts)
        answer_match = INLINE_ANSWER_RE.search(body)
        if not answer_match:
            pending_num = None
            pending_parts = []
            return
        qtype = _inline_answer_type(answer_match.group(1), current_qtype)
        question = _question_core(body)
        answer = _normalize_choice_answer(answer_match.group(1), qtype)
        options = _choice_options(body, qtype)
        if question and _section_type(question) is None and answer and _is_valid_question_bank_question(question):
            qtype_counts[qtype] += 1
            items.append(
                QuestionBankItem(
                    qtype=qtype,
                    ordinal=qtype_counts[qtype],
                    question=question,
                    answer=answer,
                    question_path=path,
                    answer_path=path,
                    score_key=_compact_for_match(body),
                    options=options,
                    answer_detail=_visible_answer(answer, options, qtype),
                )
            )
        pending_num = None
        pending_parts = []

    for line in expanded:
        maybe_qtype = _section_type(line)
        numbered = QUESTION_NUM_RE.match(line)
        if maybe_qtype and (not numbered or re.search(r"每[题空]|题型|一[、.．]|二[、.．]|三[、.．]", line)):
            flush()
            current_qtype = maybe_qtype
            continue
        if numbered:
            flush()
            pending_num = int(numbered.group(1))
            pending_parts = [numbered.group(2)]
            continue
        if pending_num is not None:
            pending_parts.append(line)
    flush()
    return items


def _split_answer_sections(text: str) -> dict[str, str]:
    clean = _strip_markdown_noise(text)
    matches = list(SECTION_RE.finditer(clean))
    sections: dict[str, list[str]] = defaultdict(list)
    for idx, match in enumerate(matches):
        qtype = _section_type(match.group(1))
        if not qtype:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        sections[qtype].append(clean[start:end])
    return {qtype: "\n".join(parts) for qtype, parts in sections.items()}


def _parse_explicit_answer_pairs(section: str, qtype: str) -> dict[int, str]:
    found: dict[int, str] = {}
    for m in ANSWER_PAIR_RE.finditer(section):
        idx = int(m.group(1))
        found[idx] = _normalize_choice_answer(m.group(2), qtype)
    return found


def _parse_compact_answers(section: str, qtype: str) -> list[str]:
    if qtype in {"fill", "short"}:
        raw = unicodedata.normalize("NFKC", section or "").replace("\u3000", " ")
        lines = [_clean_answer_line(line) for line in raw.splitlines()]
        return [line for line in lines if line and _section_type(line) is None]
    section = normalize_text(section)
    section = re.sub(r"^(?:答案|参考答案)\s*[:：]?", "", section)
    if qtype in {"single", "bool"}:
        tokens = re.findall(r"[A-Z]+|[√×VXx对错]", section)
        out: list[str] = []
        for token in tokens:
            if qtype == "single" and re.fullmatch(r"[A-Z]{2,}", token):
                out.extend(_normalize_choice_answer(ch, qtype) for ch in token)
            elif qtype == "bool" and re.fullmatch(r"[ABVX]+", token.upper()) and len(token) > 1:
                out.extend(_normalize_choice_answer(ch, qtype) for ch in token)
            else:
                out.append(_normalize_choice_answer(token, qtype))
        return out
    if qtype == "multiple":
        tokens = re.findall(r"[A-Z]{1,8}", section)
        return [_normalize_choice_answer(token, qtype) for token in tokens]
    return []


def _compact_answer_group(answers: list[str], qtype: str, question_counts: Counter[str]) -> AnswerGroup | None:
    expected_count = question_counts.get(qtype, 0)
    if not answers or expected_count <= 0:
        return None
    if len(answers) != expected_count:
        return None
    return AnswerGroup({idx: answer for idx, answer in enumerate(answers, 1)}, "compact_exact_count")


def _answer_group_from_body(body: str, qtype: str, question_counts: Counter[str]) -> AnswerGroup | None:
    explicit = _parse_explicit_answer_pairs(body, qtype)
    if explicit:
        return AnswerGroup(explicit, "explicit_numbered")
    return _compact_answer_group(_parse_compact_answers(body, qtype), qtype, question_counts)


def _parse_numbered_answer_groups(text: str, qtype_order: list[str], question_counts: Counter[str]) -> dict[str, AnswerGroup]:
    text = unicodedata.normalize("NFKC", _answer_tail(text)).replace("\u3000", " ")
    out: dict[str, AnswerGroup] = {}
    for label, body in ARABIC_SECTION_RE.findall(text):
        group_idx = int(label) - 1
        if group_idx < 0 or group_idx >= len(qtype_order):
            break
        qtype = qtype_order[group_idx]
        group = _answer_group_from_body(body, qtype, question_counts)
        if group:
            out[qtype] = group

    for label, body in CHINESE_SECTION_RE.findall(text):
        idx = _chinese_section_index(label)
        if idx is None or idx < 1 or idx > len(qtype_order):
            continue
        explicit_qtype = _section_type(body)
        qtype = explicit_qtype or qtype_order[idx - 1]
        group = _answer_group_from_body(body, qtype, question_counts)
        if group:
            out[qtype] = group
    return out


def _parse_answers(text: str, qtype_order: list[str], question_counts: Counter[str]) -> dict[str, AnswerGroup]:
    answer_text = _answer_tail(text)
    sections = _split_answer_sections(answer_text)
    out: dict[str, AnswerGroup] = {}
    for qtype, body in sections.items():
        group = _answer_group_from_body(body, qtype, question_counts)
        if group:
            out[qtype] = group
    numbered = _parse_numbered_answer_groups(answer_text, qtype_order, question_counts)
    for qtype, group in numbered.items():
        existing = out.get(qtype)
        if existing is None or group.source == "explicit_numbered" or len(group.answers_by_ordinal) >= len(existing.answers_by_ordinal):
            out[qtype] = group
    return out


class QuestionBankIndex:
    def __init__(self, docs_dir: str | Path | None = None):
        self.docs_dir = Path(docs_dir) if docs_dir else None
        self.items: list[QuestionBankItem] = []
        self.question_source_paths: set[str] = set()
        self._cache: dict[tuple[str, str, int], tuple[QuestionBankHit, ...]] = {}
        self._build()

    def _build(self) -> None:
        root = self.docs_dir
        if root is None or not root.is_dir():
            return
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
        answer_files = [p for p in files if ANSWER_FILE_RE.search(str(p.relative_to(root)))]
        question_files = [
            p
            for p in files
            if QUESTION_FILE_RE.search(str(p.relative_to(root))) and not ANSWER_FILE_RE.search(str(p.relative_to(root)))
        ]
        by_dir: dict[Path, list[Path]] = defaultdict(list)
        for path in question_files:
            by_dir[path.parent].append(path)

        pairs: list[tuple[Path, Path]] = []
        for answer_path in answer_files:
            question_path = self._best_question_file(answer_path, by_dir.get(answer_path.parent, []))
            if question_path is None:
                question_path = self._best_question_file(answer_path, question_files)
            if question_path is not None:
                pairs.append((question_path, answer_path))
            elif QUESTION_FILE_RE.search(str(answer_path.relative_to(root))) or "答案版" in answer_path.stem:
                pairs.append((answer_path, answer_path))

        seen_pairs: set[tuple[str, str]] = set()
        paired_question_paths: set[str] = set()
        for question_path, answer_path in pairs:
            key = (str(question_path), str(answer_path))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            items = self._items_from_pair(question_path, answer_path)
            if not items:
                continue
            self.items.extend(items)
            self.question_source_paths.add(str(question_path))
            paired_question_paths.add(str(question_path))

        for question_path in question_files:
            if str(question_path) in paired_question_paths:
                continue
            items = self._items_from_pair(question_path, question_path)
            if not items:
                continue
            self.items.extend(items)
            self.question_source_paths.add(str(question_path))

    def _best_question_file(self, answer_path: Path, candidates: list[Path]) -> Path | None:
        answer_key = _answer_key_for_pair(answer_path)
        scored: list[tuple[float, Path]] = []
        for candidate in candidates:
            score = SequenceMatcher(None, answer_key, _answer_key_for_pair(candidate)).ratio()
            scored.append((score, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_path = scored[0]
        if best_score < 0.72:
            return None
        if answer_key == _answer_key_for_pair(best_path):
            return best_path
        if len(scored) > 1 and best_score - scored[1][0] < 0.08:
            return None
        return best_path

    def _items_from_pair(self, question_path: Path, answer_path: Path) -> list[QuestionBankItem]:
        try:
            question_text = question_path.read_text(encoding="utf-8", errors="ignore")
            answer_text = answer_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        if _looks_like_conversion_failure_manifest(question_text) or _looks_like_conversion_failure_manifest(answer_text):
            return []
        question_items, qtype_order = _parse_question_items(question_text, str(question_path))
        inline_items = _parse_inline_answer_items(answer_text, str(answer_path))
        if not question_items:
            return inline_items

        question_counts = Counter(item.qtype for item in question_items)
        answers = _parse_answers(answer_text, qtype_order, question_counts)
        inline_by_type: dict[str, list[QuestionBankItem]] = defaultdict(list)
        for inline_item in inline_items:
            inline_by_type[inline_item.qtype].append(inline_item)
        out: list[QuestionBankItem] = []
        for item in question_items:
            q_answers = answers.get(item.qtype)
            answer = ""
            answer_path_for_item = str(answer_path)
            if q_answers is not None:
                answer = q_answers.answers_by_ordinal.get(item.ordinal, "")
            if not answer:
                inline_for_type = inline_by_type.get(item.qtype, [])
                if item.ordinal <= len(inline_for_type):
                    inline_item = inline_for_type[item.ordinal - 1]
                    answer = inline_item.answer
                    answer_path_for_item = inline_item.answer_path
                    if not item.options and inline_item.options:
                        item_options = inline_item.options
                    else:
                        item_options = item.options
                else:
                    item_options = item.options
            else:
                item_options = item.options
            if not answer or not _is_valid_question_bank_question(item.question):
                continue
            out.append(
                QuestionBankItem(
                    qtype=item.qtype,
                    ordinal=item.ordinal,
                    question=item.question,
                    answer=answer,
                    question_path=item.question_path,
                    answer_path=answer_path_for_item,
                    score_key=item.score_key,
                    options=item_options,
                    answer_detail=_visible_answer(answer, item_options, item.qtype),
                )
            )
        return out

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        terms = {_compact_for_match(term) for term in query_keywords(query)}
        norm = normalize_text(query)
        terms.update(_compact_for_match(term) for term in EN_TOKEN_RE.findall(norm) if len(term) >= 2)
        terms.update(_compact_for_match(term) for term in cjk_ngrams(norm, min_n=2, max_n=4))
        weak_terms = {_compact_for_match(term) for term in WEAK_STANDALONE_QUERY_TERMS}
        return {
            term
            for term in terms
            if len(term) >= 2
            and term not in weak_terms
            and not _is_generic_search_term(term)
        }

    def search(self, query: str, *, answer_type: str = "", top_k: int = 3) -> tuple[QuestionBankHit, ...]:
        query_key = _compact_for_match(_question_core(query))
        cache_key = (query_key, answer_type, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not query_key or not self.items:
            return tuple()
        query_terms = self._query_terms(query)
        scored: list[QuestionBankHit] = []
        for item in self.items:
            if answer_type and item.qtype != answer_type:
                continue
            matched_terms = [term for term in query_terms if term in item.score_key]
            long_matches = [term for term in matched_terms if len(term) >= 4]
            english_matches = [term for term in matched_terms if re.search(r"[a-z]", term, flags=re.IGNORECASE)]
            cjk_matches = [term for term in matched_terms if re.search(r"[\u4e00-\u9fff]", term)]
            coverage = len(matched_terms) / max(len(query_terms), 1)
            ratio = SequenceMatcher(None, query_key, item.score_key).ratio()
            score = 0.65 * coverage + 0.35 * ratio + min(len(long_matches), 4) * 0.03
            if query_key in item.score_key or item.score_key in query_key:
                score = max(score, 0.96)
            technical_short_overlap = len(matched_terms) >= 4 and len(english_matches) >= 2 and len(cjk_matches) >= 1
            cjk_phrase_overlap = len(long_matches) >= 1 and len(cjk_matches) >= 3
            strong_overlap = (
                len(long_matches) >= 2
                or technical_short_overlap
                or cjk_phrase_overlap
                or ratio >= 0.72
                or query_key in item.score_key
                or item.score_key in query_key
            )
            if (
                (score >= 0.64 and strong_overlap)
                or (len(long_matches) >= 2 and coverage >= 0.4)
                or (technical_short_overlap and coverage >= 0.35)
                or (cjk_phrase_overlap and coverage >= 0.35)
            ):
                scored.append(QuestionBankHit(item, score))
        scored.sort(key=lambda hit: (hit.score, len(hit.item.score_key)), reverse=True)
        result = tuple(scored[:top_k])
        if len(self._cache) >= 1024:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result


def format_question_bank_item_text(item: QuestionBankItem) -> str:
    answer = item.answer_detail or normalize_text(item.answer)
    return "\n".join(
        [
            f"题目：{normalize_text(item.question)}",
            f"答案：{answer}",
        ]
    )


def format_question_bank_hits(hits: tuple[QuestionBankHit, ...], *, max_tokens: int = 280) -> tuple[str, tuple[QuestionBankHit, ...]]:
    remaining = max_tokens
    parts: list[str] = []
    shown: list[QuestionBankHit] = []
    for i, hit in enumerate(hits, 1):
        item = hit.item
        text = f"{i}. {format_question_bank_item_text(item)}"
        if remaining <= estimate_tokens(text) + 8:
            break
        parts.append(text)
        shown.append(hit)
        remaining -= estimate_tokens(text)
    return "\n".join(parts), tuple(shown)


class SimpleSearchEngine:
    def __init__(self, docs_dir: str | Path | None = None, *, chunks: list[SearchChunk] | None = None):
        self.docs_dir = docs_dir or ""
        self.question_bank = QuestionBankIndex(self.docs_dir if chunks is None else None)
        self.chunks = (
            chunks
            if chunks is not None
            else load_doc_chunks(self.docs_dir, exclude_paths=self.question_bank.question_source_paths)
        )
        self._doc_chunks: dict[str, list[SearchChunk]] = {}
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._postings: dict[str, set[int]] = {}
        self._cache: dict[tuple[str, int], tuple[SearchHit, ...]] = {}
        self._catalog_cache: dict[tuple[str, int], tuple[DocumentSectionHit, ...]] = {}
        self._catalog_sections: list[DocumentSection] = []
        self._whoosh_storage = None
        self._whoosh_index = None
        self._whoosh_tempdir = None
        self._build()

    def search_question_bank(self, query: str, *, answer_type: str = "", top_k: int = 3) -> tuple[QuestionBankHit, ...]:
        return self.question_bank.search(query, answer_type=answer_type, top_k=top_k)

    def _build(self) -> None:
        for chunk in self.chunks:
            self._doc_chunks.setdefault(chunk.doc_path, []).append(chunk)
            tf = Counter(feature_tokens(chunk.text, heading=chunk.heading, path=chunk.doc_path))
            self._tf.append(tf)
            self._lengths.append(sum(tf.values()) or 1)
            self._df.update(tf.keys())
            idx = len(self._tf) - 1
            for term in tf:
                self._postings.setdefault(term, set()).add(idx)
        for doc_chunks in self._doc_chunks.values():
            doc_chunks.sort(key=lambda c: c.chunk_id)
        self._catalog_sections = self._build_document_catalog()
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)
        if self._should_build_whoosh_index():
            self._build_whoosh_index()

    def _build_document_catalog(self) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        for doc_path, doc_chunks in self._doc_chunks.items():
            if not doc_chunks:
                continue
            content_chunks = [
                chunk for chunk in doc_chunks if not _looks_like_toc(chunk.text, heading=chunk.heading)
            ]
            if not content_chunks:
                continue
            title_candidates = [
                *(chunk.doc_title for chunk in content_chunks if chunk.doc_title),
                Path(doc_path).stem,
                *(chunk.heading for chunk in content_chunks if chunk.heading),
            ]
            doc_title = next(
                (
                    title
                    for title in (normalize_text(candidate) for candidate in title_candidates)
                    if title and not _is_toc_title(title)
                ),
                normalize_text(Path(doc_path).stem),
            )
            doc_search_text = " ".join(
                [
                    doc_title,
                    Path(doc_path).stem,
                    " ".join(normalize_text(chunk.heading) for chunk in content_chunks[:12] if chunk.heading),
                    " ".join(_catalog_text_sample(chunk.text) for chunk in content_chunks[:3]),
                ]
            )
            sections.append(
                DocumentSection(
                    doc_path=doc_path,
                    doc_title=doc_title,
                    section_title=doc_title,
                    section_path=(doc_title,),
                    heading_level=1,
                    chunk_id=content_chunks[0].chunk_id,
                    search_text=doc_search_text,
                    feature_terms=tuple(feature_tokens(doc_search_text)),
                )
            )

            grouped: dict[tuple[str, ...], list[SearchChunk]] = defaultdict(list)
            for chunk in content_chunks:
                path = tuple(normalize_text(part) for part in (chunk.section_path or (chunk.heading,)) if normalize_text(part))
                if not path or _section_path_has_toc(path):
                    continue
                grouped[path].append(chunk)

            for section_path, chunks in grouped.items():
                chunks = [chunk for chunk in chunks if not _looks_like_toc(chunk.text, heading=chunk.heading)]
                if not chunks:
                    continue
                section_title = section_path[-1]
                if section_title == doc_title and len(section_path) == 1:
                    continue
                first = chunks[0]
                search_text = " ".join(
                    [
                        doc_title,
                        Path(doc_path).stem,
                        " ".join(section_path),
                        " ".join(_catalog_text_sample(chunk.text) for chunk in chunks[:3]),
                    ]
                )
                sections.append(
                    DocumentSection(
                        doc_path=doc_path,
                        doc_title=doc_title,
                        section_title=section_title,
                        section_path=section_path,
                        heading_level=first.heading_level or len(section_path),
                        chunk_id=first.chunk_id,
                        search_text=search_text,
                        feature_terms=tuple(feature_tokens(search_text)),
                    )
                )
        return sections

    def _should_build_whoosh_index(self) -> bool:
        if _whoosh_fields is None:
            return False
        if str(os.environ.get("QA_RL_ENABLE_WHOOSH_INDEX", "")).strip() != "1":
            return False
        try:
            max_chunks = int(os.environ.get("QA_RL_WHOOSH_MAX_CHUNKS", "5000"))
        except ValueError:
            max_chunks = 5000
        return len(self.chunks) <= max_chunks

    def _build_whoosh_index(self) -> None:
        if (
            _whoosh_fields is None
            or _WhooshFileStorage is None
            or _WhooshRegexTokenizer is None
            or _WhooshLowercaseFilter is None
            or not self.chunks
        ):
            return
        analyzer = _WhooshRegexTokenizer(r"[^ \t\r\n]+") | _WhooshLowercaseFilter()
        schema = _whoosh_fields.Schema(
            idx=_whoosh_fields.ID(stored=True, unique=True),
            title=_whoosh_fields.TEXT(stored=False, analyzer=analyzer, field_boost=3.0),
            path=_whoosh_fields.TEXT(stored=False, analyzer=analyzer, field_boost=2.2),
            body=_whoosh_fields.TEXT(stored=False, analyzer=analyzer, field_boost=1.0),
        )
        tempdir = tempfile.TemporaryDirectory(prefix="qa_whoosh_")
        storage = _WhooshFileStorage(tempdir.name)
        try:
            index = storage.create_index(schema)
            writer = index.writer()
            for idx, chunk in enumerate(self.chunks):
                clean_body = clean_retrieval_snippet(chunk.text) or normalize_text(chunk.text)
                writer.add_document(
                    idx=str(idx),
                    title=_lexical_document_text(chunk.heading),
                    path=_lexical_document_text(Path(chunk.doc_path).stem),
                    body=_lexical_document_text(clean_body),
                )
            writer.commit()
        except Exception:
            tempdir.cleanup()
            self._whoosh_storage = None
            self._whoosh_index = None
            self._whoosh_tempdir = None
            return
        self._whoosh_storage = storage
        self._whoosh_index = index
        self._whoosh_tempdir = tempdir

    def _whoosh_candidates(self, query: str, *, limit: int) -> dict[int, float]:
        if (
            self._whoosh_index is None
            or _whoosh_scoring is None
            or _WhooshMultifieldParser is None
            or _WhooshOrGroup is None
        ):
            return {}
        tokens = lexical_tokens(query, for_query=True)
        if not tokens:
            return {}
        query_text = " ".join(tokens)
        weighting = _whoosh_scoring.BM25F(field_B={"title": 0.4, "path": 0.45, "body": 0.9})
        with self._whoosh_index.searcher(weighting=weighting) as searcher:
            parser = _WhooshMultifieldParser(
                ["title", "path", "body"],
                schema=self._whoosh_index.schema,
                group=_WhooshOrGroup.factory(0.85),
            )
            parsed = parser.parse(query_text)
            results = searcher.search(parsed, limit=limit)
            return {int(hit["idx"]): float(hit.score) for hit in results}

    def _candidate_indices(self, query_features: Iterable[str], *, top_k: int) -> set[int]:
        terms = _dedupe_ordered(str(term) for term in query_features if term)
        if not terms:
            return set()
        n_docs = max(len(self.chunks), 1)
        high_df_cutoff = max(800, int(n_docs * 0.12))
        weak_terms = {_compact_for_match(term) for term in WEAK_STANDALONE_QUERY_TERMS}
        terms_with_df = [
            (term, self._df.get(term, 0))
            for term in terms
            if self._df.get(term, 0)
            and _compact_for_match(term) not in weak_terms
            and not _is_generic_search_term(term)
        ]
        if not terms_with_df:
            return set()
        selective = [(term, df) for term, df in terms_with_df if df <= high_df_cutoff]
        if not selective:
            selective = sorted(terms_with_df, key=lambda item: item[1])[:24]
        else:
            selective.sort(key=lambda item: item[1])

        out: set[int] = set()
        max_terms = 48
        soft_cap = max(top_k * 240, 1200)
        for term, _df in selective[:max_terms]:
            out.update(self._postings.get(term, ()))
            if len(out) >= soft_cap:
                break
        return out

    def _bm25(self, query_features: list[str], idx: int) -> float:
        if not self.chunks:
            return 0.0
        tf = self._tf[idx]
        dl = self._lengths[idx]
        score = 0.0
        k1 = 1.4
        b = 0.75
        n_docs = len(self.chunks)
        for term in query_features:
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * dl / max(self._avg_len, 1e-9))
            score += idf * freq * (k1 + 1) / denom
        return score

    def _outline_expansion_hits(self, query: str, scored: list[SearchHit], *, limit: int) -> list[SearchHit]:
        query_terms = tuple(proper_query_terms(query)) or query_keywords(query)
        if not query_terms:
            return []

        out: list[SearchHit] = []
        seen: set[tuple[str, int]] = {(hit.chunk.doc_path, hit.chunk.chunk_id) for hit in scored}
        outline_hits = [
            hit
            for hit in sorted(scored, key=lambda item: item.score, reverse=True)[: max(limit * 2, 6)]
            if _looks_like_toc(hit.chunk.text, heading=hit.chunk.heading)
        ]
        for outline_hit in outline_hits:
            matched_entries = [
                entry for entry in _toc_entries(outline_hit.chunk.text) if _text_has_any_query_term(entry, query_terms)
            ]
            for entry in matched_entries[:3]:
                entry_terms = tuple(proper_query_terms(entry)) or query_keywords(entry)
                if not entry_terms:
                    continue
                per_entry = 0
                for chunk in self._doc_chunks.get(outline_hit.chunk.doc_path, []):
                    key = (chunk.doc_path, chunk.chunk_id)
                    if key in seen or _looks_like_toc(chunk.text, heading=chunk.heading):
                        continue
                    haystack = " ".join([chunk.heading, chunk.text])
                    if normalize_text(entry).casefold() not in normalize_text(haystack).casefold() and not _text_has_any_query_term(
                        haystack,
                        entry_terms,
                    ):
                        continue
                    matched = matched_query_keywords(query, text=chunk.text, heading=chunk.heading, path=chunk.doc_path)
                    proper_matches = proper_term_match_count(
                        query,
                        text=chunk.text,
                        heading=chunk.heading,
                        path=chunk.doc_path,
                    )
                    score = outline_hit.score + 1.5 + len(matched) + 2.0 * proper_matches
                    out.append(SearchHit(chunk, score, query, matched))
                    seen.add(key)
                    per_entry += 1
                    if per_entry >= 2 or len(out) >= limit:
                        break
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out

    def search(self, query: str, top_k: int = 4) -> tuple[SearchHit, ...]:
        query = clean_query(query, max_chars=96)
        cache_key = (query.casefold(), top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not query or not self.chunks:
            return tuple()
        profile = build_query_profile(query)
        q_norm = normalize_text(query).casefold()
        q_features = list(profile.lexical_terms) or feature_tokens(query)
        q_keywords = profile.keywords
        q_proper_terms = profile.proper_terms
        q_mandatory_terms = profile.mandatory_terms
        q_quantity_required = profile.quantity_required
        q_concept_groups = profile.concept_groups
        q_domain_anchors = query_domain_anchors(query)
        english_terms = {t.casefold() for t in EN_TOKEN_RE.findall(query)}
        whoosh_scores = self._whoosh_candidates(query, limit=max(top_k * 12, 48))
        candidate_features = list(profile.lexical_terms)
        candidate_features.extend(q_proper_terms)
        candidate_features.extend(q_mandatory_terms)
        candidate_features.extend(q_quantity_required)
        for phrase in profile.english_phrases:
            candidate_features.extend(lexical_tokens(phrase, for_query=False))
        candidate_indices = self._candidate_indices(candidate_features, top_k=top_k)
        candidate_indices.update(whoosh_scores.keys())
        if not candidate_indices:
            return tuple()
        scored: list[SearchHit] = []

        for idx in candidate_indices:
            chunk = self.chunks[idx]
            text_norm = normalize_text(chunk.text).casefold()
            heading_norm = normalize_text(chunk.heading).casefold()
            score = self._bm25(q_features, idx)
            if q_norm and q_norm in text_norm:
                score += 1.5
            if q_norm and q_norm in heading_norm:
                score += 0.8
            if english_terms and any(t in text_norm for t in english_terms):
                score += 0.5
            if idx in whoosh_scores:
                score += 0.75 * whoosh_scores[idx]
            proper_matches = proper_term_match_count_from_terms(
                q_proper_terms,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            mandatory_matches = proper_term_match_count_from_terms(
                q_mandatory_terms,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            domain_matches = domain_anchor_match_count(
                query,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            if q_mandatory_terms and mandatory_matches == 0 and domain_matches == 0:
                continue
            if q_domain_anchors and domain_matches == 0:
                continue
            quantity_score = count_evidence_score(
                query,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            if q_quantity_required and quantity_score <= 0:
                continue
            core_stats = _core_match_stats_for_groups(
                q_concept_groups,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            if core_stats.total and core_stats.matched == 0:
                continue
            high_value_matches = high_value_match_terms(
                profile,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            phrase_score = phrase_match_score(
                profile,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            if core_stats.matched:
                score += 1.4 * core_stats.matched
            if proper_matches:
                score += 2.4 * proper_matches
            elif q_proper_terms:
                score -= 0.25
            if mandatory_matches:
                score += 2.0 * mandatory_matches
            if domain_matches:
                score += 3.0 * domain_matches
            if quantity_score:
                score += 2.2 * quantity_score
            if high_value_matches:
                coverage = len(high_value_matches) / max(len(profile.high_value_terms), 1)
                score += 2.0 * len(high_value_matches) + 2.5 * coverage
            if phrase_score:
                score += phrase_score
            if _looks_like_toc(chunk.text, heading=chunk.heading):
                score -= 0.8
            matched_terms = matched_query_keywords(
                query,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            if core_stats.matched_units:
                matched_terms = _dedupe_ordered((*matched_terms, *core_stats.matched_units))
            if high_value_matches:
                matched_terms = _dedupe_ordered((*matched_terms, *high_value_matches))
            if q_quantity_required:
                matched_terms = _dedupe_ordered((*matched_terms, *q_quantity_required))
            if q_keywords:
                score += 0.25 * len(matched_terms)
            if score > 0:
                scored.append(SearchHit(chunk, score, query, matched_terms))

        scored.extend(self._outline_expansion_hits(query, scored, limit=max(top_k * 2, 4)))
        scored.sort(key=lambda h: (len(h.matched_terms), h.score), reverse=True)
        diverse: list[SearchHit] = []
        seen_docs: Counter[str] = Counter()
        seen_text_keys: list[str] = []
        for hit in scored:
            if is_near_duplicate_text(hit.chunk.text, seen_text_keys, threshold=0.86):
                continue
            text_key = duplicate_text_key(hit.chunk.text)
            penalty = 0.45 * seen_docs[hit.chunk.doc_path]
            adjusted = SearchHit(hit.chunk, hit.score - penalty, hit.query, hit.matched_terms)
            diverse.append(adjusted)
            seen_docs[hit.chunk.doc_path] += 1
            seen_text_keys.append(text_key)
            if len(diverse) >= max(top_k * 3, top_k):
                break
        diverse.sort(key=lambda h: (len(h.matched_terms), h.score), reverse=True)
        result = tuple(diverse[:top_k])
        if len(self._cache) >= 2048:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result

    def search_catalog(self, query: str, top_k: int = 6) -> tuple[DocumentSectionHit, ...]:
        """Search document titles and section paths, returning catalog entries."""
        query = clean_query(query, max_chars=96)
        cache_key = (query.casefold(), top_k)
        if cache_key in self._catalog_cache:
            return self._catalog_cache[cache_key]
        if not query or not self._catalog_sections:
            return tuple()

        profile = build_query_profile(query)
        query_terms = set(profile.lexical_terms)
        query_terms.update(profile.proper_terms)
        query_terms.update(profile.mandatory_terms)
        query_terms.update(profile.high_value_terms)
        q_domain_anchors = query_domain_anchors(query)
        scored: list[DocumentSectionHit] = []
        for section in self._catalog_sections:
            if _is_toc_title(section.section_title) or _section_path_has_toc(section.section_path):
                continue
            catalog_text = " ".join([section.doc_title, " ".join(section.section_path), Path(section.doc_path).stem])
            search_text = section.search_text
            catalog_features = set(feature_tokens(catalog_text))
            hidden_features = set(section.feature_terms)
            catalog_overlap = query_terms & catalog_features
            hidden_overlap = query_terms & hidden_features
            if not catalog_overlap and not hidden_overlap:
                continue
            matched = matched_query_keywords(query, text=search_text, heading=section.section_title, path=section.doc_path)
            proper_matches = proper_term_match_count(query, text=search_text, heading=section.section_title, path=section.doc_path)
            domain_matches = domain_anchor_match_count(
                query,
                text=search_text,
                heading=section.section_title,
                path=section.doc_path,
            )
            if q_domain_anchors and domain_matches == 0:
                continue
            phrase_score = phrase_match_score(profile, text=search_text, heading=section.section_title, path=section.doc_path)
            high_value_matches = high_value_match_terms(
                profile,
                text=search_text,
                heading=section.section_title,
                path=section.doc_path,
            )
            if not (catalog_overlap or hidden_overlap or matched or proper_matches or phrase_score or high_value_matches):
                continue
            score = 0.0
            score += 2.8 * len(catalog_overlap)
            score += 0.55 * len(hidden_overlap)
            score += 2.2 * len(matched)
            score += 3.0 * proper_matches
            score += 3.5 * domain_matches
            score += phrase_score
            score += 2.0 * len(high_value_matches)
            if section.section_path == (section.doc_title,):
                score -= 0.5
            else:
                score += min(len(section.section_path), 6) * 0.25
            if any(term in normalize_search_terms(catalog_text).casefold() for term in profile.proper_terms):
                score += 2.0
            if score > 0:
                scored.append(DocumentSectionHit(section, score, query, tuple(matched)))

        scored.sort(
            key=lambda hit: (
                len(hit.matched_terms),
                hit.score,
                hit.section.heading_level,
                len(hit.section.section_path),
            ),
            reverse=True,
        )
        out: list[DocumentSectionHit] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        doc_counts: Counter[str] = Counter()
        for hit in scored:
            key = (hit.section.doc_path, hit.section.section_path)
            if key in seen:
                continue
            if doc_counts[hit.section.doc_path] >= 3:
                continue
            out.append(hit)
            seen.add(key)
            doc_counts[hit.section.doc_path] += 1
            if len(out) >= max(top_k * 2, top_k):
                break
        result = tuple(out[: max(top_k * 2, top_k)])
        if len(self._catalog_cache) >= 1024:
            self._catalog_cache.pop(next(iter(self._catalog_cache)))
        self._catalog_cache[cache_key] = result
        return result

    def read_context(self, doc_path: str, chunk_id: int, *, radius: int = 1) -> tuple[SearchChunk, ...]:
        """Return the requested chunk plus neighboring chunks from the same document."""
        doc_chunks = self._doc_chunks.get(doc_path, [])
        if not doc_chunks:
            return tuple()
        center = None
        for idx, chunk in enumerate(doc_chunks):
            if chunk.chunk_id == chunk_id:
                center = idx
                break
        if center is None:
            return tuple()
        start = max(center - max(radius, 0), 0)
        end = min(center + max(radius, 0) + 1, len(doc_chunks))
        return tuple(doc_chunks[start:end])

    def read_relevant_context(
        self,
        doc_path: str,
        chunk_id: int,
        *,
        query: str,
        radius: int = 1,
        top_k: int = 3,
        prefer_relevant: bool = False,
    ) -> tuple[SearchChunk, ...]:
        """Return neighboring chunks plus other high-signal chunks from the same document."""
        nearby = list(self.read_context(doc_path, chunk_id, radius=radius))
        doc_chunks = self._doc_chunks.get(doc_path, [])
        if not doc_chunks:
            return tuple(nearby)

        query_features = feature_tokens(query)
        query_features.extend(query_expansion_tokens(query))
        query_terms = set(query_features)
        if not query_terms:
            return tuple(nearby)

        profile = build_query_profile(query)
        q_domain_anchors = query_domain_anchors(query)
        nearby_keys = {(chunk.doc_path, chunk.chunk_id) for chunk in nearby}
        scored: list[tuple[float, SearchChunk]] = []
        for chunk in doc_chunks:
            key = (chunk.doc_path, chunk.chunk_id)
            if key in nearby_keys and not prefer_relevant:
                continue
            chunk_terms = set(feature_tokens(chunk.text, heading=chunk.heading, path=chunk.doc_path))
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            matched = matched_query_keywords(query, text=chunk.text, heading=chunk.heading, path=chunk.doc_path)
            proper_matches = proper_term_match_count(query, text=chunk.text, heading=chunk.heading, path=chunk.doc_path)
            domain_matches = domain_anchor_match_count(
                query,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            if q_domain_anchors and domain_matches == 0:
                continue
            phrase_score = phrase_match_score(profile, text=chunk.text, heading=chunk.heading, path=chunk.doc_path)
            high_value_matches = high_value_match_terms(
                profile,
                text=chunk.text,
                heading=chunk.heading,
                path=chunk.doc_path,
            )
            score = (
                len(overlap)
                + 2.0 * len(matched)
                + 3.0 * proper_matches
                + 3.5 * domain_matches
                + phrase_score
                + 1.5 * len(high_value_matches)
            )
            scored.append((score, chunk))
        scored.sort(key=lambda item: (item[0], -abs(item[1].chunk_id - chunk_id)), reverse=True)

        out: list[SearchChunk] = []
        seen: set[tuple[str, int]] = set()
        if prefer_relevant:
            ordered_chunks = [chunk for _, chunk in scored[: max(top_k, 1)]] + nearby
        else:
            ordered_chunks = nearby + [chunk for _, chunk in scored[:top_k]]
        for chunk in ordered_chunks:
            key = (chunk.doc_path, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)

        min_readable_chunks = min(max(top_k, 1), 4)
        readable_chunks = sum(1 for chunk in out if clean_retrieval_snippet(chunk.text))
        if readable_chunks < min_readable_chunks:
            center_idx = None
            for idx, chunk in enumerate(doc_chunks):
                if chunk.chunk_id == chunk_id:
                    center_idx = idx
                    break
            max_scan_chunks = min(len(doc_chunks), max(top_k * 4, radius * 2 + top_k + 8, 24))
            if center_idx is not None:
                scanned = 0
                for distance in range(len(doc_chunks)):
                    candidate_indices = [center_idx] if distance == 0 else [center_idx - distance, center_idx + distance]
                    for idx in candidate_indices:
                        if idx < 0 or idx >= len(doc_chunks):
                            continue
                        chunk = doc_chunks[idx]
                        key = (chunk.doc_path, chunk.chunk_id)
                        if key in seen:
                            continue
                        scanned += 1
                        body = clean_retrieval_snippet(chunk.text)
                        if not body:
                            continue
                        seen.add(key)
                        out.append(chunk)
                        readable_chunks += 1
                        if readable_chunks >= min_readable_chunks:
                            break
                    if readable_chunks >= min_readable_chunks or scanned >= max_scan_chunks:
                        break
        return tuple(out)


def _format_chunk_parts(chunks: Iterable[SearchChunk], *, max_tokens: int) -> str:
    remaining = max_tokens
    parts: list[str] = []
    last_heading = ""
    for chunk in chunks:
        title = normalize_text(chunk.heading or Path(chunk.doc_path).stem)
        if re.fullmatch(r"(?:page|slide)\s*\d+", title, flags=re.IGNORECASE):
            title = "内部文档片段"
        heading = "" if title == last_heading else f"资料：{title}\n"
        prefix_tokens = estimate_tokens(heading)
        if remaining <= prefix_tokens + 8:
            break
        body = clean_retrieval_snippet(chunk.text)
        if not body:
            continue
        text = trim_to_token_budget(body, remaining - prefix_tokens)
        parts.append(heading + text)
        remaining -= estimate_tokens(heading + text)
        last_heading = title
    return "\n".join(parts)


def format_hits_with_refs(
    hits: list[SearchHit] | tuple[SearchHit, ...],
    *,
    max_tokens: int = 240,
) -> tuple[str, tuple[SearchHit, ...]]:
    if not hits:
        return "", tuple()
    remaining = max_tokens
    parts: list[str] = []
    shown: list[SearchHit] = []
    seen_text_keys: list[str] = []
    for i, hit in enumerate(hits, 1):
        title = _source_label(hit)
        prefix = f"{i}. 资料：{title}\n"
        prefix_tokens = estimate_tokens(prefix)
        if remaining <= prefix_tokens + 8:
            break
        body = clean_retrieval_snippet(hit.chunk.text)
        if not body:
            continue
        if is_near_duplicate_text(body, seen_text_keys):
            continue
        seen_text_keys.append(duplicate_text_key(body))
        text = trim_to_token_budget(body, remaining - prefix_tokens)
        parts.append(prefix + text)
        shown.append(hit)
        remaining -= estimate_tokens(prefix + text)
    return "\n".join(parts), tuple(shown)


def format_hits(hits: list[SearchHit] | tuple[SearchHit, ...], *, max_tokens: int = 240) -> str:
    return format_hits_with_refs(hits, max_tokens=max_tokens)[0]


def format_read_context(chunks: tuple[SearchChunk, ...], *, max_tokens: int = 320) -> str:
    if not chunks:
        return ""
    return _format_chunk_parts(chunks, max_tokens=max_tokens)


def format_structured_read_context(
    chunks: tuple[SearchChunk, ...],
    *,
    center_chunk_id: int,
    source_id: int,
    neighbor_radius: int = 1,
    max_tokens: int = 320,
) -> str:
    if not chunks:
        return ""
    title = normalize_text(chunks[0].heading or Path(chunks[0].doc_path).stem)
    if re.fullmatch(r"(?:page|slide)\s*\d+", title, flags=re.IGNORECASE):
        title = "内部文档片段"
    parts = [
        f"读取对象：上一轮第 {source_id} 条资料",
        "内容类型：普通文档上下文",
        f"资料：{title}",
    ]
    remaining = max_tokens - estimate_tokens("\n".join(parts))
    per_chunk_cap = max(260, max_tokens // max(len(chunks), 1))
    added = False
    seen_text_keys: list[str] = []
    for chunk in chunks:
        delta = chunk.chunk_id - center_chunk_id
        if remaining <= 8:
            break
        body = clean_retrieval_snippet(chunk.text)
        if not body:
            continue
        if is_near_duplicate_text(body, seen_text_keys):
            continue
        seen_text_keys.append(duplicate_text_key(body))
        cap = per_chunk_cap
        if delta == 0:
            cap = int(per_chunk_cap * 1.8)
        text = trim_to_token_budget(body, min(remaining, cap))
        parts.append(text)
        added = True
        remaining -= estimate_tokens(text)
    if not added:
        return ""
    return "\n\n".join(parts)
