"""Lightweight mixed Chinese/English retrieval for the QA-RL agent.

The exam environment may not have extra search packages installed, so this file
uses only the standard library. It indexes markdown/text chunks with substring
matches, English tokens, Chinese character n-grams, and heading/path features.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]*|\d+(?:\.\d+)?")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True)
class SearchChunk:
    doc_path: str
    heading: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: SearchChunk
    score: float
    query: str


def normalize_text(text: str) -> str:
    """Normalize full-width forms and whitespace without changing semantics."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_query(query: str, max_chars: int = 64) -> str:
    query = normalize_text(query)
    query = re.sub(r"[<>`*_#]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip(" \t\r\n,，;；、。.")
    return query[:max_chars].strip()


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
    candidates.extend(t for t in english_terms if len(t) >= 2)

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
    text_n = normalize_text(text)
    heading_n = normalize_text(heading)
    path_n = normalize_text(Path(path).stem)

    tokens: list[str] = []
    tokens.extend(t.casefold() for t in EN_TOKEN_RE.findall(text_n))
    tokens.extend(cjk_ngrams(text_n))

    # Heading/path features are duplicated to provide a mild prior.
    heading_tokens = [t.casefold() for t in EN_TOKEN_RE.findall(heading_n)] + cjk_ngrams(heading_n)
    path_tokens = [t.casefold() for t in EN_TOKEN_RE.findall(path_n)] + cjk_ngrams(path_n)
    tokens.extend(heading_tokens * 2)
    tokens.extend(path_tokens)
    return [t for t in tokens if t]


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


def chunk_markdown(path: Path, *, max_chars: int = 900) -> list[SearchChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path)
    heading = ""
    chunks: list[SearchChunk] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        raw = "\n".join(buf).strip()
        buf.clear()
        if not raw:
            return
        for part in _split_long_paragraph(raw, max_chars):
            chunks.append(SearchChunk(rel, heading, len(chunks), part))

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            heading = normalize_text(m.group(1))
            continue
        if not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return chunks


def load_doc_chunks(docs_dir: str | Path, *, max_chars: int = 900) -> list[SearchChunk]:
    root = Path(docs_dir)
    if not root.exists():
        return []
    chunks: list[SearchChunk] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES):
        try:
            chunks.extend(chunk_markdown(path, max_chars=max_chars))
        except OSError:
            continue
    return chunks


class SimpleSearchEngine:
    def __init__(self, docs_dir: str | Path | None = None, *, chunks: list[SearchChunk] | None = None):
        self.chunks = chunks if chunks is not None else load_doc_chunks(docs_dir or "")
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._postings: dict[str, set[int]] = {}
        self._cache: dict[tuple[str, int], tuple[SearchHit, ...]] = {}
        self._build()

    def _build(self) -> None:
        for chunk in self.chunks:
            tf = Counter(feature_tokens(chunk.text, heading=chunk.heading, path=chunk.doc_path))
            self._tf.append(tf)
            self._lengths.append(sum(tf.values()) or 1)
            self._df.update(tf.keys())
            idx = len(self._tf) - 1
            for term in tf:
                self._postings.setdefault(term, set()).add(idx)
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)

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

    def search(self, query: str, top_k: int = 4) -> tuple[SearchHit, ...]:
        query = clean_query(query, max_chars=96)
        cache_key = (query.casefold(), top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not query or not self.chunks:
            return tuple()
        q_norm = normalize_text(query).casefold()
        q_features = feature_tokens(query)
        english_terms = {t.casefold() for t in EN_TOKEN_RE.findall(query)}
        candidate_indices: set[int] = set()
        for term in q_features:
            candidate_indices.update(self._postings.get(term, ()))
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
            if score > 0:
                scored.append(SearchHit(chunk, score, query))

        scored.sort(key=lambda h: h.score, reverse=True)
        diverse: list[SearchHit] = []
        seen_docs: Counter[str] = Counter()
        seen_text: set[str] = set()
        for hit in scored:
            text_key = normalize_text(hit.chunk.text[:160]).casefold()
            if text_key in seen_text:
                continue
            penalty = 0.3 * seen_docs[hit.chunk.doc_path]
            adjusted = SearchHit(hit.chunk, hit.score - penalty, hit.query)
            diverse.append(adjusted)
            seen_docs[hit.chunk.doc_path] += 1
            seen_text.add(text_key)
            if len(diverse) >= max(top_k * 3, top_k):
                break
        diverse.sort(key=lambda h: h.score, reverse=True)
        result = tuple(diverse[:top_k])
        if len(self._cache) >= 2048:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result


def format_hits(hits: list[SearchHit] | tuple[SearchHit, ...], *, max_tokens: int = 240) -> str:
    if not hits:
        return ""
    remaining = max_tokens
    parts: list[str] = []
    for i, hit in enumerate(hits, 1):
        title = hit.chunk.heading or Path(hit.chunk.doc_path).stem
        prefix = f"{i}. {title}: "
        prefix_tokens = estimate_tokens(prefix)
        if remaining <= prefix_tokens + 8:
            break
        text = trim_to_token_budget(normalize_text(hit.chunk.text), remaining - prefix_tokens)
        parts.append(prefix + text)
        remaining -= estimate_tokens(prefix + text)
    return "\n".join(parts)
