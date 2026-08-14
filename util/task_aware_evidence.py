from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence


PUNCT_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:\n\r\t]+")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


def _iter_hint_texts(sample: Dict, hint_texts: Sequence[str] | None) -> List[str]:
    explanation_target = sample.get("explanation_target") or {}
    hints: List[str] = []

    focus_text = explanation_target.get("focus_text")
    if focus_text:
        hints.append(focus_text)

    triple = explanation_target.get("target_triple") or sample.get("target_triple")
    if isinstance(triple, dict):
        hints.extend([triple.get("subject"), triple.get("predicate"), triple.get("object")])
    elif triple:
        hints.append(triple)

    output_segment = explanation_target.get("output_segment")
    if output_segment:
        hints.append(output_segment)

    task_aware_target_tokens = sample.get("task_aware_target_tokens")
    if isinstance(task_aware_target_tokens, str) and 1 < len(task_aware_target_tokens) <= 24:
        hints.append(task_aware_target_tokens)

    if hint_texts:
        hints.extend(hint_texts)

    expanded: List[str] = []
    for hint in hints:
        text = _clean_text(hint)
        if not text:
            continue
        expanded.append(text)
        expanded.extend(seg for seg in PUNCT_SPLIT_RE.split(text) if 1 < len(seg.strip()) <= 24)
    return _unique_keep_order(expanded)


def _split_prompt_candidates(prompt: str) -> List[str]:
    return _unique_keep_order(
        seg for seg in PUNCT_SPLIT_RE.split(prompt) if 1 < len(_clean_text(seg)) <= 48
    )


def _build_local_window(prompt: str, start: int, end: int, window_size: int) -> str:
    extra = max(int(window_size), 0) // 2
    left = max(0, start - extra)
    right = min(len(prompt), end + extra)
    return _clean_text(prompt[left:right])


def _focus_candidates(sample: Dict, prompt: str, window_size: int) -> List[tuple[str, float]]:
    explanation_target = sample.get("explanation_target") or {}
    candidates: List[tuple[str, float]] = []

    focus_text = _clean_text(explanation_target.get("focus_text"))
    if focus_text:
        start = prompt.find(focus_text)
        if start >= 0:
            local = _build_local_window(prompt, start, start + len(focus_text), window_size)
            if local:
                candidates.append((local, 2.0))

    positions = explanation_target.get("positions") or []
    for raw_pos in positions:
        try:
            pos = int(raw_pos)
        except (TypeError, ValueError):
            continue
        left = max(0, pos - max(int(window_size), 0))
        right = min(len(prompt), pos + max(int(window_size), 0) + 1)
        local = _clean_text(prompt[left:right])
        if local:
            candidates.append((local, 1.0))

    return candidates


def _score_candidate(text: str, hint_texts: Sequence[str], hint_scores: Sequence[float] | None) -> float:
    if not text:
        return 0.0

    total = 0.0
    for idx, hint in enumerate(hint_texts):
        clean_hint = _clean_text(hint)
        if not clean_hint:
            continue
        weight = 1.0
        if hint_scores and idx < len(hint_scores):
            try:
                weight = float(hint_scores[idx])
            except (TypeError, ValueError):
                weight = 1.0
        if clean_hint in text:
            total += weight
            continue

        hint_chars = set(clean_hint)
        if not hint_chars:
            continue
        overlap = sum(1 for ch in hint_chars if ch in text) / len(hint_chars)
        if overlap >= 0.6:
            total += weight * overlap * 0.5
    return total


def build_task_aware_evidence_spans(
    *,
    sample: Dict,
    prompt: str,
    hint_texts: Sequence[str] | None = None,
    hint_scores: Sequence[float] | None = None,
    top_k: int = 3,
    window_size: int = 8,
) -> List[Dict[str, float | str]]:
    prompt = _clean_text(prompt)
    if not prompt:
        return []

    ranked: Dict[str, float] = {}
    all_hints = _iter_hint_texts(sample, hint_texts)

    for candidate in _split_prompt_candidates(prompt):
        score = _score_candidate(candidate, all_hints, hint_scores)
        if score > 0.0:
            ranked[candidate] = max(ranked.get(candidate, 0.0), score)

    for candidate, bonus in _focus_candidates(sample, prompt, window_size):
        score = bonus + _score_candidate(candidate, all_hints, hint_scores)
        ranked[candidate] = max(ranked.get(candidate, 0.0), score)

    if not ranked:
        fallback = _split_prompt_candidates(prompt)[: max(int(top_k), 1)]
        return [{"text": text, "score": 0.0} for text in fallback]

    ordered = sorted(
        ({"text": text, "score": round(score, 6)} for text, score in ranked.items()),
        key=lambda item: (-float(item["score"]), len(str(item["text"])), prompt.find(str(item["text"]))),
    )
    return ordered[: max(int(top_k), 0)]


__all__ = ["build_task_aware_evidence_spans"]
