from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List

import torch

from diagnose.diagnose import diagnosing
from util.runtime import get_cached_model_tok_runtime
from util.sample_adapter import adapt_sample_for_method


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ranked_evidence_spans(result: Dict, top_k: int) -> List[Dict[str, Any]]:
    spans = result.get("evidence_spans", []) or []
    normalized = []
    seen = set()
    for idx, span in enumerate(spans):
        text = _clean_text(span.get("text") or span.get("span"))
        if not text or text in seen:
            continue
        try:
            score = float(span.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        normalized.append({"text": text, "score": score, "index": idx})
        seen.add(text)
    normalized.sort(key=lambda item: (-item["score"], item["index"]))
    return normalized[:top_k]


def select_top_evidence_texts(result: Dict, top_k: int = 3) -> List[str]:
    return [span["text"] for span in _ranked_evidence_spans(result, top_k)]


def erase_spans_from_prompt(prompt: str, evidence_texts: Iterable[str]) -> str:
    updated = _clean_text(prompt)
    for text in evidence_texts:
        clean = _clean_text(text)
        if clean:
            updated = updated.replace(clean, "")
    return updated


def compress_prompt_by_spans(prompt: str, evidence_texts: Iterable[str]) -> str:
    prompt = _clean_text(prompt)
    indexed = []
    seen = set()
    for text in evidence_texts:
        clean = _clean_text(text)
        if not clean or clean in seen:
            continue
        idx = prompt.find(clean)
        if idx >= 0:
            indexed.append((idx, clean))
            seen.add(clean)

    indexed.sort(key=lambda item: item[0])
    compressed = "".join(text for _, text in indexed)
    return compressed or prompt


def _jaccard_overlap(left_items: Iterable, right_items: Iterable) -> float:
    left = set(left_items or [])
    right = set(right_items or [])
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def extract_task_anchor_text(sample: Dict, window_size: int = 8) -> str:
    explanation_target = sample.get("explanation_target") or {}
    evaluation_anchor = sample.get("evaluation_anchor") or {}

    triple = explanation_target.get("target_triple") or sample.get("target_triple")
    if isinstance(triple, dict):
        return "".join(str(triple.get(k, "")) for k in ("subject", "predicate", "object"))
    if triple:
        return _clean_text(triple)

    focus_text = evaluation_anchor.get("focus_text") or explanation_target.get("focus_text")
    if focus_text:
        return _clean_text(focus_text)

    source_text = _clean_text(evaluation_anchor.get("source_text") or sample.get("prompt") or "")
    positions = evaluation_anchor.get("positions") or explanation_target.get("positions") or []
    if source_text and positions:
        anchors = []
        for pos in positions:
            try:
                idx = int(pos)
            except (TypeError, ValueError):
                continue
            left = max(0, idx - window_size)
            right = min(len(source_text), idx + window_size)
            anchors.append(source_text[left:right])
        if anchors:
            return "".join(anchors)

    return _clean_text(sample.get("task_aware_target_tokens_full") or sample.get("task_aware_target_tokens") or source_text)


def compute_tta_at_k(sample: Dict, evidence_texts: Iterable[str], top_k: int = 3) -> float:
    anchor_text = extract_task_anchor_text(sample)
    anchor_chars = list(anchor_text)
    if not anchor_chars:
        return 0.0

    best = 0.0
    for text in list(evidence_texts)[:top_k]:
        evidence = _clean_text(text)
        if not evidence:
            continue
        overlap = sum(1 for ch in anchor_chars if ch in evidence)
        best = max(best, overlap / len(anchor_chars))
    return round(best, 6)


def score_target_probability(prompt: str, target_text: str, *, model_name: str) -> float:
    target_text = _clean_text(target_text)
    prompt = _clean_text(prompt)
    if not target_text:
        return 0.0

    mt = get_cached_model_tok_runtime(model_name_or_path=model_name)
    tokenizer = mt.tokenizer
    model = mt.model

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        return 0.0

    full_text = prompt + target_text
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    avg_token_logprob = -float(outputs.loss.item())
    return round(math.exp(avg_token_logprob), 6)


def _score_with_removed_spans(
    prompt: str,
    removed_spans: List[str],
    scorer_fn: Callable[..., float],
    target_text: str,
    model_name: str,
) -> float:
    ablated_prompt = erase_spans_from_prompt(prompt, removed_spans)
    return float(scorer_fn(prompt=ablated_prompt, target_text=target_text, model_name=model_name))


def _compute_sequential_aopc(
    prompt: str,
    target_text: str,
    scorer_fn: Callable[..., float],
    model_name: str,
    order_texts: List[str],
    original_score: float,
) -> float:
    if not order_texts:
        return 0.0
    drops = []
    removed = []
    for text in order_texts:
        removed.append(text)
        score = _score_with_removed_spans(prompt, removed, scorer_fn, target_text, model_name)
        drops.append(original_score - score)
    return sum(drops) / len(drops)


def _approximate_greedy_aopc_bound(
    prompt: str,
    target_text: str,
    scorer_fn: Callable[..., float],
    model_name: str,
    ranked_spans: List[Dict[str, Any]],
    original_score: float,
    mode: str,
) -> float:
    remaining = [dict(span) for span in ranked_spans]
    removed = []
    drops = []
    while remaining:
        best_index = None
        best_drop = None
        for idx, span in enumerate(remaining):
            candidate_removed = removed + [span["text"]]
            score = _score_with_removed_spans(prompt, candidate_removed, scorer_fn, target_text, model_name)
            drop = original_score - score
            if best_index is None:
                best_index = idx
                best_drop = drop
                continue
            if mode == "upper" and drop > best_drop:
                best_index = idx
                best_drop = drop
            elif mode == "lower" and drop < best_drop:
                best_index = idx
                best_drop = drop
        if best_index is None:
            break
        chosen = remaining.pop(best_index)
        removed.append(chosen["text"])
        resolved_drop = best_drop if best_drop is not None else 0.0
        drops.append(resolved_drop)
    if not drops:
        return 0.0
    return sum(drops) / len(drops)


def _normalize_naopc(raw: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    value = (raw - lower) / (upper - lower)
    value = max(0.0, min(value, 1.0))
    return round(value, 6)


def evaluate_explanation_sample(
    sample: Dict,
    model_name: str,
    method_name: str,
    scorer_fn: Callable[..., float] | None = None,
    diagnose_fn: Callable[..., Dict] = diagnosing,
    top_k: int = 3,
    variant_limit: int = 3,
):
    adapted = adapt_sample_for_method(sample=sample, method_name=method_name)
    prompt = _clean_text(adapted.get("prompt"))
    target_text = _clean_text(adapted.get("task_aware_target_tokens_full") or adapted.get("task_aware_target_tokens"))
    scorer = scorer_fn or score_target_probability

    base_result = diagnose_fn(sample=deepcopy(adapted), model_name_or_path=model_name, method=method_name)
    ranked_evidence = _ranked_evidence_spans(base_result, top_k=top_k)
    evidence_texts = [span["text"] for span in ranked_evidence]

    original_score = float(scorer(prompt=prompt, target_text=target_text, model_name=model_name))
    ablated_prompt = erase_spans_from_prompt(prompt, evidence_texts)
    compressed_prompt = compress_prompt_by_spans(prompt, evidence_texts)

    ablated_score = float(scorer(prompt=ablated_prompt, target_text=target_text, model_name=model_name))
    kept_score = float(scorer(prompt=compressed_prompt, target_text=target_text, model_name=model_name))

    raw_aopc = _compute_sequential_aopc(
        prompt,
        target_text,
        scorer,
        model_name,
        evidence_texts,
        original_score,
    )

    if ranked_evidence:
        lower_aopc = _approximate_greedy_aopc_bound(
            prompt=prompt,
            target_text=target_text,
            scorer_fn=scorer,
            model_name=model_name,
            ranked_spans=ranked_evidence,
            original_score=original_score,
            mode="lower",
        )
        upper_aopc = _approximate_greedy_aopc_bound(
            prompt=prompt,
            target_text=target_text,
            scorer_fn=scorer,
            model_name=model_name,
            ranked_spans=ranked_evidence,
            original_score=original_score,
            mode="upper",
        )
    else:
        lower_aopc = 0.0
        upper_aopc = 0.0

    naopc_score = _normalize_naopc(raw_aopc, lower_aopc, upper_aopc)

    variants = (adapted.get("prompts") or [prompt])[: max(1, variant_limit)]
    base_spans = set(evidence_texts)
    span_overlaps = []
    for variant_prompt in variants[1:]:
        variant_sample = deepcopy(adapted)
        variant_sample["prompt"] = variant_prompt
        variant_result = diagnose_fn(
            sample=variant_sample,
            model_name_or_path=model_name,
            method=method_name,
        )
        variant_spans = set(select_top_evidence_texts(variant_result, top_k=top_k))
        span_overlaps.append(_jaccard_overlap(base_spans, variant_spans))
    tta_at_k = compute_tta_at_k(adapted, evidence_texts, top_k=top_k)

    return {
        "faithfulness": {
            "naopc": naopc_score,
        },
        "stability": {
            "span_iou": round(sum(span_overlaps) / len(span_overlaps), 6) if span_overlaps else 1.0,
        },
        "target_alignment": {
            "tta_at_3": tta_at_k,
        },
        "evidence": {
            "top_evidence_texts": evidence_texts,
            "compressed_prompt": compressed_prompt,
            "ablated_prompt": ablated_prompt,
        },
        "diagnostics": {
            "original_score": round(original_score, 6),
            "ablated_score": round(ablated_score, 6),
            "kept_score": round(kept_score, 6),
            "raw_aopc": round(raw_aopc, 6),
            "aopc_lower": round(lower_aopc, 6),
            "aopc_upper": round(upper_aopc, 6),
        },
    }
