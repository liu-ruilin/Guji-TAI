import re
from copy import deepcopy
from typing import Dict, Iterable, List

from util.explanation_target import build_evaluation_anchor, build_explanation_target
from util.task_schema import infer_task_schema


PUNCT_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:\n\r\t]+")
SPAN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}")


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    ordered = []
    seen = set()
    for item in items:
        x = _clean_text(item)
        if not x or x in seen:
            continue
        ordered.append(x)
        seen.add(x)
    return ordered


def _first_non_empty(sample: Dict, keys: List[str]) -> str:
    for key in keys:
        value = _clean_text(sample.get(key, ""))
        if value:
            return value
    return ""


def _build_prompt(sample: Dict) -> str:
    prompt = _clean_text(sample.get("prompt", ""))
    if prompt:
        return prompt

    instruction = _clean_text(sample.get("instruction", ""))
    input_text = _clean_text(sample.get("input", ""))
    question = _clean_text(sample.get("question", ""))
    context = _clean_text(sample.get("context", ""))

    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    if instruction:
        return instruction
    if question and context:
        return f"内容：{context}\n问题：{question}\n回答："
    if question:
        return f"问题：{question}\n回答："
    return input_text or context


def _build_prompt_variants(sample: Dict, prompt: str) -> List[str]:
    """Vary only the framing of known fields; retain all task/source content.

    Opaque or custom prompts cannot safely be reconstructed from partial fields.
    Callers can supply explicit ``prompts`` for those samples.
    """
    variants = [prompt]
    fields = dict(sample)
    fields.pop("prompt", None)
    if prompt != _build_prompt(fields):
        return _unique_keep_order(variants)

    instruction = _clean_text(sample.get("instruction", ""))
    input_text = _clean_text(sample.get("input", ""))
    question = _clean_text(sample.get("question", ""))
    context = _clean_text(sample.get("context", ""))

    if instruction and input_text:
        variants.extend([
            f"{instruction}\n\n{input_text}",
            f"{instruction}\n输入：{input_text}",
        ])
    elif instruction:
        variants.extend([f"{instruction}\n回答：", f"任务：{instruction}\n回答："])
    elif question and context:
        variants.extend([
            f"内容：{context}\n\n问题：{question}\n回答：",
            f"上下文：{context}\n问题：{question}\n回答：",
        ])
    elif question:
        variants.extend([f"问题：{question}\n\n回答：", f"请回答以下问题：{question}\n回答："])

    return _unique_keep_order(variants)[:3]


def _extract_subject_candidates(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    candidates = []
    for seg in PUNCT_SPLIT_RE.split(text):
        seg = _clean_text(seg)
        if 2 <= len(seg) <= 24:
            candidates.append(seg)
        for match in SPAN_RE.findall(seg):
            if 2 <= len(match) <= 24:
                candidates.append(match)

    return _unique_keep_order(candidates)


def infer_triple_subject(sample: Dict) -> str:
    prompt = _build_prompt(sample)
    existing = _clean_text(sample.get("triple_subject", ""))
    if existing and existing in prompt:
        return existing

    key_order = [
        "subject",
        "entity",
        "keyword",
        "question",
        "source_text",
        "context",
        "prompt",
    ]

    candidates = []
    for key in key_order:
        candidates.extend(_extract_subject_candidates(_clean_text(sample.get(key, ""))))

    for cand in candidates:
        if cand in prompt:
            return cand

    if candidates:
        return candidates[0]

    prompt_candidates = _extract_subject_candidates(prompt)
    if prompt_candidates:
        return prompt_candidates[0]

    return _clean_text(prompt[:8])


def build_focus_task_aware_target_tokens(text: str, max_chars: int = 32) -> str:
    text = _clean_text(text)
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    clauses = [x.strip() for x in PUNCT_SPLIT_RE.split(text) if x.strip()]
    if clauses:
        first = clauses[0]
        if 2 <= len(first) <= max_chars:
            return first
        joined = "".join(clauses[:2])
        if joined:
            return joined[:max_chars]

    return text[:max_chars]


def adapt_sample_for_method(sample: Dict, method_name: str) -> Dict:
    adapted = deepcopy(sample) if isinstance(sample, dict) else {}

    prompt = _build_prompt(adapted)
    task_aware_target_tokens = _first_non_empty(adapted, ["task_aware_target_tokens", "output", "answer", "target"])

    adapted["prompt"] = prompt
    adapted["task_aware_target_tokens"] = task_aware_target_tokens
    adapted["prompts"] = adapted.get("prompts") or _build_prompt_variants(adapted, prompt)
    adapted["triple_subject"] = infer_triple_subject(adapted)

    # Keep a stable short target for token-level explainability methods.
    if method_name in {"Attribution", "KN", "FiNE"}:
        adapted["task_aware_target_tokens_full"] = task_aware_target_tokens
        adapted["task_aware_target_tokens"] = build_focus_task_aware_target_tokens(task_aware_target_tokens, max_chars=24)

    schema = infer_task_schema(adapted)
    explanation_target = build_explanation_target(adapted, schema)
    evaluation_anchor = build_evaluation_anchor(adapted, explanation_target)

    adapted["task_schema"] = schema
    adapted["explanation_target"] = explanation_target
    adapted["evaluation_anchor"] = evaluation_anchor

    return adapted
