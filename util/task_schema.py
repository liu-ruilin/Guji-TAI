from __future__ import annotations

import re
from typing import Dict, Iterable, List

PUNCTUATION_KEYWORDS = [
    "标点",
    "断句",
    "加标点",
    "加标点符号",
    "停顿",
    "逗号",
    "句号",
    "分句",
    "punct",
]

RESTORATION_KEYWORDS = [
    "补全",
    "缺",
    "修复",
    "恢复",
    "还原",
    "restore",
    "missing",
]

RELATION_KEYWORDS = [
    "关系",
    "三元",
    "triple",
    "scheme",
    "抽取",
]

TRIPLE_RE = re.compile(r"\(([^)]+?)\)")


def _collect_text(sample: Dict) -> str:
    fields = [
        sample.get("instruction", ""),
        sample.get("question", ""),
        sample.get("prompt", ""),
        sample.get("context", ""),
        sample.get("source_text", ""),
        sample.get("task_aware_target_tokens", ""),
    ]
    return " ".join(str(x) for x in fields if x).lower()


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    text = text.lower()
    return any(keyword.lower() in text for keyword in keywords)


def infer_task_schema(sample: Dict) -> Dict:
    schema: Dict[str, object] = {
        "target_type": "text",
        "task_family": "generic",
    }

    text_blob = _collect_text(sample)

    triple = sample.get("target_triple") or sample.get("target")
    if triple:
        schema["target_type"] = "target_triple"
        schema["task_family"] = "relation_extraction"
        schema["target_triple"] = triple
        return schema

    task_aware_target_tokens = sample.get("task_aware_target_tokens")
    if isinstance(task_aware_target_tokens, dict) and task_aware_target_tokens.get("target_triple"):
        schema["target_type"] = "target_triple"
        schema["task_family"] = "relation_extraction"
        schema["target_triple"] = task_aware_target_tokens.get("target_triple")
        return schema

    missing_positions = sample.get("missing_positions")
    focus_text = sample.get("focus_text")
    if missing_positions or focus_text or _contains_keyword(text_blob, RESTORATION_KEYWORDS):
        schema["target_type"] = "missing_position"
        schema["task_family"] = "restoration"
        if focus_text:
            schema["focus_text"] = focus_text
        if missing_positions:
            schema["positions"] = missing_positions
        return schema

    if _contains_keyword(text_blob, PUNCTUATION_KEYWORDS):
        schema["target_type"] = "decision_position"
        schema["task_family"] = "punctuation"
        return schema

    if _contains_keyword(text_blob, RELATION_KEYWORDS) and (not missing_positions):
        schema["target_type"] = "target_triple"
        schema["task_family"] = "relation_extraction"
        return schema

    return schema


def extract_triple_from_text(text: str) -> Dict | None:
    if not text:
        return None
    match = TRIPLE_RE.search(text)
    if not match:
        return None

    segments = [seg.strip() for seg in match.group(1).split(",") if seg.strip()]
    if len(segments) < 3:
        return None

    return {"subject": segments[0], "predicate": segments[1], "object": segments[2]}
