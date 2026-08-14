from __future__ import annotations

from typing import Dict, Optional

from util.task_schema import extract_triple_from_text, infer_task_schema


def build_explanation_target(sample: Dict, task_schema: Optional[Dict] = None) -> Dict:
    schema = task_schema or infer_task_schema(sample)
    target_type = schema.get("target_type", "text")

    explanation = {
        "target_type": target_type,
        "task_family": schema.get("task_family"),
    }

    if target_type == "target_triple":
        triple = sample.get("target_triple") or schema.get("target_triple")
        if not triple:
            triple = extract_triple_from_text(sample.get("task_aware_target_tokens", ""))
        if triple:
            explanation["target_triple"] = triple

    if target_type == "missing_position":
        focus_text = sample.get("focus_text") or schema.get("focus_text")
        if focus_text:
            explanation["focus_text"] = focus_text
        positions = sample.get("missing_positions") or schema.get("positions")
        if positions:
            explanation["positions"] = positions

    if target_type == "decision_position":
        explanation["focus_text"] = sample.get("task_aware_target_tokens") or schema.get("focus_text") or sample.get("prompt")

    return explanation


def build_evaluation_anchor(sample: Dict, explanation_target: Optional[Dict] = None) -> Dict:
    target = explanation_target or build_explanation_target(sample)
    anchor = {
        "target_type": target.get("target_type", "text"),
    }

    focus_text = target.get("focus_text")
    if focus_text:
        anchor["focus_text"] = focus_text

    source_text = sample.get("source_text") or sample.get("context") or sample.get("prompt")
    if source_text:
        anchor["source_text"] = source_text

    positions = target.get("positions")
    if positions:
        anchor["positions"] = positions

    return anchor
