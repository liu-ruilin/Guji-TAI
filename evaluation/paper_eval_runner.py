from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from evaluation.guji_paper_eval import _is_observed_metric, evaluate_dataset_paper


EXCLUDED_DATASET_NAMES = {
    "面向大模型的常识类动态知识探测与编辑数据",
}


def filter_guji_dataset_names(dataset_names):
    return [name for name in dataset_names if name not in EXCLUDED_DATASET_NAMES]


def default_guji_dataset_names():
    from data.chain_of_thought.dataset_process import name2dataset_module

    return filter_guji_dataset_names(sorted(name2dataset_module.keys()))


def build_chunk_ranges(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    total = max(int(total), 0)
    chunk_size = max(int(chunk_size), 1)
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def merge_weighted_metric_summaries(weighted_summaries: Iterable[Tuple[int, Dict]]) -> Dict:
    weighted_summaries = [(int(weight), summary) for weight, summary in weighted_summaries if weight > 0 and summary]
    if not weighted_summaries:
        return {}

    merged = {}
    methods = dict.fromkeys(method for _, summary in weighted_summaries for method in summary)
    for method_name in methods:
        entries = [(weight, summary[method_name]) for weight, summary in weighted_summaries if method_name in summary]
        merged[method_name] = {}
        sections = dict.fromkeys(section for _, summary in entries for section in summary)
        for section_name in sections:
            merged[method_name][section_name] = {}
            keys = dict.fromkeys(key for _, summary in entries for key in summary.get(section_name, {}))
            for metric_name in keys:
                numerator = 0.0
                denominator = 0
                for weight, summary in entries:
                    value = summary.get(section_name, {}).get(metric_name)
                    if not _is_observed_metric(value):
                        continue
                    if section_name == "diagnostics":
                        # Coverage fields are counts, not per-sample averages.
                        numerator += value
                        continue
                    if section_name == "stability" and metric_name == "span_iou":
                        # Old summaries lack coverage; retain their historical
                        # chunk weight, while new summaries use observed samples.
                        weight = summary.get("diagnostics", {}).get("stability_evaluated_samples", weight)
                    numerator += weight * float(value)
                    denominator += weight
                merged[method_name][section_name][metric_name] = (
                    int(numerator) if section_name == "diagnostics"
                    else round(numerator / denominator, 6) if denominator else None
                )
    return merged


def interpret_subprocess_returncode(returncode: int) -> Tuple[str, str | None]:
    if returncode == 0:
        return "ok", None
    if returncode in {-9, 137}:
        return "skipped", "killed"
    return "error", "failed"


def summarize_chunk_entries(chunk_entries: Dict) -> Dict:
    weighted_summaries = []
    completed_num_samples = 0
    skipped_num_samples = 0
    for chunk_info in chunk_entries.values():
        if not isinstance(chunk_info, dict):
            continue
        num_samples = int(chunk_info.get("num_samples", 0) or 0)
        if "summary" in chunk_info:
            weighted_summaries.append((num_samples, chunk_info["summary"]))
            completed_num_samples += num_samples
        elif chunk_info.get("skipped"):
            skipped_num_samples += num_samples
    return {
        "weighted_summaries": weighted_summaries,
        "completed_num_samples": completed_num_samples,
        "skipped_num_samples": skipped_num_samples,
    }


def run_full_paper_eval(
    *,
    model_name: str,
    out_dir: str | Path,
    sample_limit: int = 1,
    dataset_names=None,
    method_names=None,
):
    from data.chain_of_thought.dataset_process import name2dataset_module
    from methods.assist import ALL_SELECTED_METHODS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = list(dataset_names or default_guji_dataset_names())
    method_names = list(method_names or ALL_SELECTED_METHODS)

    payload = {
        "model_name": model_name,
        "sample_limit": sample_limit,
        "dataset_names": dataset_names,
        "method_names": method_names,
        "datasets": {},
    }

    for dataset_name in dataset_names:
        dataset = name2dataset_module[dataset_name].get_default_dataset()
        try:
            summary = evaluate_dataset_paper(
                dataset=dataset,
                model_name=model_name,
                dataset_name=dataset_name,
                sample_limit=sample_limit,
                selected_methods=method_names,
            )
            payload["datasets"][dataset_name] = {
                "methods": method_names,
                "summary": summary,
            }
        except Exception as e:
            payload["datasets"][dataset_name] = {
                "methods": method_names,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }

        out_name = model_name.replace("/", "__").replace("-", "_") + "__paper_eval.json"
        (out_dir / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    return payload
