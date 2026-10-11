from __future__ import annotations

from statistics import mean
from math import isfinite
from numbers import Real

from evaluation.paper_metrics import evaluate_explanation_sample


def _is_observed_metric(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def summarize_dataset_paper_metrics(metric_list):
    if not metric_list:
        return {}

    summary = {}
    for section in ["faithfulness", "stability", "target_alignment"]:
        keys = dict.fromkeys(key for item in metric_list for key in item.get(section, {}))
        summary[section] = {}
        for key in keys:
            values = [item.get(section, {}).get(key) for item in metric_list]
            observed = [value for value in values if _is_observed_metric(value)]
            summary[section][key] = round(mean(observed), 6) if observed else None
    evaluated = sum(
        _is_observed_metric(item.get("stability", {}).get("span_iou"))
        for item in metric_list
    )
    summary["diagnostics"] = {
        "stability_evaluated_samples": evaluated,
        "stability_unevaluated_samples": len(metric_list) - evaluated,
        "total_samples": len(metric_list),
    }
    return summary


def evaluate_dataset_paper(dataset, model_name, dataset_name, sample_limit=3, selected_methods=None):
    if not selected_methods:
        from methods.assist import get_methods_by_dataset_and_model_name

    methods = selected_methods or get_methods_by_dataset_and_model_name(dataset_name, model_name)
    aggregated = {method: [] for method in methods}

    for sample in dataset[:sample_limit]:
        for method in methods:
            aggregated[method].append(
                evaluate_explanation_sample(
                    sample=sample,
                    model_name=model_name,
                    method_name=method,
                )
            )

    return {
        method: summarize_dataset_paper_metrics(metric_list)
        for method, metric_list in aggregated.items()
    }
