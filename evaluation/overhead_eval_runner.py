from __future__ import annotations

import gc
import json
import math
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import torch

from data.chain_of_thought.dataset_process import name2dataset_module
from diagnose.diagnose import diagnosing
from methods.assist import ALL_SELECTED_METHODS
from methods import method_name2sub_module
from util.hparams import resolve_hparams_json_path
from util.model_tokenizer import model_name2obj
from util.runtime import get_cached_model_tok_runtime
from util.sample_adapter import adapt_sample_for_method


EXCLUDED_DATASET_NAMES = {
    "面向大模型的常识类动态知识探测与编辑数据",
}


def filter_guji_dataset_names(dataset_names):
    return [name for name in dataset_names if name not in EXCLUDED_DATASET_NAMES]


def default_guji_dataset_names():
    return filter_guji_dataset_names(sorted(name2dataset_module.keys()))


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sync_cuda():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)


def _reset_peak_memory_stats():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)


def _peak_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak_bytes = 0
    for device_idx in range(torch.cuda.device_count()):
        peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated(device_idx)))
    return round(peak_bytes / (1024 ** 3), 6)


def _clear_runtime_caches(model_name: str):
    mt = model_name2obj.get(model_name)
    if mt is not None:
        mt.cache_hiddenstates.clear()
        mt.cache_attentionweights.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_runtime_with_local_path(model_name: str, method_names: List[str]):
    for method_name in method_names:
        method_module = method_name2sub_module.get(method_name)
        if method_module is None:
            continue
        hparams_dir = Path(method_module.__file__).parent / "hparams"
        hparams_json_path = resolve_hparams_json_path(hparams_dir, model_name)
        payload = json.loads(hparams_json_path.read_text(encoding="utf-8"))
        model_path = payload.get("model_path")
        return get_cached_model_tok_runtime(
            model_name_or_path=model_name,
            hparams_model_path=model_path,
        )

    return get_cached_model_tok_runtime(model_name_or_path=model_name)


def _score_target_probability_with_runtime(prompt: str, target_text: str, mt) -> float:
    target_text = _clean_text(target_text)
    prompt = _clean_text(prompt)
    if not target_text:
        return 0.0

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


def _measure_callable(fn):
    _sync_cuda()
    _reset_peak_memory_stats()
    start_time = time.perf_counter()
    result = fn()
    _sync_cuda()
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_gb = _peak_gpu_memory_gb()
    return result, round(elapsed_seconds, 6), peak_memory_gb


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def _summarize_numeric(values: List[float]) -> Dict:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    ordered = sorted(float(v) for v in values)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "mean": round(mean(ordered), 6),
        "median": round(median(ordered), 6),
        "p90": round(ordered[p90_index], 6),
        "max": round(max(ordered), 6),
    }


def _build_markdown_summary(payload: Dict) -> str:
    lines = [
        "# Interpretability Overhead Summary",
        "",
        f"- Model: `{payload['model_name']}`",
        f"- Sample limit per dataset: `{payload['sample_limit']}`",
        f"- Datasets: `{', '.join(payload['dataset_names'])}`",
        f"- Methods: `{', '.join(payload['method_names'])}`",
        "",
        "| Dataset | Method | Baseline s/sample | Explain s/sample | Time Ratio | Baseline GB | Explain GB | Mem Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for dataset_name, dataset_info in payload.get("datasets", {}).items():
        summaries = dataset_info.get("method_summaries", {})
        for method_name, summary in summaries.items():
            lines.append(
                "| {dataset} | {method} | {base_t:.4f} | {exp_t:.4f} | {time_r:.2f} | {base_m:.3f} | {exp_m:.3f} | {mem_r:.2f} |".format(
                    dataset=dataset_name,
                    method=method_name,
                    base_t=summary["baseline_time_seconds"]["mean"],
                    exp_t=summary["explain_time_seconds"]["mean"],
                    time_r=summary["time_overhead_ratio"]["mean"],
                    base_m=summary["baseline_peak_gpu_gb"]["mean"],
                    exp_m=summary["explain_peak_gpu_gb"]["mean"],
                    mem_r=summary["memory_overhead_ratio"]["mean"],
                )
            )

    overall = payload.get("overall_method_summary", {})
    if overall:
        lines.extend(
            [
                "",
                "## Overall Means",
                "",
                "| Method | Baseline s/sample | Explain s/sample | Time Ratio | Baseline GB | Explain GB | Mem Ratio |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method_name, summary in overall.items():
            lines.append(
                "| {method} | {base_t:.4f} | {exp_t:.4f} | {time_r:.2f} | {base_m:.3f} | {exp_m:.3f} | {mem_r:.2f} |".format(
                    method=method_name,
                    base_t=summary["baseline_time_seconds"]["mean"],
                    exp_t=summary["explain_time_seconds"]["mean"],
                    time_r=summary["time_overhead_ratio"]["mean"],
                    base_m=summary["baseline_peak_gpu_gb"]["mean"],
                    exp_m=summary["explain_peak_gpu_gb"]["mean"],
                    mem_r=summary["memory_overhead_ratio"]["mean"],
                )
            )

    lines.append("")
    return "\n".join(lines)


def _aggregate_method_records(records: List[Dict]) -> Dict:
    baseline_times = [item["baseline_time_seconds"] for item in records]
    explain_times = [item["explain_time_seconds"] for item in records]
    baseline_mems = [item["baseline_peak_gpu_gb"] for item in records]
    explain_mems = [item["explain_peak_gpu_gb"] for item in records]
    time_ratios = [item["time_overhead_ratio"] for item in records]
    mem_ratios = [item["memory_overhead_ratio"] for item in records]
    prompt_chars = [item["prompt_chars"] for item in records]
    target_chars = [item["target_chars"] for item in records]

    return {
        "num_samples": len(records),
        "prompt_chars": _summarize_numeric(prompt_chars),
        "target_chars": _summarize_numeric(target_chars),
        "baseline_time_seconds": _summarize_numeric(baseline_times),
        "explain_time_seconds": _summarize_numeric(explain_times),
        "baseline_peak_gpu_gb": _summarize_numeric(baseline_mems),
        "explain_peak_gpu_gb": _summarize_numeric(explain_mems),
        "time_overhead_ratio": _summarize_numeric(time_ratios),
        "memory_overhead_ratio": _summarize_numeric(mem_ratios),
    }


def run_overhead_eval(
    *,
    model_name: str,
    out_dir: str | Path,
    sample_limit: int = 10,
    dataset_names=None,
    method_names=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = list(dataset_names or default_guji_dataset_names())
    method_names = list(method_names or ALL_SELECTED_METHODS)

    mt = _resolve_runtime_with_local_path(model_name=model_name, method_names=method_names)

    payload = {
        "model_name": model_name,
        "sample_limit": sample_limit,
        "dataset_names": dataset_names,
        "method_names": method_names,
        "protocol": {
            "baseline": "single forward target-probability scoring with the shared local runtime",
            "explain": "end-to-end diagnose() call with cold prompt-level caches before each sample",
            "metrics": [
                "baseline_time_seconds",
                "explain_time_seconds",
                "baseline_peak_gpu_gb",
                "explain_peak_gpu_gb",
                "time_overhead_ratio",
                "memory_overhead_ratio",
            ],
        },
        "datasets": {},
        "overall_method_summary": {},
    }

    per_method_all_records = {method_name: [] for method_name in method_names}

    for dataset_name in dataset_names:
        dataset_module = name2dataset_module[dataset_name]
        dataset = dataset_module.get_default_dataset()
        dataset_records = {method_name: [] for method_name in method_names}
        dataset_payload = {
            "num_samples_requested": sample_limit,
            "num_samples_completed": 0,
            "method_summaries": {},
            "errors": [],
        }

        for sample_idx, sample in enumerate(dataset[:sample_limit]):
            for method_name in method_names:
                try:
                    adapted = adapt_sample_for_method(sample=sample, method_name=method_name)
                    prompt = _clean_text(adapted.get("prompt"))
                    target_text = _clean_text(adapted.get("task_aware_target_tokens_full") or adapted.get("task_aware_target_tokens"))

                    _clear_runtime_caches(model_name)
                    _, baseline_time, baseline_peak_gpu_gb = _measure_callable(
                        lambda: _score_target_probability_with_runtime(prompt=prompt, target_text=target_text, mt=mt)
                    )

                    _clear_runtime_caches(model_name)
                    _, explain_time, explain_peak_gpu_gb = _measure_callable(
                        lambda: diagnosing(sample=sample, model_name_or_path=model_name, method=method_name)
                    )

                    record = {
                        "sample_index": sample_idx,
                        "method_name": method_name,
                        "prompt_chars": len(prompt),
                        "target_chars": len(target_text),
                        "baseline_time_seconds": baseline_time,
                        "explain_time_seconds": explain_time,
                        "baseline_peak_gpu_gb": baseline_peak_gpu_gb,
                        "explain_peak_gpu_gb": explain_peak_gpu_gb,
                        "time_overhead_ratio": _safe_ratio(explain_time, baseline_time),
                        "memory_overhead_ratio": _safe_ratio(explain_peak_gpu_gb, baseline_peak_gpu_gb),
                    }
                    dataset_records[method_name].append(record)
                    per_method_all_records[method_name].append(record)
                except Exception as exc:
                    dataset_payload["errors"].append(
                        {
                            "sample_index": sample_idx,
                            "method_name": method_name,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

            completed_counts = [len(records) for records in dataset_records.values()]
            dataset_payload["num_samples_completed"] = min(completed_counts) if completed_counts else 0

        for method_name, records in dataset_records.items():
            dataset_payload["method_summaries"][method_name] = _aggregate_method_records(records)

        payload["datasets"][dataset_name] = dataset_payload
        out_name = model_name.replace("/", "__").replace("-", "_") + "__overhead_eval.json"
        (out_dir / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for method_name, records in per_method_all_records.items():
        payload["overall_method_summary"][method_name] = _aggregate_method_records(records)

    summary_md = _build_markdown_summary(payload)
    (out_dir / "overhead_summary.md").write_text(summary_md, encoding="utf-8")

    out_name = model_name.replace("/", "__").replace("-", "_") + "__overhead_eval.json"
    (out_dir / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
