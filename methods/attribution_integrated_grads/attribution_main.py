from copy import deepcopy
from typing import Dict, List, Tuple
import gc
import threading
from util.model_tokenizer import ModelAndTokenizer
from util.runtime import get_cached_model_tok_runtime
from . import AttributionHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
from util.guji_labels import attribution_heatmap_meta, supporting_spans_text
from util.plot_style import chinese_plot_context
from util.span_evidence import build_ranked_spans
import torch
from tqdm import tqdm
import numpy as np
from util.nethook import Trace
from .hook import Attribution
import pandas as pd

lock_kn = threading.Lock()


def _resolve_target_text(sample: Dict) -> str:
    explanation_target = sample.get("explanation_target", {}) or {}
    target_type = explanation_target.get("target_type", "")

    if target_type in {"decision_position", "missing_position", "answer_unit"}:
        focus_text = explanation_target.get("focus_text")
        if focus_text:
            return str(focus_text)

    if target_type == "target_triple":
        triple = explanation_target.get("target_triple") or sample.get("target_triple")
        if isinstance(triple, dict):
            ordered = [triple.get("subject"), triple.get("predicate"), triple.get("object")]
            joined = " ".join(str(x) for x in ordered if x)
            if joined:
                return joined
        if triple:
            return str(triple)

    output_segment = explanation_target.get("output_segment")
    if output_segment:
        return str(output_segment)

    return str(sample.get("task_aware_target_tokens", ""))


def _aggregate_token_scores_to_spans(prompt_tokens, score_matrix, top_k: int = 3):
    if not prompt_tokens:
        return []

    scores = np.array(score_matrix, dtype=float)
    if scores.size == 0:
        return []
    if scores.ndim == 1:
        token_scores = scores
    else:
        token_scores = scores.sum(axis=0)
    return build_ranked_spans(prompt_tokens, token_scores.tolist(), top_k=top_k)


def _normalize_score_matrix(score_matrix):
    scores = np.array(score_matrix, dtype=float)
    if scores.size == 0:
        return scores

    score_min = np.nanmin(scores)
    score_max = np.nanmax(scores)
    if np.isnan(score_min) or np.isnan(score_max):
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    denom = score_max - score_min
    if denom == 0:
        return np.zeros_like(scores, dtype=float)

    normalized = (scores - score_min) / denom
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _top_score_indices(score_matrix, top_k: int = 3) -> List[Tuple[int, int]]:
    scores = np.array(score_matrix, dtype=float)
    if scores.size == 0:
        return []
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    flat_scores = scores.reshape(-1)
    k = min(max(int(top_k), 0), flat_scores.size)
    if k == 0:
        return []

    ranked_flat_indices = np.argsort(flat_scores)[-k:][::-1]
    unraveled = np.unravel_index(ranked_flat_indices, scores.shape)
    return list(zip(*unraveled))


def _candidate_batch_sizes(initial_batch_size: int, steps: int) -> List[int]:
    upper = max(1, min(int(initial_batch_size), int(steps)))
    candidates = [batch for batch in range(upper, 0, -1) if steps % batch == 0]
    return candidates or [1]


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _run_attribution_with_oom_retry(att, *, prompt: str, task_aware_target_tokens: str, batch_size: int, steps: int):
    last_error = None
    for effective_batch_size in _candidate_batch_sizes(batch_size, steps):
        try:
            with torch.set_grad_enabled(True):
                return att.get_attribution_scores(
                    prompt=prompt,
                    task_aware_target_tokens=task_aware_target_tokens,
                    batch_size=effective_batch_size,
                    steps=steps,
                )
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            last_error = exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if last_error is not None:
        raise last_error
    raise RuntimeError("Attribution retry exited without result or captured error.")

def get_attributes(x: torch.nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


        


def diagnose(sample, model_name_or_path, hparams=None):
    """
    return: dic: {"output": task_aware_target_tokens of data, "image": image save path, "neuron_dic": {"neuron_name": [list of strings]}}
    """
    result = dict()
    result["output"] = sample["task_aware_target_tokens"]
    result["image"] = []
    target_text = _resolve_target_text(sample)
    with lock_kn:
        # method prepare
        hparams = AttributionHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok_runtime(
            model_name_or_path=model_name_or_path,
            hparams_model_path=hparams.model_path,
        )

        model_device = getattr(mt.model, "device", None)
        if model_device is None:
            try:
                model_device = next(mt.model.parameters()).device
            except Exception:
                model_device = "cuda" if torch.cuda.is_available() else "cpu"

        att = Attribution(mt, model_type=mt.model_type, device=model_device)
        score, prompt_list, te = _run_attribution_with_oom_retry(
            att,
            prompt=sample["prompt"],
            task_aware_target_tokens=target_text,
            batch_size=hparams.batch_size,
            steps=hparams.num_steps,
        )
        # len(task_aware_target_tokens_tokens)*len(prompt_tokens)
        tem_score = _normalize_score_matrix(score)
        max_indices = _top_score_indices(tem_score, top_k=3)
        score = tem_score.tolist() 
        prompt_token = mt.tokenizer.batch_decode([[i] for i in prompt_list])   
        task_aware_target_tokens_token = mt.tokenizer.batch_decode([[i] for i in te])
        evidence_spans = _aggregate_token_scores_to_spans(prompt_token, score, top_k=3)
        plot_meta = attribution_heatmap_meta()
        # result["result_des"] = f"The attribution scores for the top 3 input-output pairs are: {[f'{prompt_token[ind[1]]}->{task_aware_target_tokens_token[ind[0]]}' for ind in max_indices]}."

        import seaborn as sns
        import matplotlib.pyplot as plt
        tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
        with chinese_plot_context():
            plt.figure()
            df = pd.DataFrame(np.array(score), index=task_aware_target_tokens_token, columns=prompt_token)
            cmap = sns.heatmap(data=df, annot=False, cmap="Reds")
            cmap.set_title(plot_meta["title"])
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.xlabel(plot_meta["x_label"])
            plt.ylabel(plot_meta["y_label"])
            fig = cmap.get_figure()
            fig.savefig(tmp_png_file, transparent=True)
            plt.close(fig)
        result["image"].append({"image_name": plot_meta["image_name"], "image_path": tmp_png_file, 
                                "image_des": plot_meta["image_des"],
                                "image_res": f"The attribution scores for the top 3 input-output pairs are: \n{', '.join([f'{prompt_token[ind[1]]}->{task_aware_target_tokens_token[ind[0]]}' for ind in max_indices])}."})
        result["origin_data"] = {
            "Attribution": score,
            "prompt_tokens": prompt_token,
            "task_aware_target_tokens_tokens": task_aware_target_tokens_token,
            "target_text": target_text,
        }
        result["evidence_spans"] = evidence_spans
        result["result_des"] = supporting_spans_text(evidence_spans)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
