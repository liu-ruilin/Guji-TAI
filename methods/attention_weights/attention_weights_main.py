from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.runtime import get_cached_model_tok_runtime
from . import AttentionWeightsHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
from util.guji_labels import attention_head_id, attention_head_title, supporting_spans_text
from util.plot_style import chinese_plot_context
from util.span_evidence import build_ranked_spans
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd
import os 


lock_kn = threading.Lock()


def _collect_attentions_with_fallback(mt, model_name_or_path, inputs):
    if "t5" in model_name_or_path.lower():
        decoder_start_token_id = mt.model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(mt.model.device)
        model_output = mt.model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)
        attentions = model_output.encoder_attentions
    else:
        model_output = mt.model(**inputs, output_attentions=True)
        attentions = model_output.attentions

    if attentions is None and hasattr(mt.model, "set_attn_implementation"):
        mt.model.set_attn_implementation("eager")
        if "t5" in model_name_or_path.lower():
            decoder_start_token_id = mt.model.config.decoder_start_token_id
            decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(mt.model.device)
            model_output = mt.model(**inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)
            attentions = model_output.encoder_attentions
        else:
            model_output = mt.model(**inputs, output_attentions=True)
            attentions = model_output.attentions

    return attentions


def _summarize_attention_tokens(token_list, attention_weights, top_k=3):
    if not token_list:
        return []

    att_array = np.array(attention_weights, dtype=float)
    if att_array.size == 0:
        return []

    token_scores = att_array.sum(axis=tuple(range(att_array.ndim - 1)))
    return build_ranked_spans(token_list, token_scores.tolist(), top_k=top_k)


def diagnose(sample, model_name_or_path, hparams=None):
    """
    return: dic: {"output": task_aware_target_tokens of data, "image": image save path, "neuron_dic": {"neuron_name": [list of strings]}}
    """
    result = dict()
    result["output"] = sample["task_aware_target_tokens"]
    result["image"] = []
    result["table"] = []
    tem_img = []
    with lock_kn:
        # method prepare
        hparams = AttentionWeightsHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok_runtime(
            model_name_or_path=model_name_or_path,
            hparams_model_path=hparams.model_path,
        )
        prob_dic_fntoken = dict()
        prompt = sample["prompt"]
        if prompt not in mt.cache_attentionweights:
            with torch.no_grad():
                inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)
                ##qwen
                original_use_cache_quant = None
                if "qwen" in model_name_or_path.lower():
                    if hasattr(mt.model.config, 'use_cache_quantization'):
                        original_use_cache_quant = mt.model.config.use_cache_quantization
                        mt.model.config.use_cache_quantization = True
                attentions = _collect_attentions_with_fallback(mt, model_name_or_path, inputs)
                ##qwen
                if original_use_cache_quant is not None:
                    mt.model.config.use_cache_quantization = original_use_cache_quant

                mt.cache_attentionweights[prompt] = torch.stack(attentions).cpu()

        attention_weights = [] # num_layer * num_head * num_tokens
        tem_att = []
        for att in mt.cache_attentionweights[sample["prompt"]]:
            # num_layer * batch_size * num_head * num_tokens
            att_perlayer = att[0].tolist()
            attention_weights.append(att_perlayer)
            for h in att_perlayer:
                tem_att.append(-np.var(h))
        orig_tokens = [[t] for t in mt.tokenizer.encode(sample["prompt"])]
        token_list = mt.tokenizer.batch_decode(orig_tokens)

        _, ids = torch.topk(torch.tensor(tem_att), k=hparams.num_heads)
        att_ids = ids.cpu().tolist()
        min_var_id = []

        import seaborn as sns
        import matplotlib.pyplot as plt
        for l, atts in enumerate(attention_weights):
            for h, att in enumerate(atts):
                if (l*len(atts)+h) not in att_ids:
                    continue
                tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
                head_label = attention_head_title(l, h)
                with chinese_plot_context():
                    plt.figure()
                    df = pd.DataFrame(att, index=token_list, columns=token_list)
                    cmap = sns.heatmap(data=df, annot=False, cmap="Blues")
                    cmap.set_title(head_label)
                    plt.xticks(fontsize=5)
                    plt.yticks(fontsize=5)
                    fig = cmap.get_figure()
                    fig.savefig(tmp_png_file)
                    plt.close(fig)
                min_var_id.append([l, h])
                tem_img.append({"image_name": head_label, "image_path": tmp_png_file})

        result["origin_data"] = {"tokens": token_list, "attention_weight": attention_weights, "min_var_id": min_var_id, "imgs": tem_img}
        result["key_heads"] = [attention_head_id(layer, head) for layer, head in min_var_id]
        result["evidence_spans"] = _summarize_attention_tokens(token_list, attention_weights, top_k=3)
        result["result_des"] = supporting_spans_text(result["evidence_spans"], prefix="Top attention-focused tokens")
                  
        return result
