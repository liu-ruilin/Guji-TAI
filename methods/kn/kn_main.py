from copy import deepcopy
from typing import Dict, List, Tuple
import threading
from util.runtime import get_cached_model_tok_runtime
from .knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type
from . import KNHyperParams
from util.fileutil import get_temp_file_with_prefix_suffix
from util.guji_labels import neuron_contribution_meta
from util.plot_style import chinese_plot_context
import numpy as np
from tqdm import tqdm
import torch


def _resolve_neuron_target_text(sample):
    explanation_target = sample.get("explanation_target", {}) or {}
    target_type = explanation_target.get("target_type", "")
    if target_type in {"missing_position", "decision_position", "answer_unit"}:
        focus_text = explanation_target.get("focus_text")
        if focus_text:
            return str(focus_text)
    if target_type == "target_triple":
        triple = explanation_target.get("target_triple") or sample.get("target_triple")
        if isinstance(triple, dict):
            joined = " ".join(str(x) for x in [triple.get("subject"), triple.get("predicate"), triple.get("object")] if x)
            if joined:
                return joined
        if triple:
            return str(triple)
    output_segment = explanation_target.get("output_segment")
    if output_segment:
        return str(output_segment)
    return str(sample.get("task_aware_target_tokens", ""))

def get_refined_neurons(model, tok, sample, hparams, mt):
    runtime_device = getattr(model, "device", None)
    if runtime_device is None:
        try:
            runtime_device = next(model.parameters()).device
        except Exception:
            runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_path),
        device=runtime_device,
    )
    refined_neurons = kn.get_refined_neurons(
        prompts=sample["prompts"],
        task_aware_target_tokens=sample["task_aware_target_tokens"],
        p=hparams.p,
        batch_size=hparams.batch_size,
        steps=hparams.steps,
        coarse_adaptive_threshold=hparams.adaptive_threshold,
        refine=hparams.refine,
    )
    scores = kn.score

    neuron_dic = {}
    un_neuron_dic = {}
    top_token_list, top_score_list = [], []
    embedding_dim = kn._get_word_embeddings().T # input_dim * vocab_size
    try:
        un_emb_dim = model.get_output_embeddings().weight.T
    except AttributeError:
    # ChatGLM2
        un_emb_dim = model.transformer.output_layer.weight.T
    _, neuron_flat = torch.topk(scores.flatten(), k=5)
    neuron_set = [(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in neuron_flat]
    # output = False
    for neuron in tqdm(neuron_set, desc="Getting top tokens for neurons"):
        if mt.model_type == "chatglm2":
            mlp = kn._get_output_ff_layer(layer_idx=(neuron[0]))
        else:
            mlp = kn._get_input_ff_layer(layer_idx=(neuron[0])) # output_dim * input_dim
        neur = mlp[neuron[1], :]
        with torch.no_grad():
            smilarity = neur.to(embedding_dim.device) @ embedding_dim
            try:
                un_sim = neur.to(un_emb_dim.device) @ un_emb_dim
                output = True
            except:
                output = False
        _, index = torch.topk(smilarity, k=3) 
        token_id = [[i] for i in index.cpu().tolist()]
        token = tok.batch_decode(token_id, skip_special_tokens=False)
        name = f"L{neuron[0]}.U{neuron[1]}"
        neuron_dic[name] = token
        top_score_list.append(scores[neuron].item())
        if output:
            _, un_index = torch.topk(un_sim, k=3)
            un_token_id = [[i] for i in un_index.cpu().tolist()]
            un_token = tok.batch_decode(un_token_id, skip_special_tokens=False)
            un_neuron_dic[name] = un_token
            top_token_list.append(un_token[0])
        else:
            un_neuron_dic = None
    
    return np.array(scores.cpu()), neuron_dic, refined_neurons, un_neuron_dic, [t.replace("<s>", "bos") for t in top_token_list], top_score_list

lock_kn = threading.Lock()
def diagnose(sample, model_name_or_path, hparams=None):
    result = dict()
    result["output"] = sample["task_aware_target_tokens"]
    result["image"] = []
    result["table"] = []
    with lock_kn:
        # method prepare
        hparams = KNHyperParams.from_model_name_or_path(model_name_or_path) if hparams is None else hparams
        mt = get_cached_model_tok_runtime(
            model_name_or_path=model_name_or_path,
            hparams_model_path=hparams.model_path,
        )
        with torch.set_grad_enabled(True):
            data, neuron_dic, neuron_set, un_neuron_dic, top_token_list, top_score_list = get_refined_neurons(model=mt.model, tok=mt.tokenizer, sample=sample, hparams=hparams, mt=mt)
        result["origin_data"] = {"Contribution of the neuron": data.tolist(), "Neuron index": neuron_set, "imgs": []}
        table = []
        top_tokens = []
        top_neurons = []
        for k, v in neuron_dic.items():
            top_neurons.append(k)
            top_tokens.append(v[0])
            if un_neuron_dic is not None:
                table.append({"Top neurons": k, "Corresponding top tokens": v, "Unemb Corresponding top tokens": un_neuron_dic[k]}) 
            else:
                 table.append({"Top neurons": k, "Corresponding top tokens": v}) 
        # result["result_des"] = f"Through the KN neuron localization method, we have obtained the following top 5 neuron sets, they are: {list(neuron_dic.keys())}.\nTheir corresponding semantic information is: {top_token_list}.\nThe contribution scores are respectively: {top_score_list}."
        result["result_des"] = ""
        tem = [f'{neu}({tok})' for neu, tok in zip(top_neurons, top_tokens)]
        target_text = _resolve_neuron_target_text(sample)
        neuron_meta = neuron_contribution_meta()
        result["table"].append({"table_name": neuron_meta["table_name"], "table_list": table, 
                                "table_des": neuron_meta["table_des"],
                                "table_res": f"The top neurons (and their meanings) are: {', '.join(tem)}."})
        result["key_neurons"] = top_neurons
        result["target_relevance_score"] = max(top_score_list) if top_score_list else 0.0
        result["decoded_neuron_semantics"] = top_tokens
        result["evidence_spans"] = [{"text": target_text, "score": round(float(result["target_relevance_score"]), 6)}] if target_text else []
        tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
        import matplotlib.pyplot as plt
        fn_data = []
        step = 100
        for da in data:
            tem = []
            batch = int(len(da)/step)
            for i in range(batch):
                tem.append(np.max(np.abs(da[i*step: (i+1)*step])))
            fn_data.append(tem)
        import seaborn as sns
        with chinese_plot_context():
            fig = plt.figure()
            cmap = sns.heatmap(data=fn_data, cmap="crest", cbar=False)
            cmap.set_xlabel(neuron_meta["x_label"], fontsize=10)
            cmap.invert_yaxis()
            cmap.set_ylabel(neuron_meta["y_label"], fontsize=20)
            fig = cmap.get_figure()
            fig.savefig(tmp_png_file)
            plt.close(fig)
        result["origin_data"]["imgs"].append({"image_name": neuron_meta["image_name"], "image_path": tmp_png_file})
        result["origin_data"]["target_text"] = target_text
        result["result_des"] = ""
        return result
