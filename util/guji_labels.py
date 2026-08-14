from __future__ import annotations


def supporting_spans_text(spans, prefix: str = "Top supporting context spans") -> str:
    if not spans:
        return ""
    formatted = []
    for item in spans:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        score = float(item.get("score", 0.0))
        formatted.append(f"{text}({score:.3f})")
    if not formatted:
        return ""
    return f"{prefix}: " + ", ".join(formatted)


def attention_head_id(layer: int, head: int) -> str:
    return f"Layer_{layer}-Head_{head}"


def attention_head_title(layer: int, head: int) -> str:
    return f"{attention_head_id(layer, head)} attention weights"


def attribution_heatmap_meta():
    return {
        "title": "Attribution score for output",
        "image_name": "Attribution score for output",
        "x_label": "Prompt tokens",
        "y_label": "Task-aware target tokens",
        "image_des": "The graph above represents the attribution score of the input prompt's tokens on predicting the tokens in the task-aware target tokens.",
    }


def neuron_contribution_meta():
    return {
        "image_name": "Contribution of the neuron",
        "x_label": "Index of neurons",
        "y_label": "Layer",
        "table_name": "Top neuron and relative tokens",
        "table_des": "We decode the semantic information of the neurons ranked in the top 5 using the model's embedding layer and unembedding layer.",
    }


def causal_trace_plot_meta(kind=None, modelname: str = "GPT", window: int = 10):
    if not kind:
        return {
            "title": "Impact of restoring state after corrupted input",
            "x_label": f"single restored layer within {modelname}",
            "image_name": "Restoring state",
            "image_des": "The above images separately indicate the influence of different hidden layer vectors on the model input.",
        }

    kindname = "MLP" if str(kind).lower() == "mlp" else "Attn"
    return {
        "title": f"Impact of restoring {kindname} after corrupted input",
        "x_label": f"center of interval of {window} restored {kindname} layers",
        "image_name": f"Restoring {kindname}",
        "image_des": "The above images separately indicate the influence of different hidden layer vectors on the model input.",
    }
