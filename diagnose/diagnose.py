import os
import sys
from pathlib import Path
base_dir = str(Path(__file__).absolute().parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)
import threading
import methods
from methods import result_template
from util.generate import get_model_output_
from util.sample_adapter import adapt_sample_for_method

def get_model_output(sample, model_name_or_path, method=None, hparams=None):
    try:
        model_output = sample["task_aware_target_tokens"]
    except:
        model_output = ""
    return model_output

def diagnosing(sample, model_name_or_path, method, hparams=None):
    if method not in methods.method_name2diagnose_fun:
        raise KeyError(
            f"Unsupported method: {method}. Loaded methods: {list(methods.method_name2diagnose_fun.keys())}"
        )

    adapted_sample = adapt_sample_for_method(sample=sample, method_name=method)
    diagnose_proxy = methods.method_name2diagnose_fun[method]
    result = diagnose_proxy(sample=adapted_sample, model_name_or_path=model_name_or_path, hparams=hparams)
    return result_template.normalize_method_result(result, adapted_sample, method)


if __name__ == "__main__":
    pass
