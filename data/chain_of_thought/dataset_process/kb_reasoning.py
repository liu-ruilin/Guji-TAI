import json
from data.chain_of_thought.dataset_process.dataset_base import Dataset
from data.chain_of_thought.dataset_process.guji_common import normalize_kb_reasoning_sample, resolve_repo_data_path

dataset_info = {
    "name": "知识库推理_0_shot",
    "des": "结构化理解类任务：知识库推理",
    "dataset_type": "结构化理解类"
}

support_template_keys = [
    "prompt",
    "source_text",
    "task_aware_target_tokens",
    "task_type",
    "sub_task",
    "reasoning_mode",
    "domain",
]

class GujiStructDataset(Dataset):
    def __init__(self, loc, domain="traditional_culture"):
        self.loc = str(loc)
        self.domain = domain

        with open(self.loc, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.data = [
            normalize_kb_reasoning_sample(x, domain=self.domain)
            for x in self.raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def samples(self):
        return self.data


def get_processed_kvs(sample, keys=None, required_keys=None):
    item = {
        "prompt": sample.get("prompt", ""),
        "source_text": sample.get("source_text", ""),
        "task_aware_target_tokens": sample.get("task_aware_target_tokens", ""),
        "task_type": sample.get("task_type", "structured_understanding"),
        "sub_task": sample.get("sub_task", "kb_reasoning"),
        "reasoning_mode": sample.get("reasoning_mode", "0_shot"),
        "domain": sample.get("domain", "traditional_culture"),
    }
    selected_keys = keys if keys is not None else required_keys
    if selected_keys is not None:
        return {k: item.get(k, "") for k in selected_keys}
    return item


def get_default_dataset():
    return GujiStructDataset(
        loc=resolve_repo_data_path("0_shot", "knowledge_base_reasoning.json"),
        domain="traditional_culture",
    )
