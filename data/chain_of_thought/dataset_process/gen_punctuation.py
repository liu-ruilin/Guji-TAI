import json
from data.chain_of_thought.dataset_process.dataset_base import Dataset
from data.chain_of_thought.dataset_process.guji_common import normalize_generation_sample, resolve_repo_data_path

dataset_info = {
    "name": "自动标点_chain_of_thought",
    "des": "生成类任务：古文自动标点",
    "dataset_type": "生成类"
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

class GujiGenerationDataset(Dataset):
    def __init__(self, loc, sub_task, reasoning_mode="chain_of_thought", domain="generic_guji"):
        self.loc = str(loc)
        self.sub_task = sub_task
        self.reasoning_mode = reasoning_mode
        self.domain = domain

        with open(self.loc, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.data = [
            normalize_generation_sample(
                x,
                sub_task=self.sub_task,
                reasoning_mode=self.reasoning_mode,
                domain=self.domain,
            )
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
        "task_type": sample.get("task_type", "generation"),
        "sub_task": sample.get("sub_task", "punctuation"),
        "reasoning_mode": sample.get("reasoning_mode", "chain_of_thought"),
        "domain": sample.get("domain", "generic_guji"),
    }
    selected_keys = keys if keys is not None else required_keys
    if selected_keys is not None:
        return {k: item.get(k, "") for k in selected_keys}
    return item


def get_default_dataset():
    return GujiGenerationDataset(
        loc=resolve_repo_data_path("chain_of_thought", "punctuation_restoration.json"),
        sub_task="punctuation",
        reasoning_mode="chain_of_thought",
        domain="traditional_culture",
    )
