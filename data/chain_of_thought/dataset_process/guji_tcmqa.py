import json
from data.chain_of_thought.dataset_process.dataset_base import Dataset
from data.chain_of_thought.dataset_process.guji_common import normalize_qa_sample, resolve_repo_data_path

dataset_info = {
    "name": "中医知识问答_0_shot",
    "des": "古籍问答类任务：中医知识问答",
    "dataset_type": "问答类"
}

support_template_keys = [
    "prompt",
    "question",
    "context",
    "task_aware_target_tokens",
    "task_type",
    "domain",
]

class GujiQADataset(Dataset):
    def __init__(self, loc, domain="generic"):
        self.loc = str(loc)
        self.domain = domain

        with open(self.loc, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.data = [normalize_qa_sample(x, domain=domain) for x in self.raw_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def samples(self):
        return self.data


def get_processed_kvs(sample, keys=None, required_keys=None):
    item = {
        "prompt": sample.get("prompt", ""),
        "question": sample.get("question", ""),
        "context": sample.get("context", ""),
        "task_aware_target_tokens": sample.get("task_aware_target_tokens", ""),
        "task_type": sample.get("task_type", "qa"),
        "domain": sample.get("domain", "tcm"),
    }
    selected_keys = keys if keys is not None else required_keys
    if selected_keys is not None:
        return {k: item.get(k, "") for k in selected_keys}
    return item


def get_default_dataset():
    return GujiQADataset(
        loc=resolve_repo_data_path("0_shot", "chinese_medicine_qa.json"),
        domain="tcm",
    )
