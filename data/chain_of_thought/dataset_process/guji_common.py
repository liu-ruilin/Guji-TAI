
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_data_path(*parts):
    return str(PROJECT_ROOT.joinpath("data", *parts))


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()

def normalize_qa_sample(sample, domain="generic"):
    question = clean_text(sample.get("instruction", ""))
    context = clean_text(sample.get("input", ""))
    answer = clean_text(sample.get("output", ""))

    if context:
        prompt = f"请根据给定内容回答问题。\n\n内容：{context}\n\n问题：{question}\n\n回答："
    else:
        prompt = f"请回答下列古籍知识问题。\n\n问题：{question}\n\n回答："

    return {
        "prompt": prompt,
        "question": question,
        "context": context,
        "task_aware_target_tokens": answer,
        "task_type": "qa",
        "domain": domain,
    }
def normalize_generation_sample(
    sample,
    sub_task="generic_generation",
    reasoning_mode="0_shot",
    domain="generic_guji"
):
    instruction = clean_text(sample.get("instruction", ""))
    source_text = clean_text(sample.get("input", ""))
    answer = clean_text(sample.get("output", ""))

    # 生成类保留原 instruction，直接拼成 prompt
    if source_text:
        prompt = f"{instruction}\n{source_text}"
    else:
        prompt = instruction

    return {
        "prompt": prompt,
        "source_text": source_text,
        "task_aware_target_tokens": answer,
        "task_type": "generation",
        "sub_task": sub_task,
        "reasoning_mode": reasoning_mode,
        "domain": domain,
    }
def normalize_relation_extraction_sample(
    sample,
    domain="traditional_culture"
):
    instruction = clean_text(sample.get("instruction", ""))
    source_text = clean_text(sample.get("input", ""))
    answer = clean_text(sample.get("output", ""))

    prompt = f"{instruction}\n文本：{source_text}"

    return {
        "prompt": prompt,
        "source_text": source_text,
        "task_aware_target_tokens": answer,
        "task_type": "structured_understanding",
        "sub_task": "relation_extraction",
        "reasoning_mode": "0_shot",
        "domain": domain,
    }


def normalize_kb_reasoning_sample(
    sample,
    domain="traditional_culture"
):
    instruction = clean_text(sample.get("instruction", ""))
    source_text = clean_text(sample.get("input", ""))
    answer = clean_text(sample.get("output", ""))

    prompt = f"{instruction}\n{source_text}"

    return {
        "prompt": prompt,
        "source_text": source_text,
        "task_aware_target_tokens": answer,
        "task_type": "structured_understanding",
        "sub_task": "kb_reasoning",
        "reasoning_mode": "0_shot",
        "domain": domain,
    }
