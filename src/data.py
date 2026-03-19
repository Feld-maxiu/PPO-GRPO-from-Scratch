"""Dataset construction for GSM8K training and evaluation."""

import re
from datasets import load_dataset


def _post_process(example, system_prompt):
    """Extract ground truth and format query."""
    match = re.search(r"####\s*(-?\d+)", example["answer"])
    example["ground_truth"] = match.group(1) if match else None
    example["query"] = f"{system_prompt}\n\nQuestion: {example['question']}"
    return example


def build_dataset(dataset_path: str, system_prompt: str, num_samples: int, split: str = "train"):
    """Build training dataset from GSM8K."""
    dataset_dict = load_dataset(dataset_path, "main")
    dataset = dataset_dict[split]
    dataset = dataset.map(lambda ex: _post_process(ex, system_prompt))
    dataset = dataset.remove_columns(["question", "answer"])
    num_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))
    return dataset


def build_eval_dataset(dataset_path: str, system_prompt: str, num_samples: int):
    """Build evaluation dataset from GSM8K test split."""
    dataset_dict = load_dataset(dataset_path, "main")
    dataset = dataset_dict["test"]
    dataset = dataset.map(lambda ex: _post_process(ex, system_prompt))
    dataset = dataset.remove_columns(["question", "answer"])
    num_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))
    return dataset
