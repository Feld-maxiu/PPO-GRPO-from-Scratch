"""Model evaluation on test set."""

import torch
import numpy as np
from src.generation import batch_generate
from src.rewards import correctness_reward, format_reward


@torch.no_grad()
def evaluate_model(model, tokenizer, eval_dataset, max_new_tokens=256, batch_size=8, device="cuda"):
    """Evaluate model on a dataset, computing accuracy and format rate.

    Args:
        model: The causal LM (policy_model).
        tokenizer: Tokenizer.
        eval_dataset: Dataset with 'query' and 'ground_truth' columns.
        max_new_tokens: Max generation length.
        batch_size: Batch size for generation.
        device: Device string.

    Returns:
        Dict with 'accuracy', 'format_rate', 'num_samples', and sample-level details.
    """
    model.eval()

    all_correct = []
    all_format = []
    sample_details = []

    queries = eval_dataset["query"]
    ground_truths = eval_dataset["ground_truth"]

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = queries[start:end]
        batch_gts = ground_truths[start:end]

        responses, _, _ = batch_generate(
            model, tokenizer, batch_queries,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            device=device,
        )

        c_rewards = correctness_reward(responses, batch_gts)
        f_rewards = format_reward(responses)

        all_correct.extend(c_rewards)
        all_format.extend(f_rewards)

        for q, r, gt, c, f in zip(batch_queries, responses, batch_gts, c_rewards, f_rewards):
            sample_details.append({
                "query": q[:100],
                "response": r[:200],
                "ground_truth": gt,
                "correct": c,
                "has_format": f,
            })

    accuracy = float(np.mean(all_correct)) if all_correct else 0.0
    format_rate = float(np.mean(all_format)) if all_format else 0.0

    return {
        "accuracy": accuracy,
        "format_rate": format_rate,
        "num_samples": len(all_correct),
        "details": sample_details,
    }
