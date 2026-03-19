"""Reward functions for GSM8K math tasks."""

import re
from typing import List, Dict, Tuple


def correctness_reward(completions: List[str], ground_truths: List[str]) -> List[float]:
    """Check if \\boxed{} content matches ground truth. Returns 1.0 or 0.0."""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        match = re.search(r"\\boxed\{(.*?)\}", completion)
        content = match.group(1).strip() if match else ""
        rewards.append(1.0 if content == gt else 0.0)
    return rewards


def format_reward(completions: List[str]) -> List[float]:
    """Check if response contains the \\boxed{} format. Returns 1.0 or 0.0."""
    rewards = []
    for completion in completions:
        has_format = bool(re.search(r"\\boxed\{.*?\}", completion))
        rewards.append(1.0 if has_format else 0.0)
    return rewards


def length_penalty(completions: List[str], max_length: int = 512) -> List[float]:
    """Penalize overly long responses. Returns 0.0 (good) to -1.0 (too long)."""
    rewards = []
    for completion in completions:
        length = len(completion)
        if length <= max_length:
            rewards.append(0.0)
        else:
            penalty = -min((length - max_length) / max_length, 1.0)
            rewards.append(penalty)
    return rewards


def compute_combined_reward(
    completions: List[str],
    ground_truths: List[str],
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    length_penalty_weight: float = 0.0,
    max_response_length: int = 512,
) -> Tuple[List[float], List[Dict[str, float]]]:
    """Compute weighted combination of all reward components.

    Returns:
        combined: List of combined reward scores.
        components: List of dicts with per-component scores for logging.
    """
    c_rewards = correctness_reward(completions, ground_truths)
    f_rewards = format_reward(completions)
    l_rewards = length_penalty(completions, max_response_length)

    combined = []
    components = []
    for c, f, l in zip(c_rewards, f_rewards, l_rewards):
        score = correctness_weight * c + format_weight * f + length_penalty_weight * l
        combined.append(score)
        components.append({"correctness": c, "format": f, "length_penalty": l, "combined": score})

    return combined, components
