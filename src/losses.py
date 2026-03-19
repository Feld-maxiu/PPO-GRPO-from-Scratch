"""Loss functions: PPO, GRPO, DPO, GAE, GRPO advantages, KL divergence."""

import torch
import torch.nn as nn


def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: List of per-token rewards (length T).
        values: List of per-token value estimates (length T).
        gamma: Discount factor.
        gae_lambda: GAE lambda for bias-variance tradeoff.

    Returns:
        advantages: Tensor of shape (T,).
    """
    advantages = []
    gae = 0.0
    T = len(rewards)

    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)


def compute_grpo_advantages(rewards_group):
    """Compute GRPO group-relative advantages.

    A_i = (r_i - mean(r)) / (std(r) + eps)

    Args:
        rewards_group: List of reward scores for one group.

    Returns:
        advantages: List of float advantages.
    """
    rewards_tensor = torch.tensor(rewards_group, dtype=torch.float32)
    mean_r = rewards_tensor.mean()
    std_r = rewards_tensor.std()
    advantages = ((rewards_tensor - mean_r) / (std_r + 1e-8)).tolist()
    return advantages


def compute_kl_divergence(log_probs, ref_log_probs):
    """Approximate KL(ref || policy) = ref_log_probs - log_probs."""
    return ref_log_probs - log_probs


def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """PPO clipped surrogate loss.

    Args:
        old_log_probs: Log-probs from behavior policy (detached).
        new_log_probs: Log-probs from current policy.
        advantages: Advantage estimates.
        epsilon: Clip range.

    Returns:
        Scalar loss (negated, for gradient descent).
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    return -torch.min(surr1, surr2).mean()


def grpo_loss(old_log_probs, new_log_probs, ref_log_probs, advantages, epsilon=0.2, beta=0.01):
    """GRPO loss = PPO-clip policy loss + KL penalty.

    Args:
        old_log_probs: Log-probs from collection phase.
        new_log_probs: Log-probs from current policy.
        ref_log_probs: Log-probs from frozen reference model.
        advantages: GRPO advantages (tensor, same length as log_probs).
        epsilon: PPO clip range.
        beta: KL penalty coefficient.

    Returns:
        loss: Total scalar loss.
        policy_loss_val: Float policy loss component.
        kl_penalty_val: Float KL penalty component.
    """
    if not isinstance(advantages, torch.Tensor):
        advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = advantages.to(new_log_probs.device)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    kl = compute_kl_divergence(new_log_probs, ref_log_probs)
    kl_penalty = beta * kl.mean()

    loss = policy_loss + kl_penalty
    return loss, policy_loss.item(), kl_penalty.item()


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Direct Preference Optimization loss.

    L_DPO = -log(sigmoid(beta * (log(pi/ref)_chosen - log(pi/ref)_rejected)))

    Args:
        policy_chosen_logps: Sum of per-token log-probs for chosen, shape (batch,).
        policy_rejected_logps: Same for rejected.
        ref_chosen_logps: Reference model log-probs for chosen.
        ref_rejected_logps: Reference model log-probs for rejected.
        beta: Temperature parameter controlling deviation from reference.

    Returns:
        loss: Scalar DPO loss.
        reward_margin: Mean (chosen_reward - rejected_reward) for logging.
        implicit_accuracy: Fraction where chosen is preferred.
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    with torch.no_grad():
        reward_margin = (chosen_logratios - rejected_logratios).mean().item()
        implicit_accuracy = (logits > 0).float().mean().item()

    return loss, reward_margin, implicit_accuracy
