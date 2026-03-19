"""PPO training entry point.

Usage:
    python scripts/train_ppo.py --config configs/ppo.yaml
    python scripts/train_ppo.py --config configs/ppo.yaml --resume_from outputs/checkpoints/ppo/iter_5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.train_utils import get_device, set_seed, amp_backward_step
from src.data import build_dataset, build_eval_dataset
from src.rewards import compute_combined_reward
from src.losses import compute_gae, ppo_loss
from src.agents.ppo_agent import PPOAgent
from src.evaluation import evaluate_model
from src.checkpoint import save_checkpoint, load_checkpoint
from src.logging_utils import setup_logger, log_training_step, log_eval_results


def train_ppo():
    cfg = load_config()
    device = get_device()
    set_seed(cfg.seed)

    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Model: {cfg.model_path}")
    print(f"Device: {device}")
    print(f"Iterations: {cfg.num_iterations}, Steps/iter: {cfg.steps_per_iteration}")
    print(f"LR: {cfg.learning_rate}, AMP: {cfg.use_amp}")
    print("=" * 60)

    # Initialize
    agent = PPOAgent(cfg.model_path, device=str(device))
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler() if cfg.use_amp and device.type == "cuda" else None

    writer = setup_logger(cfg.log_dir, "ppo")

    # Datasets
    train_dataset = build_dataset(cfg.dataset_path, cfg.system_prompt, cfg.train_samples)
    eval_dataset = build_eval_dataset(cfg.dataset_path, cfg.system_prompt, cfg.eval_samples)

    # Resume
    start_iter = 0
    if cfg.resume_from:
        start_iter, _ = load_checkpoint(cfg.resume_from, agent, optimizer, scaler, device=str(device))
        start_iter += 1  # Resume from next iteration

    global_step = start_iter * cfg.steps_per_iteration

    for iteration in range(start_iter, cfg.num_iterations):
        agent.policy_model.train()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{cfg.num_iterations}")
        print(f"{'='*60}")

        start_idx = iteration * cfg.steps_per_iteration
        end_idx = min(start_idx + cfg.steps_per_iteration, len(train_dataset))
        if start_idx >= len(train_dataset):
            start_idx = start_idx % len(train_dataset)
            end_idx = min(start_idx + cfg.steps_per_iteration, len(train_dataset))
        iter_dataset = train_dataset.select(range(start_idx, end_idx))

        queries = iter_dataset["query"]
        ground_truths = iter_dataset["ground_truth"]

        iter_rewards = []
        iter_policy_loss = []
        iter_value_loss = []
        iter_entropy_loss = []

        # Process in mini-batches
        batch_size = min(8, len(queries))
        for batch_start in tqdm(range(0, len(queries), batch_size), desc="PPO steps"):
            batch_end = min(batch_start + batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]
            batch_gts = ground_truths[batch_start:batch_end]

            # 1. Generate responses
            with torch.no_grad():
                responses, full_ids, prompt_lengths = agent.generate_responses(
                    batch_queries, max_new_tokens=cfg.max_new_tokens
                )

            # 2. Compute combined rewards
            combined_rewards, reward_components = compute_combined_reward(
                responses, batch_gts,
                correctness_weight=cfg.correctness_weight,
                format_weight=cfg.format_weight,
                length_penalty_weight=cfg.length_penalty_weight,
                max_response_length=cfg.max_response_length,
            )
            iter_rewards.extend(combined_rewards)

            # 3. Get old log-probs and values (detached)
            with torch.no_grad():
                old_log_probs_list = agent.get_policy_log_probs(full_ids, prompt_lengths, use_amp=cfg.use_amp)
                values_list = agent.get_values(full_ids, prompt_lengths, use_amp=cfg.use_amp)

            # 4. PPO update for each sample
            for i in range(len(batch_queries)):
                old_lp = old_log_probs_list[i].detach()
                vals = values_list[i]
                reward = combined_rewards[i]

                if len(old_lp) == 0:
                    continue

                # Per-token rewards: 0 for all except last token
                token_rewards = [0.0] * (len(old_lp) - 1) + [reward]

                # GAE
                advantages = compute_gae(
                    token_rewards, vals.tolist(),
                    gamma=cfg.gae_gamma, gae_lambda=cfg.gae_lambda,
                ).to(device)

                returns = advantages + vals.to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO epochs
                for _ in range(cfg.ppo_epochs):
                    # Re-compute log-probs and values under current policy
                    new_lp_list = agent.get_policy_log_probs(
                        full_ids[i:i+1], [prompt_lengths[i]], use_amp=cfg.use_amp
                    )
                    new_lp = new_lp_list[0]

                    new_vals_list = agent.get_values(
                        full_ids[i:i+1], [prompt_lengths[i]], use_amp=cfg.use_amp
                    )
                    new_vals = new_vals_list[0]

                    # Policy loss
                    p_loss = ppo_loss(old_lp, new_lp, advantages, epsilon=cfg.epsilon)

                    # Value loss
                    v_loss = nn.MSELoss()(new_vals, returns)

                    # Entropy bonus (approximate from log-probs)
                    entropy = -(new_lp * torch.exp(new_lp)).mean()

                    loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

                    amp_backward_step(loss, optimizer, scaler, agent, cfg.max_grad_norm)

                    iter_policy_loss.append(p_loss.item())
                    iter_value_loss.append(v_loss.item())
                    iter_entropy_loss.append(entropy.item())

            global_step += (batch_end - batch_start)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Log iteration metrics
        metrics = {
            "reward_mean": float(np.mean(iter_rewards)),
            "reward_std": float(np.std(iter_rewards)),
            "policy_loss": float(np.mean(iter_policy_loss)) if iter_policy_loss else 0,
            "value_loss": float(np.mean(iter_value_loss)) if iter_value_loss else 0,
            "entropy": float(np.mean(iter_entropy_loss)) if iter_entropy_loss else 0,
        }
        log_training_step(writer, iteration, metrics)
        print(f"  Reward: {metrics['reward_mean']:.4f} +/- {metrics['reward_std']:.4f}")
        print(f"  Policy loss: {metrics['policy_loss']:.4f}, Value loss: {metrics['value_loss']:.4f}")

        # Evaluation
        if (iteration + 1) % cfg.eval_every == 0:
            print("  Evaluating...")
            eval_results = evaluate_model(
                agent.policy_model, agent.tokenizer, eval_dataset,
                max_new_tokens=cfg.eval_max_new_tokens, device=str(device),
            )
            log_eval_results(writer, iteration, {
                "accuracy": eval_results["accuracy"],
                "format_rate": eval_results["format_rate"],
            })
            print(f"  Eval accuracy: {eval_results['accuracy']:.4f}, format_rate: {eval_results['format_rate']:.4f}")

        # Checkpoint
        if (iteration + 1) % cfg.save_every == 0:
            save_checkpoint(agent, optimizer, scaler, iteration + 1, metrics, cfg.output_dir, "ppo")

    writer.close()
    print("\nPPO training complete!")


if __name__ == "__main__":
    train_ppo()
