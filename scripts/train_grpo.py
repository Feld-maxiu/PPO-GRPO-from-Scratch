"""GRPO training entry point.

Usage:
    python scripts/train_grpo.py --config configs/grpo.yaml
    python scripts/train_grpo.py --config configs/grpo.yaml --resume_from outputs/checkpoints/grpo/iter_5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.amp import GradScaler
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.train_utils import get_device, set_seed, amp_backward_step
from src.data import build_dataset, build_eval_dataset
from src.rewards import compute_combined_reward
from src.losses import compute_grpo_advantages, grpo_loss
from src.agents.grpo_agent import GRPOAgent
from src.evaluation import evaluate_model
from src.checkpoint import save_checkpoint, load_checkpoint
from src.logging_utils import setup_logger, log_training_step, log_eval_results


def train_grpo():
    cfg = load_config()
    device = get_device()
    set_seed(cfg.seed)

    print("=" * 60)
    print("GRPO Training")
    print("=" * 60)
    print(f"Model: {cfg.model_path}")
    print(f"Device: {device}")
    print(f"Iterations: {cfg.num_iterations}, Steps/iter: {cfg.steps_per_iteration}")
    print(f"Group size: {cfg.group_size}, Beta: {cfg.beta}")
    print(f"LR: {cfg.learning_rate}, AMP: {cfg.use_amp}")
    print("=" * 60)

    # Initialize
    agent = GRPOAgent(cfg.model_path, device=str(device))
    # Only optimize policy model parameters (reference model is frozen)
    optimizer = optim.Adam(agent.policy_model.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler() if cfg.use_amp and device.type == "cuda" else None

    writer = setup_logger(cfg.log_dir, "grpo")

    # Datasets
    train_dataset = build_dataset(cfg.dataset_path, cfg.system_prompt, cfg.train_samples)
    eval_dataset = build_eval_dataset(cfg.dataset_path, cfg.system_prompt, cfg.eval_samples)

    # Resume
    start_iter = 0
    if cfg.resume_from:
        start_iter, _ = load_checkpoint(cfg.resume_from, agent, optimizer, scaler, device=str(device))
        start_iter += 1

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
        iter_kl_loss = []

        for step_idx in tqdm(range(len(queries)), desc="GRPO steps"):
            query = queries[step_idx]
            gt = ground_truths[step_idx]

            # 1. Replicate prompt group_size times and generate
            group_queries = [query] * cfg.group_size
            group_gts = [gt] * cfg.group_size

            with torch.no_grad():
                responses, full_ids, prompt_lengths = agent.generate_responses(
                    group_queries, max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                )

            # 2. Compute combined rewards
            combined_rewards, reward_components = compute_combined_reward(
                responses, group_gts,
                correctness_weight=cfg.correctness_weight,
                format_weight=cfg.format_weight,
                length_penalty_weight=cfg.length_penalty_weight,
                max_response_length=cfg.max_response_length,
            )
            iter_rewards.extend(combined_rewards)

            # 3. GRPO advantages
            advantages = compute_grpo_advantages(combined_rewards)

            # 4. Get old log-probs and reference log-probs
            with torch.no_grad():
                old_log_probs_list = agent.get_policy_log_probs(full_ids, prompt_lengths, use_amp=cfg.use_amp)
                ref_log_probs_list = agent.get_reference_log_probs(full_ids, prompt_lengths, use_amp=cfg.use_amp)

            # 5. Update for each sample in the group
            for g in range(cfg.group_size):
                old_lp = old_log_probs_list[g].detach()
                ref_lp = ref_log_probs_list[g]
                adv = advantages[g]

                if len(old_lp) == 0:
                    continue

                adv_tensor = torch.full_like(old_lp, adv)

                for _ in range(cfg.ppo_epochs):
                    new_lp_list = agent.get_policy_log_probs(
                        full_ids[g:g+1], [prompt_lengths[g]], use_amp=cfg.use_amp
                    )
                    new_lp = new_lp_list[0]

                    loss, p_loss, kl_loss = grpo_loss(
                        old_lp, new_lp, ref_lp, adv_tensor,
                        epsilon=cfg.epsilon, beta=cfg.beta,
                    )

                    amp_backward_step(loss, optimizer, scaler, agent.policy_model, cfg.max_grad_norm)

                    iter_policy_loss.append(p_loss)
                    iter_kl_loss.append(kl_loss)

            global_step += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Log iteration metrics
        metrics = {
            "reward_mean": float(np.mean(iter_rewards)),
            "reward_std": float(np.std(iter_rewards)),
            "policy_loss": float(np.mean(iter_policy_loss)) if iter_policy_loss else 0,
            "kl_penalty": float(np.mean(iter_kl_loss)) if iter_kl_loss else 0,
        }
        log_training_step(writer, iteration, metrics)
        print(f"  Reward: {metrics['reward_mean']:.4f} +/- {metrics['reward_std']:.4f}")
        print(f"  Policy loss: {metrics['policy_loss']:.4f}, KL penalty: {metrics['kl_penalty']:.4f}")

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
            save_checkpoint(agent, optimizer, scaler, iteration + 1, metrics, cfg.output_dir, "grpo")

    writer.close()
    print("\nGRPO training complete!")


if __name__ == "__main__":
    train_grpo()
