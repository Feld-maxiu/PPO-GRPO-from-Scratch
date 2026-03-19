"""DPO (Direct Preference Optimization) training entry point.

Online DPO: generate 2 responses per query, score with reward model,
higher=chosen, lower=rejected, skip if equal.

Usage:
    python scripts/train_dpo.py --config configs/dpo.yaml
    python scripts/train_dpo.py --config configs/dpo.yaml --resume_from outputs/checkpoints/dpo/iter_5
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
from src.losses import dpo_loss
from src.agents.dpo_agent import DPOAgent
from src.evaluation import evaluate_model
from src.checkpoint import save_checkpoint, load_checkpoint
from src.logging_utils import setup_logger, log_training_step, log_eval_results


def train_dpo():
    cfg = load_config()
    device = get_device()
    set_seed(cfg.seed)

    num_gens = cfg.num_generations_per_query or 2

    print("=" * 60)
    print("DPO Training (Online)")
    print("=" * 60)
    print(f"Model: {cfg.model_path}")
    print(f"Device: {device}")
    print(f"Iterations: {cfg.num_iterations}, Steps/iter: {cfg.steps_per_iteration}")
    print(f"Beta: {cfg.beta}, Generations/query: {num_gens}")
    print(f"LR: {cfg.learning_rate}, AMP: {cfg.use_amp}")
    print("=" * 60)

    # Initialize
    agent = DPOAgent(cfg.model_path, device=str(device))
    optimizer = optim.Adam(agent.policy_model.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler() if cfg.use_amp and device.type == "cuda" else None

    writer = setup_logger(cfg.log_dir, "dpo")

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

        iter_dpo_loss = []
        iter_reward_margin = []
        iter_implicit_acc = []
        iter_rewards = []
        pairs_skipped = 0
        pairs_total = 0

        for step_idx in tqdm(range(len(queries)), desc="DPO steps"):
            query = queries[step_idx]
            gt = ground_truths[step_idx]

            # 1. Generate num_gens responses for this query
            gen_queries = [query] * num_gens
            gen_gts = [gt] * num_gens

            with torch.no_grad():
                responses, full_ids, prompt_lengths = agent.generate_responses(
                    gen_queries, max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                )

            # 2. Score with combined reward
            combined_rewards, _ = compute_combined_reward(
                responses, gen_gts,
                correctness_weight=cfg.correctness_weight,
                format_weight=cfg.format_weight,
                length_penalty_weight=cfg.length_penalty_weight,
                max_response_length=cfg.max_response_length,
            )
            iter_rewards.extend(combined_rewards)

            # 3. Select chosen/rejected pair (highest vs lowest reward)
            pairs_total += 1
            sorted_indices = sorted(range(num_gens), key=lambda i: combined_rewards[i], reverse=True)
            chosen_idx = sorted_indices[0]
            rejected_idx = sorted_indices[-1]

            if combined_rewards[chosen_idx] == combined_rewards[rejected_idx]:
                pairs_skipped += 1
                global_step += 1
                continue

            # 4. Compute log-probs for chosen and rejected under policy and reference
            chosen_ids = full_ids[chosen_idx:chosen_idx+1]
            rejected_ids = full_ids[rejected_idx:rejected_idx+1]
            chosen_plen = [prompt_lengths[chosen_idx]]
            rejected_plen = [prompt_lengths[rejected_idx]]

            # Policy log-probs (with gradients)
            policy_chosen_lp = agent.get_policy_log_probs(chosen_ids, chosen_plen, use_amp=cfg.use_amp)[0]
            policy_rejected_lp = agent.get_policy_log_probs(rejected_ids, rejected_plen, use_amp=cfg.use_amp)[0]

            # Reference log-probs (no gradients)
            with torch.no_grad():
                ref_chosen_lp = agent.get_reference_log_probs(chosen_ids, chosen_plen, use_amp=cfg.use_amp)[0]
                ref_rejected_lp = agent.get_reference_log_probs(rejected_ids, rejected_plen, use_amp=cfg.use_amp)[0]

            # Sum log-probs over tokens to get sequence-level log-probs
            policy_chosen_sum = policy_chosen_lp.sum().unsqueeze(0)
            policy_rejected_sum = policy_rejected_lp.sum().unsqueeze(0)
            ref_chosen_sum = ref_chosen_lp.sum().unsqueeze(0)
            ref_rejected_sum = ref_rejected_lp.sum().unsqueeze(0)

            # 5. DPO loss
            loss, reward_margin, implicit_acc = dpo_loss(
                policy_chosen_sum, policy_rejected_sum,
                ref_chosen_sum, ref_rejected_sum,
                beta=cfg.beta,
            )

            amp_backward_step(loss, optimizer, scaler, agent.policy_model, cfg.max_grad_norm)

            iter_dpo_loss.append(loss.item())
            iter_reward_margin.append(reward_margin)
            iter_implicit_acc.append(implicit_acc)

            global_step += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Log iteration metrics
        skip_rate = pairs_skipped / pairs_total if pairs_total > 0 else 0
        metrics = {
            "reward_mean": float(np.mean(iter_rewards)),
            "dpo_loss": float(np.mean(iter_dpo_loss)) if iter_dpo_loss else 0,
            "reward_margin": float(np.mean(iter_reward_margin)) if iter_reward_margin else 0,
            "implicit_accuracy": float(np.mean(iter_implicit_acc)) if iter_implicit_acc else 0,
            "pairs_skipped_rate": skip_rate,
        }
        log_training_step(writer, iteration, metrics)
        print(f"  Reward: {metrics['reward_mean']:.4f}")
        print(f"  DPO loss: {metrics['dpo_loss']:.4f}, Reward margin: {metrics['reward_margin']:.4f}")
        print(f"  Implicit accuracy: {metrics['implicit_accuracy']:.4f}, Skip rate: {skip_rate:.2%}")

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
            save_checkpoint(agent, optimizer, scaler, iteration + 1, metrics, cfg.output_dir, "dpo")

    writer.close()
    print("\nDPO training complete!")


if __name__ == "__main__":
    train_dpo()
