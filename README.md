# LLM Post-Training: PPO, GRPO & DPO

From-scratch implementations of three post-training algorithms for large language models, applied to the GSM8K math reasoning benchmark.

## Algorithms

| Algorithm | Key Idea | Value Network | Reference Model |
|-----------|----------|:-------------:|:---------------:|
| **PPO** | Clipped surrogate + GAE advantage estimation | Yes | No |
| **GRPO** | Group-relative advantages, no value network needed | No | Yes (frozen) |
| **DPO** | Implicit reward from preference pairs, no RL loop | No | Yes (frozen) |

## Project Structure

```
L7/
├── configs/
│   ├── base.yaml              # Shared defaults (model, training, rewards)
│   ├── ppo.yaml               # PPO overrides (value_coef, GAE params)
│   ├── grpo.yaml              # GRPO overrides (group_size, beta, temperature)
│   └── dpo.yaml               # DPO overrides (beta, num_generations_per_query)
├── src/
│   ├── agents/
│   │   ├── base_agent.py      # BaseAgent: model loading, batch generate, log-probs
│   │   ├── ppo_agent.py       # PPOAgent + ValueHead
│   │   ├── grpo_agent.py      # GRPOAgent + frozen reference model
│   │   └── dpo_agent.py       # DPOAgent + frozen reference model
│   ├── config.py              # YAML loading + argparse + Config dataclass
│   ├── data.py                # Dataset construction (train/eval splits)
│   ├── rewards.py             # correctness, format, length_penalty, combined
│   ├── losses.py              # ppo_loss, grpo_loss, dpo_loss, GAE, GRPO advantages
│   ├── generation.py          # batch_generate(), compute_log_probs_for_tokens()
│   ├── evaluation.py          # evaluate_model() on test set
│   ├── checkpoint.py          # save/load with resume support
│   ├── logging_utils.py       # TensorBoard writer helpers
│   └── train_utils.py         # set_seed, get_device, amp_backward_step
├── scripts/
│   ├── train_ppo.py           # PPO training entry point
│   ├── train_grpo.py          # GRPO training entry point
│   ├── train_dpo.py           # DPO training entry point (online)
│   └── evaluate.py            # Standalone evaluation
├── gsm8k/                     # GSM8K dataset (local copy)
├── ppo_from_scratch.py        # Original reference implementation
├── grpo_from_scratch.py       # Original reference implementation
└── requirements.txt
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
# PPO
python scripts/train_ppo.py --config configs/ppo.yaml

# GRPO
python scripts/train_grpo.py --config configs/grpo.yaml

# DPO (online)
python scripts/train_dpo.py --config configs/dpo.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --model_path outputs/checkpoints/ppo/iter_10
```

### Monitor with TensorBoard

```bash
tensorboard --logdir outputs/runs
```

### Resume from checkpoint

```bash
python scripts/train_ppo.py --config configs/ppo.yaml --resume_from outputs/checkpoints/ppo/iter_5
```

### Override config from CLI

```bash
python scripts/train_ppo.py --config configs/ppo.yaml --learning_rate 1e-5 --num_iterations 20
```

## Configuration

Configs follow a 3-level merge hierarchy: `base.yaml` (shared) ← `algo.yaml` (overrides) ← CLI args (highest priority).

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_samples` | 500 | Number of training examples |
| `eval_samples` | 200 | Number of evaluation examples |
| `num_iterations` | 10 | Training iterations |
| `steps_per_iteration` | 50 | Steps per iteration |
| `max_new_tokens` | 256 | Max generation length |
| `learning_rate` | 5e-6 | Adam learning rate |
| `epsilon` | 0.2 | PPO clip range |

### Algorithm-specific

| PPO | GRPO | DPO |
|-----|------|-----|
| `value_coef=0.5` | `group_size=4` | `beta=0.1` |
| `entropy_coef=0.01` | `beta=0.01` | `num_generations_per_query=2` |
| `gae_gamma=0.99` | `temperature=1.0` | `temperature=1.0` |
| `gae_lambda=0.95` | | |

## Multi-Reward System

Combined reward = `correctness_weight × correctness + format_weight × format + length_penalty_weight × length_penalty`

- **Correctness**: 1.0 if `\boxed{}` answer matches ground truth, else 0.0
- **Format**: 1.0 if response contains `\boxed{}`, else 0.0
- **Length penalty**: 0.0 if within limit, scales to -1.0 for overly long responses

## Architecture Decisions

1. **Batch generation**: Uses `model.generate()` with KV-cache instead of token-by-token loops. Per-token log-probs are recovered via a single forward pass over completed sequences.

2. **Online DPO**: Generates 2 responses per query, uses combined reward to label chosen/rejected. Pairs with equal reward are skipped (skip rate is logged).

3. **Shared BaseAgent**: All three algorithms inherit from `BaseAgent` which handles model loading, tokenizer setup, and batch generation/log-prob computation.

## Algorithm Details

### PPO (Proximal Policy Optimization)

```
for each iteration:
    generate responses → compute rewards → get old log-probs + values
    → GAE advantages → PPO clipped update (policy + value + entropy)
```

Loss: `L = L_clip + value_coef * L_value - entropy_coef * H`

### GRPO (Group Relative Policy Optimization)

```
for each query:
    generate G responses → compute rewards → group-relative advantages
    → get policy + reference log-probs → PPO-clip + KL penalty update
```

Loss: `L = L_clip + beta * KL(ref || policy)`

### DPO (Direct Preference Optimization)

```
for each query:
    generate 2 responses → score with reward → chosen/rejected
    → policy + ref log-probs for both → DPO loss
```

Loss: `L = -log(σ(β * (log(π/ref)_chosen - log(π/ref)_rejected)))`

## References

1. Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
3. Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)

## License

MIT License
