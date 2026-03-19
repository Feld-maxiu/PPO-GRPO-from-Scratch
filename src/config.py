"""Configuration: YAML loading + argparse + Config dataclass."""

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class Config:
    # Model & dataset
    model_path: str = ""
    dataset_path: str = ""
    output_dir: str = "./outputs"
    system_prompt: str = ""

    # Training
    algorithm: str = "ppo"
    seed: int = 42
    learning_rate: float = 5e-6
    max_grad_norm: float = 0.5
    use_amp: bool = True
    train_samples: int = 500
    eval_samples: int = 200
    num_iterations: int = 10
    steps_per_iteration: int = 50
    ppo_epochs: int = 2
    max_new_tokens: int = 256
    epsilon: float = 0.2

    # Rewards
    correctness_weight: float = 1.0
    format_weight: float = 0.2
    length_penalty_weight: float = 0.0
    max_response_length: int = 512

    # Evaluation
    eval_every: int = 1
    eval_max_new_tokens: int = 256

    # Checkpoint
    save_every: int = 1
    resume_from: Optional[str] = None

    # Logging
    log_dir: str = "./outputs/runs"

    # PPO-specific
    value_coef: Optional[float] = None
    entropy_coef: Optional[float] = None
    gae_gamma: Optional[float] = None
    gae_lambda: Optional[float] = None

    # GRPO-specific
    group_size: Optional[int] = None
    beta: Optional[float] = None
    temperature: Optional[float] = None

    # DPO-specific
    num_generations_per_query: Optional[int] = None


def _merge_dicts(*dicts):
    """Merge multiple dicts, later values override earlier."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def load_config(argv=None) -> Config:
    """Load config by merging base.yaml <- algo.yaml <- CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to algorithm config (e.g. configs/ppo.yaml)")
    # Allow any --key value override from CLI
    known, unknown = parser.parse_known_args(argv)

    config_dir = os.path.dirname(known.config)
    base_path = os.path.join(config_dir, "base.yaml")

    base_cfg = {}
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    with open(known.config, "r", encoding="utf-8") as f:
        algo_cfg = yaml.safe_load(f) or {}

    # Parse CLI overrides: --key value pairs from unknown args
    cli_cfg = {}
    i = 0
    field_names = {f.name for f in fields(Config)}
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg[2:]
            if key in field_names and i + 1 < len(unknown):
                val = unknown[i + 1]
                # Infer type from dataclass field
                for f in fields(Config):
                    if f.name == key:
                        if f.type in (int, "int", Optional[int]):
                            val = int(val)
                        elif f.type in (float, "float", Optional[float]):
                            val = float(val)
                        elif f.type in (bool, "bool"):
                            val = val.lower() in ("true", "1", "yes")
                        break
                cli_cfg[key] = val
                i += 2
                continue
        i += 1

    merged = _merge_dicts(base_cfg, algo_cfg, cli_cfg)

    # Build Config, only pass fields that exist in the dataclass
    valid_keys = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in merged.items() if k in valid_keys}
    return Config(**filtered)
