"""Checkpoint save/load with resume support."""

import os
import torch


def save_checkpoint(model, optimizer, scaler, iteration, metrics, output_dir, algorithm):
    """Save a training checkpoint.

    Args:
        model: The agent (nn.Module).
        optimizer: Optimizer.
        scaler: GradScaler or None.
        iteration: Current iteration number.
        metrics: Dict of current metrics.
        output_dir: Base output directory.
        algorithm: Algorithm name (ppo/grpo/dpo).
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints", algorithm, f"iter_{iteration}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the policy model and tokenizer
    model.policy_model.save_pretrained(ckpt_dir)
    model.tokenizer.save_pretrained(ckpt_dir)

    # Save training state
    state = {
        "iteration": iteration,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
    print(f"Checkpoint saved: {ckpt_dir}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device="cuda"):
    """Load a checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint directory.
        model: The agent (nn.Module). Its policy_model will be reloaded.
        optimizer: Optimizer to restore state into (optional).
        scaler: GradScaler to restore state into (optional).
        device: Device string.

    Returns:
        iteration: The iteration number to resume from.
        metrics: The saved metrics dict.
    """
    from transformers import AutoModelForCausalLM

    # Reload model weights
    model.policy_model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)

    state_path = os.path.join(checkpoint_path, "training_state.pt")
    state = torch.load(state_path, map_location=device, weights_only=False)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])

    print(f"Checkpoint loaded from: {checkpoint_path} (iteration {state['iteration']})")
    return state["iteration"], state.get("metrics", {})
