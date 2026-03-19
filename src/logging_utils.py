"""TensorBoard logging helpers."""

import os
from torch.utils.tensorboard import SummaryWriter


_writer = None


def setup_logger(log_dir: str, algorithm: str) -> SummaryWriter:
    """Create and return a TensorBoard SummaryWriter."""
    global _writer
    full_path = os.path.join(log_dir, algorithm)
    os.makedirs(full_path, exist_ok=True)
    _writer = SummaryWriter(log_dir=full_path)
    return _writer


def log_training_step(writer: SummaryWriter, step: int, metrics: dict):
    """Log a dict of scalar metrics at a given step."""
    for key, value in metrics.items():
        writer.add_scalar(f"train/{key}", value, step)
    writer.flush()


def log_eval_results(writer: SummaryWriter, step: int, metrics: dict):
    """Log evaluation metrics."""
    for key, value in metrics.items():
        writer.add_scalar(f"eval/{key}", value, step)
    writer.flush()
