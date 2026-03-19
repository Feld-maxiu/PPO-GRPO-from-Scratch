"""BaseAgent: model loading, tokenizer setup, batch generation, log-prob extraction."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.generation import batch_generate, compute_log_probs_for_tokens


class BaseAgent(nn.Module):
    """Base agent wrapping a causal LM with generation and log-prob utilities."""

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_path)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.policy_model.config.pad_token_id = self.tokenizer.eos_token_id

        self.policy_model.to(self.device)

    def generate_responses(self, queries, max_new_tokens=256, temperature=1.0):
        """Generate responses for a batch of queries.

        Returns:
            responses: List of str.
            full_ids: Tensor (batch, seq_len).
            prompt_lengths: List of int.
        """
        return batch_generate(
            self.policy_model, self.tokenizer, queries,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=str(self.device),
        )

    def get_policy_log_probs(self, full_ids, prompt_lengths, use_amp=False):
        """Compute per-token log-probs under the current policy.

        Returns:
            List of 1D tensors (one per sample).
        """
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
        return compute_log_probs_for_tokens(
            self.policy_model, full_ids, prompt_lengths,
            attention_mask=attention_mask,
            use_amp=use_amp,
            device=str(self.device),
        )
