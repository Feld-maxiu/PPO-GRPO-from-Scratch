"""GRPOAgent: BaseAgent + frozen reference model for KL divergence."""

import torch
from transformers import AutoModelForCausalLM

from src.agents.base_agent import BaseAgent
from src.generation import compute_log_probs_for_tokens


class GRPOAgent(BaseAgent):
    """GRPO agent with a frozen reference model for KL-penalized updates."""

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)

        # Frozen reference model
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.to(self.device)

    @torch.no_grad()
    def get_reference_log_probs(self, full_ids, prompt_lengths, use_amp=False):
        """Compute per-token log-probs under the frozen reference model.

        Returns:
            List of 1D tensors.
        """
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
        return compute_log_probs_for_tokens(
            self.reference_model, full_ids, prompt_lengths,
            attention_mask=attention_mask,
            use_amp=use_amp,
            device=str(self.device),
        )
