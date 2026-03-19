"""PPOAgent: BaseAgent + ValueHead for per-token value estimates."""

import torch
import torch.nn as nn
from torch.amp import autocast

from src.agents.base_agent import BaseAgent


class ValueHead(nn.Module):
    """Linear head that predicts a scalar value from hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden)
        return self.linear(hidden_states).squeeze(-1)  # (batch, seq_len)


class PPOAgent(BaseAgent):
    """PPO agent with a value head for advantage estimation."""

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path, device)
        hidden_size = self.policy_model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)

    def get_values(self, full_ids, prompt_lengths, use_amp=False):
        """Compute per-token value estimates for response tokens.

        Returns:
            List of 1D tensors (one per sample) with value estimates.
        """
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long().to(self.device)
        full_ids = full_ids.to(self.device)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = self.policy_model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
            all_values = self.value_head(hidden)  # (batch, seq_len)

        values_list = []
        for i in range(full_ids.size(0)):
            plen = int(prompt_lengths[i])
            seq_len = full_ids.size(1)
            # Values for response tokens (aligned with log-probs: positions plen..seq_len-1)
            # We take values at positions plen-1..seq_len-2 to align with the prediction targets
            vals = all_values[i, plen - 1: seq_len - 1]
            values_list.append(vals.detach())

        return values_list
