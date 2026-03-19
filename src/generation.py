"""Batch generation and log-prob computation utilities."""

import torch
from torch.amp import autocast


@torch.no_grad()
def batch_generate(model, tokenizer, queries, max_new_tokens=256, temperature=1.0, device="cuda"):
    """Generate responses for a batch of queries using model.generate().

    Args:
        model: The causal LM (policy_model).
        tokenizer: The tokenizer.
        queries: List of query strings.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        device: Device string.

    Returns:
        responses: List of decoded response strings.
        full_ids: Tensor of shape (batch, seq_len) containing prompt + response token ids.
        prompt_lengths: List of ints, the length of each prompt in tokens.
    """
    encodings = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # Decode only the generated part
    responses = []
    for i in range(len(queries)):
        plen = int(prompt_lengths[i])
        gen_tokens = full_ids[i, plen:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        responses.append(text)

    return responses, full_ids, prompt_lengths


def compute_log_probs_for_tokens(model, full_ids, prompt_lengths, attention_mask=None, use_amp=False, device="cuda"):
    """Compute per-token log-probs for the generated portion of sequences.

    Does a single forward pass over the full sequence (prompt + response) and extracts
    log-probs only for the response tokens.

    Args:
        model: The causal LM.
        full_ids: (batch, seq_len) token ids.
        prompt_lengths: List of int prompt lengths.
        attention_mask: Optional attention mask.
        use_amp: Whether to use AMP.
        device: Device string.

    Returns:
        log_probs_list: List of 1D tensors, one per sample, containing per-token log-probs
                        for the response tokens. Length varies per sample.
    """
    full_ids = full_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with autocast(device_type=device_type, enabled=use_amp):
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

    # logits[:, t, :] predicts token at position t+1
    # So log_probs for token at position t is extracted from logits[:, t-1, :]
    log_probs_all = torch.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)

    log_probs_list = []
    for i in range(full_ids.size(0)):
        plen = int(prompt_lengths[i])
        seq_len = full_ids.size(1)
        # Response tokens are at positions plen..seq_len-1
        # Their log-probs come from logits at positions plen-1..seq_len-2
        if plen >= seq_len:
            log_probs_list.append(torch.tensor([], device=device))
            continue
        response_token_ids = full_ids[i, plen:seq_len]  # (response_len,)
        logits_for_response = log_probs_all[i, plen - 1 : seq_len - 1, :]  # (response_len, vocab)
        token_log_probs = logits_for_response.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        log_probs_list.append(token_log_probs)

    return log_probs_list
