# SPDX-License-Identifier: Apache-2.0
"""Sampling operations for Metal backend."""

import torch
from torch.nn import functional


def sampling_from_probs(
    probs: torch.Tensor,
    random_numbers: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor:
    """Sample token indices from probability distributions.

    Args:
        probs: Probability tensor [batch_size, vocab_size]
        random_numbers: Random numbers for sampling [batch_size]
        deterministic: If True, use argmax instead of sampling

    Returns:
        Sampled token indices [batch_size]
    """
    if deterministic:
        return probs.argmax(dim=-1)

    # Use cumulative sum for sampling
    cumsum = probs.cumsum(dim=-1)
    random_numbers = random_numbers.unsqueeze(-1)

    # Find the first index where cumsum >= random_number
    samples = (cumsum >= random_numbers).int().argmax(dim=-1)

    return samples


def top_p_sampling(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply top-p (nucleus) sampling to logits.

    Args:
        logits: Logit tensor [batch_size, vocab_size]
        top_p: Top-p threshold (0.0 to 1.0)
        temperature: Sampling temperature

    Returns:
        Sampled token indices [batch_size]
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = functional.softmax(sorted_logits, dim=-1)
    cumsum_probs = sorted_probs.cumsum(dim=-1)

    # Create mask for tokens to remove
    sorted_mask = cumsum_probs - sorted_probs > top_p

    # Set masked logits to -inf
    sorted_logits[sorted_mask] = float("-inf")

    # Sample from filtered distribution
    probs = functional.softmax(sorted_logits, dim=-1)
    sampled_sorted_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Map back to original indices
    sampled_idx = sorted_indices.gather(dim=-1, index=sampled_sorted_idx.unsqueeze(-1))
    return sampled_idx.squeeze(-1)


def top_k_sampling(
    logits: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply top-k sampling to logits.

    Args:
        logits: Logit tensor [batch_size, vocab_size]
        top_k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        Sampled token indices [batch_size]
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Get top-k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

    # Sample from top-k
    probs = functional.softmax(top_k_logits, dim=-1)
    sampled_top_k_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Map back to original indices
    sampled_idx = top_k_indices.gather(dim=-1, index=sampled_top_k_idx.unsqueeze(-1))
    return sampled_idx.squeeze(-1)


def top_k_top_p_sampling(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply combined top-k and top-p sampling.

    First applies top-k, then top-p filtering.

    Args:
        logits: Logit tensor [batch_size, vocab_size]
        top_k: Number of top tokens to consider
        top_p: Top-p threshold
        temperature: Sampling temperature

    Returns:
        Sampled token indices [batch_size]
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k first
    vocab_size = logits.shape[-1]
    if top_k > 0 and top_k < vocab_size:
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

        # Create a mask for non-top-k tokens
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
        logits = mask

    # Apply top-p
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = functional.softmax(sorted_logits, dim=-1)
        cumsum_probs = sorted_probs.cumsum(dim=-1)

        # Create mask for tokens to remove
        sorted_mask = cumsum_probs - sorted_probs > top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Unsort
        logits = sorted_logits.gather(dim=-1, index=sorted_indices.argsort(dim=-1))

    # Sample
    probs = functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Greedy (argmax) sampling.

    Args:
        logits: Logit tensor [batch_size, vocab_size]

    Returns:
        Selected token indices [batch_size]
    """
    return logits.argmax(dim=-1)


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Logit tensor
        temperature: Temperature value (> 0)

    Returns:
        Scaled logits
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits.

    Args:
        logits: Logit tensor [batch_size, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        penalty: Repetition penalty (1.0 = no penalty)

    Returns:
        Penalized logits
    """
    if penalty == 1.0:
        return logits

    batch_size = logits.shape[0]
    for i in range(batch_size):
        unique_tokens = input_ids[i].unique()
        for token_id in unique_tokens:
            if logits[i, token_id] < 0:
                logits[i, token_id] *= penalty
            else:
                logits[i, token_id] /= penalty

    return logits
