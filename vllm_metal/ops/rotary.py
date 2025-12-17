# SPDX-License-Identifier: Apache-2.0
"""Rotary positional embedding operations for Metal backend."""

import torch


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        positions: Position indices [num_tokens]
        query: Query tensor [num_tokens, num_heads * head_size]
        key: Key tensor [num_tokens, num_kv_heads * head_size]
        head_size: Size of each attention head
        cos_sin_cache: Precomputed cos/sin cache [max_seq_len, rotary_dim]
        is_neox: Whether to use NeoX-style rotary (interleaved vs split)

    Returns:
        Tuple of (rotated_query, rotated_key)
    """
    # Get rotary dimension (typically head_size or head_size // 2)
    rotary_dim = cos_sin_cache.shape[1] // 2

    # Reshape query and key
    num_tokens = query.shape[0]
    num_heads = query.shape[1] // head_size
    num_kv_heads = key.shape[1] // head_size

    query = query.view(num_tokens, num_heads, head_size)
    key = key.view(num_tokens, num_kv_heads, head_size)

    # Get cos and sin for each position
    cos = cos_sin_cache[positions, :rotary_dim]  # [num_tokens, rotary_dim]
    sin = cos_sin_cache[positions, rotary_dim:]  # [num_tokens, rotary_dim]

    # Expand for broadcasting
    cos = cos.unsqueeze(1)  # [num_tokens, 1, rotary_dim]
    sin = sin.unsqueeze(1)  # [num_tokens, 1, rotary_dim]

    # Apply rotary embedding
    if is_neox:
        # NeoX-style: split into two halves
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]

        query_rot = _apply_rotary_emb(query_rot, cos, sin)
        key_rot = _apply_rotary_emb(key_rot, cos, sin)

        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)
    else:
        # GPT-J style: interleaved
        query = _apply_rotary_emb_interleaved(query, cos, sin, rotary_dim)
        key = _apply_rotary_emb_interleaved(key, cos, sin, rotary_dim)

    # Reshape back
    query = query.view(num_tokens, num_heads * head_size)
    key = key.view(num_tokens, num_kv_heads * head_size)

    return query, key


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to tensor x.

    Args:
        x: Input tensor [..., rotary_dim]
        cos: Cosine values [..., rotary_dim]
        sin: Sine values [..., rotary_dim]

    Returns:
        Rotated tensor
    """
    # Split into even and odd indices
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    cos = cos[..., ::2]
    sin = sin[..., ::2]

    # Apply rotation
    # [x1, x2] @ [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


def _apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply rotary embedding with interleaved pattern (GPT-J style).

    Args:
        x: Input tensor [..., head_size]
        cos: Cosine values
        sin: Sine values
        rotary_dim: Dimension for rotary embedding

    Returns:
        Rotated tensor
    """
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Reshape for rotation
    x_rot = x_rot.reshape(*x_rot.shape[:-1], -1, 2)

    # Apply rotation
    x_rot_out = torch.stack(
        [
            x_rot[..., 0] * cos.squeeze(-2) - x_rot[..., 1] * sin.squeeze(-2),
            x_rot[..., 0] * sin.squeeze(-2) + x_rot[..., 1] * cos.squeeze(-2),
        ],
        dim=-1,
    ).flatten(-2)

    return torch.cat([x_rot_out, x_pass], dim=-1)


def create_cos_sin_cache(
    max_seq_len: int,
    head_size: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create cosine/sine cache for rotary embeddings.

    Args:
        max_seq_len: Maximum sequence length
        head_size: Size of attention head
        base: Base for computing frequencies
        dtype: Data type for the cache
        device: Device to create cache on

    Returns:
        Cache tensor of shape [max_seq_len, head_size]
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    )

    # Create position indices
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # Compute frequencies
    freqs = torch.outer(t, inv_freq)

    # Create cache: [max_seq_len, head_size]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    if device is not None:
        cache = cache.to(device)
    if dtype != torch.float32:
        cache = cache.to(dtype)

    return cache
