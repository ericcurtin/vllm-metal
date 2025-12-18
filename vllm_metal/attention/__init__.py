# SPDX-License-Identifier: Apache-2.0
"""Metal attention backend implementations for vLLM."""

from vllm_metal._compat import VLLM_AVAILABLE

# These modules require vLLM to be installed
# They are only imported when vLLM is available
if VLLM_AVAILABLE:
    from vllm_metal.attention.backend import (
        MetalAttentionBackend,
        MetalAttentionMetadata,
    )
    from vllm_metal.attention.metal_attention import MetalAttentionImpl

    __all__ = [
        "MetalAttentionBackend",
        "MetalAttentionMetadata",
        "MetalAttentionImpl",
    ]
else:
    __all__ = []
