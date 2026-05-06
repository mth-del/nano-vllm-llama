from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_type: str | None = None,
        scaling_factor: float = 1.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 1.0,
        original_max_position_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if rope_type in {"linear", "dynamic"} and scaling_factor > 1.0:
            inv_freq = inv_freq / scaling_factor
        elif rope_type == "llama3" and scaling_factor > 1.0:
            original_ctx = original_max_position_embeddings or max_position_embeddings
            wavelen = 2 * torch.pi / inv_freq
            low_freq_wavelen = original_ctx / low_freq_factor
            high_freq_wavelen = original_ctx / high_freq_factor
            inv_freq_scaled = inv_freq / scaling_factor
            smooth = (original_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth = smooth.clamp(0.0, 1.0)
            inv_freq = torch.where(
                wavelen > low_freq_wavelen,
                inv_freq_scaled,
                torch.where(
                    wavelen < high_freq_wavelen,
                    inv_freq,
                    (1 - smooth) * inv_freq_scaled + smooth * inv_freq,
                ),
            )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=32)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_type: str | None = None,
    scaling_factor: float = 1.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 1.0,
    original_max_position_embeddings: int | None = None,
):
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_type=rope_type,
        scaling_factor=scaling_factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        original_max_position_embeddings=original_max_position_embeddings,
    )
    return rotary_emb
