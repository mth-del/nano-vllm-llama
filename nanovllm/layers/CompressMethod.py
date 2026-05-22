"""Pluggable KV compression algorithms. Implement compress_fn or use SnapKV."""

import torch


def SnapKV(Q, K, V, num_keep=220, window=5):
    """
    Q: [B, Hq, window, D]
    K: [B, Hk, L, D]
    V: [B, Hk, L, D]  (unused; kept for API compatibility)
    Returns:
        final_idx: [B, 1 + num_keep + window] indices into K/V length L
        or False to skip compression for this batch.
    """
    del V
    B, Hk, L, D = K.shape
    Hq = Q.size(1)
    device = Q.device

    if L - window <= 0 or L <= num_keep + window:
        return False

    K_cut = K[:, :, :-window, :]
    scale = 1.0 / (D ** 0.5)

    if Hq == Hk:
        attn_scores = torch.matmul(Q, K_cut.transpose(-1, -2)) * scale
    else:
        assert Hq % Hk == 0, f"GQA requires Hq % Hk == 0, got Hq={Hq}, Hk={Hk}"
        group_size = Hq // Hk
        Q_grouped = Q.view(B, Hk, group_size, -1, D)
        K_t = K_cut.transpose(-1, -2).unsqueeze(2)
        attn_scores = torch.matmul(Q_grouped, K_t) * scale
        attn_scores = attn_scores.view(B, Hq, Q.size(2), L - window)

    attn_scores[:, :, :, 0] = -float("inf")
    attn_probs = torch.softmax(attn_scores, dim=-1)
    key_importance = attn_probs.sum(dim=2)
    if key_importance.size(1) > 1:
        key_importance = key_importance.sum(dim=1, keepdim=True)

    _, idx_keep = torch.topk(key_importance, k=num_keep, dim=-1, largest=True, sorted=False)
    idx_keep = torch.sort(idx_keep, dim=-1).values
    base_idx = idx_keep.view(B, -1)

    tail_idx = torch.arange(L - window, L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    bos_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
    return torch.cat([bos_idx, base_idx, tail_idx], dim=-1)


# Default algorithm hook: (Q, K, V, **kwargs) -> keep_idx | False
DEFAULT_COMPRESS_FN = SnapKV
