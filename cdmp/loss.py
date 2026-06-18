"""CDMP-specific losses.

cluster_local_loss   : per-cluster cross-entropy on patch features driven by
                       visual pseudo-labels derived from saved centroids.
prototype_repulsion  : margin loss preventing K anomaly text prototypes from
                       collapsing to the same point.
log_*                : diagnostic helpers (no gradients).
"""

import torch
import torch.nn.functional as F

_TEMP = 0.07  # matches AnomalyCLIP's temperature


def cluster_local_loss(patch_logits, gt_mask, patch_feats_concat, centroids):
    """Per-cluster cross-entropy driven by visual pseudo-labels.

    Args:
        patch_logits       : (B, N_patches, 1+K) raw cosine sims without temp or softmax.
                             Caller computes  patch_feat_normalised @ text_feats.T
        gt_mask            : (B, H, W) binary GT mask at full image resolution.
        patch_feats_concat : (B, N_patches, D_c) L2-normalised, multi-layer concat.
        centroids          : (K, D_c) L2-normalised cluster centroids (same space).

    Returns:
        Scalar loss.
    """
    B, N_patches, _ = patch_logits.shape
    side = int(N_patches ** 0.5)

    # Downsample GT mask to patch grid
    mask_down = F.adaptive_avg_pool2d(
        gt_mask.unsqueeze(1).float(), (side, side)
    ).squeeze(1).reshape(B, -1)          # (B, N_patches)

    anom_mask = mask_down > 0.5           # bool (B, N_patches)
    norm_mask  = ~anom_mask

    # Visual pseudo-labels: argmax cosine sim with centroids
    pseudo_k = (patch_feats_concat @ centroids.T).argmax(dim=-1)  # (B, N_patches) in [0,K-1]
    anom_targets = pseudo_k + 1                                    # shift: 0=normal, 1..K=anomaly

    loss_sum, n_terms = 0.0, 0
    for b in range(B):
        if anom_mask[b].any():
            logits_a = patch_logits[b][anom_mask[b]] / _TEMP   # (n_anom, 1+K)
            labels_a = anom_targets[b][anom_mask[b]]            # (n_anom,)
            loss_sum += F.cross_entropy(logits_a, labels_a)
            n_terms  += 1

        if norm_mask[b].any():
            logits_n = patch_logits[b][norm_mask[b]] / _TEMP   # (n_norm, 1+K)
            labels_n = torch.zeros(norm_mask[b].sum(), dtype=torch.long,
                                   device=patch_logits.device)
            loss_sum += F.cross_entropy(logits_n, labels_n)
            n_terms  += 1

    return loss_sum / max(n_terms, 1)


def prototype_repulsion_loss(t_anomaly, margin=-0.3):
    """Push pairwise cosine similarity between K anomaly prototypes below margin.

    Args:
        t_anomaly : (K, D) L2-normalised anomaly text features.
        margin    : target upper bound on cosine similarity (typically -0.3..0).

    Returns:
        Scalar (0 when K<2 or all pairs already below margin).
    """
    K = t_anomaly.shape[0]
    if K < 2:
        return t_anomaly.new_zeros(1).squeeze()
    cos = t_anomaly @ t_anomaly.T                                          # (K, K)
    upper = torch.triu(torch.ones(K, K, device=cos.device), diagonal=1).bool()
    return F.relu(cos[upper] - margin).mean()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def log_prototype_stats(t_anomaly, logger=None, prefix=""):
    """Log pairwise cosine similarities between K text prototypes."""
    K = t_anomaly.shape[0]
    if K < 2:
        return
    with torch.no_grad():
        cos = (t_anomaly @ t_anomaly.T).cpu()
    lines = [f"{prefix}pairwise proto cosine (collapse check):"]
    for i in range(K):
        for j in range(i + 1, K):
            lines.append(f"  ({i},{j}): {cos[i, j]:.4f}")
    _log("\n".join(lines), logger)


def log_activation_counts(pseudo_labels_flat, K, logger=None, prefix=""):
    """Count anomalous patches routed to each prototype (collapse = one proto >> others).

    pseudo_labels_flat : 1-D LongTensor with values in 0..K-1.
    """
    if pseudo_labels_flat.numel() == 0:
        return
    counts = torch.bincount(pseudo_labels_flat.cpu().long(), minlength=K)
    total  = counts.sum().item()
    fracs  = [f"k{k}:{counts[k].item()/max(total,1)*100:.1f}%" for k in range(K)]
    _log(f"{prefix}proto activations [{', '.join(fracs)}]  n={total}", logger)


def _log(msg, logger):
    if logger:
        logger.info(msg)
    else:
        print(msg)
