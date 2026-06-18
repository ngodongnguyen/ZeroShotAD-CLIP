"""Stage 0 of CDMP: offline prototype discovery via spherical k-means.

Extracts L2-normalised multi-scale patch features from GT-anomalous regions
of the auxiliary training set, runs sklearn KMeans, saves K centroids, and
produces a t-SNE visualisation to verify cluster separability.

Typical usage (train on MVTec, test on VisA):
    python scripts/cluster_prototypes.py \\
        --data_path ./data/mvtecdataset --dataset mvtec \\
        --K 4 --output_dir ./centroids/K4_mvtec

Outputs
-------
    {output_dir}/cluster_centroids.pt   — (K, D_c) float32, L2-normalised
    {output_dir}/cluster_info.json      — metadata
    {output_dir}/tsne.png               — t-SNE coloured by cluster
"""

import os
import sys
import json
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import AnomalyCLIP_lib
from dataset import Dataset
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_anomalous_patch_features(model, dataloader, features_list,
                                     image_size, device, max_patches=500_000):
    """Iterate the dataset and collect L2-normalised multi-layer patch features
    for every patch that falls inside an anomalous GT region.

    Returns
    -------
    all_feats : (N_anom_patches, D_c) numpy float32 — L2-normalised concat of all layers
    """
    side = None   # patch grid side length (inferred from first batch)
    collected = []
    total = 0

    for items in tqdm(dataloader, desc="extracting features"):
        if items["anomaly"].item() == 0:
            continue   # skip normal images

        image   = items["img"].to(device)     # (1, C, H, W)
        gt_mask = items["img_mask"].squeeze()  # (H, W)  after target_transform

        with torch.no_grad():
            _, patch_features = model.encode_image(image, features_list, DPAM_layer=20)
            # patch_features: list of (1, N_all, D) where N_all = 1 + side^2

        if side is None:
            N_all = patch_features[0].shape[1]
            side  = int((N_all - 1) ** 0.5)
            tqdm.write(f"Patch grid: {side}×{side} = {side*side} patches/image "
                       f"(input {image_size}px, {len(features_list)} layers)")

        # Concat + L2-normalise multi-layer features (skip CLS at idx 0)
        normed_layers = [
            F.normalize(pf[:, 1:, :].float(), dim=-1)   # (1, N_patches, D)
            for pf in patch_features
        ]
        concat = torch.cat(normed_layers, dim=-1).squeeze(0)   # (N_patches, D_c)
        concat = F.normalize(concat, dim=-1)                    # spherical

        # Downsample GT mask to patch grid
        mask_down = F.adaptive_avg_pool2d(
            gt_mask.unsqueeze(0).unsqueeze(0).float(), (side, side)
        ).squeeze()                                             # (side, side)
        anom_mask = (mask_down > 0).reshape(-1)                # (N_patches,) bool

        if not anom_mask.any():
            continue

        anom_feats = concat[anom_mask].cpu().numpy()           # (n_anom, D_c)
        collected.append(anom_feats)
        total += len(anom_feats)

        if total >= max_patches:
            tqdm.write(f"  reached max_patches={max_patches}, stopping early")
            break

    if not collected:
        raise RuntimeError("No anomalous patches found — check dataset / mask paths.")

    all_feats = np.concatenate(collected, axis=0).astype(np.float32)
    print(f"Collected {len(all_feats)} anomalous patches "
          f"(D_c={all_feats.shape[1]})")
    return all_feats


def run_spherical_kmeans(feats, K, seed):
    """KMeans on L2-normalised features = spherical k-means."""
    print(f"Running KMeans K={K} on {len(feats)} samples (D={feats.shape[1]})...")
    km = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(feats)
    centroids = km.cluster_centers_.astype(np.float32)
    # Re-normalise centroids to unit sphere
    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
    centroids = centroids / norms
    print(f"KMeans inertia: {km.inertia_:.2f}")
    return centroids, labels


def make_tsne_plot(feats, labels, K, output_dir):
    """t-SNE visualisation coloured by cluster label."""
    n = min(len(feats), 10_000)
    idx = np.random.choice(len(feats), n, replace=False)
    sub_feats  = feats[idx]
    sub_labels = labels[idx]

    print(f"Computing t-SNE on {n} samples...")
    tsne  = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    embed = tsne.fit_transform(sub_feats)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for k in range(K):
        mask = sub_labels == k
        plt.scatter(embed[mask, 0], embed[mask, 1],
                    c=[colors[k]], label=f"cluster {k}",
                    alpha=0.5, s=6)
    plt.legend(markerscale=3, fontsize=9)
    plt.title(f"t-SNE of anomalous patches — K={K}")
    plt.tight_layout()
    path = os.path.join(output_dir, "tsne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"t-SNE saved → {path}")


def main():
    parser = argparse.ArgumentParser("CDMP Stage 0: cluster anomaly prototypes")
    parser.add_argument("--data_path",   type=str, default="./data/mvtecdataset")
    parser.add_argument("--dataset",     type=str, default="mvtec")
    parser.add_argument("--K",           type=int, default=4)
    parser.add_argument("--output_dir",  type=str, default="./centroids")
    parser.add_argument("--image_size",  type=int, default=518)
    parser.add_argument("--features_list", type=int, nargs="+",
                        default=[6, 12, 18, 24],
                        help="Layers to concat for clustering (must match cdmp/train.py).")
    parser.add_argument("--depth",   type=int, default=9)
    parser.add_argument("--n_ctx",   type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--max_patches", type=int, default=500_000,
                        help="Cap on total anomalous patches collected.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    print("Loading CLIP model...")
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device,
                                     design_details=AnomalyCLIP_parameters)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=20)

    preprocess, target_transform = get_transform(args)
    dataset = Dataset(root=args.data_path, transform=preprocess,
                      target_transform=target_transform,
                      dataset_name=args.dataset)
    # batch_size=1 to handle variable anomaly density
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # -------------------------------------------------- feature extraction
    feats = extract_anomalous_patch_features(
        model, dataloader, args.features_list, args.image_size,
        device, max_patches=args.max_patches
    )

    # -------------------------------------------------- spherical k-means
    centroids, labels = run_spherical_kmeans(feats, args.K, args.seed)

    # -------------------------------------------------- save
    centroid_path = os.path.join(args.output_dir, "cluster_centroids.pt")
    torch.save(torch.from_numpy(centroids), centroid_path)
    print(f"Centroids saved → {centroid_path}  shape={centroids.shape}")

    cluster_sizes = [(labels == k).sum().tolist() for k in range(args.K)]
    info = {
        "K":             args.K,
        "D_concat":      int(centroids.shape[1]),
        "n_layers":      len(args.features_list),
        "features_list": args.features_list,
        "dataset":       args.dataset,
        "n_anom_patches": int(len(feats)),
        "cluster_sizes":  cluster_sizes,
        "seed":          args.seed,
    }
    with open(os.path.join(args.output_dir, "cluster_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Cluster sizes:", dict(enumerate(cluster_sizes)))

    # -------------------------------------------------- t-SNE
    make_tsne_plot(feats, labels, args.K, args.output_dir)

    print("\nStage 0 complete.")
    print(f"  centroids : {centroid_path}")
    print(f"  Use in training with --centroid_path {centroid_path}")


if __name__ == "__main__":
    main()
