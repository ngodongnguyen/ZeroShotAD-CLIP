"""CDMP training script.

Drop-in replacement for train.py with K independent abnormal prototypes.
K=1 reproduces original AnomalyCLIP training exactly (regression-test safe).

Stage 0 must be run first to produce cluster_centroids.pt when K>1.
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import AnomalyCLIP_lib
from dataset import Dataset
from logger import get_logger
from loss import FocalLoss, BinaryDiceLoss
from utils import get_transform, normalize
from cdmp.prompt_learner import CDMPPromptLearner
from cdmp.loss import (
    cluster_local_loss,
    prototype_repulsion_loss,
    log_prototype_stats,
    log_activation_counts,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_centroids(path, device):
    """Load K×D_c L2-normalised centroids saved by scripts/cluster_prototypes.py."""
    ckpt = torch.load(path, map_location="cpu")
    # support both raw tensor save and dict save
    if isinstance(ckpt, dict):
        centroids = ckpt["centroids"]
    else:
        centroids = ckpt
    centroids = F.normalize(centroids.float(), dim=-1).to(device)
    return centroids  # (K, D_c)


def concat_and_normalize_patch_features(patch_features):
    """Concat L2-normalised features from all layers → (B, N_patches, D_c).

    patch_features : list of (B, N_all, D) tensors with CLS at index 0.
    """
    normed = [
        F.normalize(pf[:, 1:, :].float(), dim=-1)   # (B, N_patches, D)
        for pf in patch_features
    ]
    cat = torch.cat(normed, dim=-1)                   # (B, N_patches, D*n_layers)
    return F.normalize(cat, dim=-1)                   # re-normalise concat


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device,
                                     design_details=AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess,
                         target_transform=target_transform,
                         dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )

    # ------------------------------------------------------------------ model
    prompt_learner = CDMPPromptLearner(model.to("cpu"), AnomalyCLIP_parameters,
                                       cdmp_K=args.cdmp_K)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    optimizer = torch.optim.Adam(
        list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    loss_focal = FocalLoss()
    loss_dice  = BinaryDiceLoss()
    lam = 4

    # ------------------------------------------------- load centroids for K>1
    centroids = None
    if args.cdmp_K > 1:
        if not args.centroid_path:
            raise ValueError("--centroid_path required when cdmp_K > 1")
        centroids = load_centroids(args.centroid_path, device)
        K_ckpt = centroids.shape[0]
        if K_ckpt != args.cdmp_K:
            raise ValueError(
                f"centroid file has K={K_ckpt} but cdmp_K={args.cdmp_K}"
            )
        logger.info(f"Loaded centroids: {centroids.shape} from {args.centroid_path}")

    logger.info(f"CDMP K={args.cdmp_K}, dataset={args.dataset}, "
                f"features_list={args.features_list}")

    # ---------------------------------------------------------------- training
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list, image_loss_list = [], []
        epoch_pseudo_labels = []   # for activation-count diagnostics

        for items in tqdm(train_dataloader, leave=False):
            image = items["img"].to(device)
            label = items["anomaly"]
            gt    = items["img_mask"].squeeze().to(device)
            gt[gt > 0.5]  = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(
                    image, args.features_list, DPAM_layer=20
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # ---------------------------------------- encode text (1+K, D)
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner()
            feats = model.encode_text_learn(
                prompts, tokenized_prompts, compound_prompts_text
            ).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)   # (1+K, D)

            # -------------------------------- image-level loss (binary max-K)
            img_sim = image_features @ feats.T / 0.07           # (B, 1+K)
            if args.cdmp_K == 1:
                img_logits = img_sim                             # (B, 2) — same as original
            else:
                img_logits = torch.stack(
                    [img_sim[:, 0], img_sim[:, 1:].mean(dim=-1)], dim=-1
                )                                                # (B, 2)
            image_loss = F.cross_entropy(img_logits, label.long().to(device))
            image_loss_list.append(image_loss.item())

            # ------------------------------ multi-scale concat for pseudo-labels
            if centroids is not None:
                patch_feats_concat = concat_and_normalize_patch_features(patch_features)
                # collect for epoch diagnostics
                with torch.no_grad():
                    gt_down_side = int(patch_feats_concat.shape[1] ** 0.5)
                    mask_d = F.adaptive_avg_pool2d(
                        gt.unsqueeze(1).float(), (gt_down_side, gt_down_side)
                    ).squeeze(1).reshape(image.shape[0], -1)
                    anom_idx_flat = (mask_d > 0.5).reshape(-1)
                    all_flat = (patch_feats_concat @ centroids.T).argmax(-1).reshape(-1)
                    if anom_idx_flat.any():
                        epoch_pseudo_labels.append(all_flat[anom_idx_flat].detach().cpu())

            # -------------------- local Focal+Dice + per-cluster losses
            loss = 0.0
            cluster_loss = 0.0
            similarity_map_list = []

            for idx, patch_feature in enumerate(patch_features):
                if idx < args.feature_map_layer[0]:
                    continue

                patch_norm = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                # similarity (softmax over 1+K) — shape (B, N_all, 1+K)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_norm, feats)

                # aggregate to binary for Focal+Dice
                sim_patches = similarity[:, 1:, :]           # (B, N_patches, 1+K)
                normal_prob  = sim_patches[:, :, 0:1]        # (B, N_patches, 1)
                anomaly_prob = 1.0 - normal_prob              # ≡ sum of anomaly probs
                sim_binary   = torch.cat([normal_prob, anomaly_prob], dim=-1)  # (B, N_patches, 2)

                sim_map = AnomalyCLIP_lib.get_similarity_map(
                    sim_binary, args.image_size
                ).permute(0, 3, 1, 2)                        # (B, 2, H, W)
                similarity_map_list.append(sim_map)

                # per-cluster loss for K>1
                if centroids is not None:
                    patch_only = patch_norm[:, 1:, :]        # (B, N_patches, D)
                    patch_logits = patch_only @ feats.T       # (B, N_patches, 1+K)  raw cosine
                    cluster_loss += cluster_local_loss(
                        patch_logits, gt, patch_feats_concat, centroids
                    )

            # Focal + Dice (same formulation as original AnomalyCLIP)
            for sim_map in similarity_map_list:
                loss += loss_focal(sim_map, gt)
                loss += loss_dice(sim_map[:, 1, :, :], gt)
                loss += loss_dice(sim_map[:, 0, :, :], 1 - gt)
            loss = lam * loss

            # prototype repulsion
            repulsion = 0.0
            if args.cdmp_K > 1:
                t_anom = feats[1:]   # (K, D) already normalised
                repulsion = prototype_repulsion_loss(t_anom, margin=args.repulsion_margin)

            total_loss = (loss + image_loss
                          + args.cluster_weight  * cluster_loss
                          + args.repulsion_weight * repulsion)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # ----------------------------------------------------------------- logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                "epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}".format(
                    epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)
                )
            )
            if args.cdmp_K > 1:
                with torch.no_grad():
                    t_anom = feats[1:]
                log_prototype_stats(t_anom, logger=logger,
                                    prefix=f"[epoch {epoch+1}] ")
                if epoch_pseudo_labels:
                    all_pl = torch.cat(epoch_pseudo_labels)
                    log_activation_counts(all_pl, args.cdmp_K, logger=logger,
                                          prefix=f"[epoch {epoch+1}] ")

        # --------------------------------------------------------------- save
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CDMP train")
    parser.add_argument("--train_data_path", type=str, default="./data/mvtecdataset")
    parser.add_argument("--save_path",       type=str, default="./checkpoints/cdmp")
    parser.add_argument("--dataset",         type=str, default="mvtec")

    # CLIP / prompt hyperparams — keep at AnomalyCLIP defaults for comparability
    parser.add_argument("--depth",        type=int, default=9)
    parser.add_argument("--n_ctx",        type=int, default=12)
    parser.add_argument("--t_n_ctx",      type=int, default=4)
    parser.add_argument("--features_list",     type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image_size",   type=int, default=518)

    # training
    parser.add_argument("--epoch",         type=int,   default=15)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--print_freq",    type=int,   default=1)
    parser.add_argument("--save_freq",     type=int,   default=1)
    parser.add_argument("--seed",          type=int,   default=111)

    # CDMP-specific
    parser.add_argument("--cdmp_K",           type=int,   default=1,
                        help="Number of abnormal prototypes. 1 = original AnomalyCLIP.")
    parser.add_argument("--centroid_path",    type=str,   default="",
                        help="Path to cluster_centroids.pt (required when cdmp_K > 1).")
    parser.add_argument("--cluster_weight",   type=float, default=1.0,
                        help="Weight for per-cluster contrastive loss.")
    parser.add_argument("--repulsion_weight", type=float, default=0.5,
                        help="Weight for prototype repulsion loss.")
    parser.add_argument("--repulsion_margin", type=float, default=-0.3,
                        help="Target upper bound on pairwise prototype cosine similarity.")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
