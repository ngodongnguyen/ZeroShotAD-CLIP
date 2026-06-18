"""CDMP evaluation script.

Drop-in replacement for test.py supporting K independent abnormal prototypes.
K=1 reproduces original AnomalyCLIP inference exactly.
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import AnomalyCLIP_lib
from dataset import Dataset
from logger import get_logger
from metrics import image_level_metrics, pixel_level_metrics
from utils import get_transform
from cdmp.prompt_learner import CDMPPromptLearner


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(args):
    img_size    = args.image_size
    logger      = get_logger(args.save_path)
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device,
                                     design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess,
                        target_transform=target_transform,
                        dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}
        metrics[obj] = {"pixel-auroc": 0, "pixel-aupro": 0,
                        "image-auroc": 0, "image-ap": 0}

    # -------------------------------------------------- load prompt learner
    prompt_learner = CDMPPromptLearner(model.to("cpu"), AnomalyCLIP_parameters,
                                       cdmp_K=args.cdmp_K)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompt_learner.eval()
    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner()
        feats = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)   # (1+K, D)

    logger.info(f"CDMP K={args.cdmp_K}, checkpoint={args.checkpoint_path}")

    # ------------------------------------------------------------------ eval
    for items in tqdm(test_dataloader):
        image    = items["img"].to(device)
        cls_name = items["cls_name"]
        gt_mask  = items["img_mask"]
        gt_mask[gt_mask > 0.5]  = 1
        gt_mask[gt_mask <= 0.5] = 0
        results[cls_name[0]]["imgs_masks"].append(gt_mask)
        results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                image, args.features_list, DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # image-level anomaly score: max over K anomaly logits → softmax → p(anom)
            img_sim = image_features @ feats.T / 0.07          # (B, 1+K)
            if args.cdmp_K == 1:
                img_logits = img_sim                            # (B, 2) — identical to original
            else:
                img_logits = torch.stack(
                    [img_sim[:, 0], img_sim[:, 1:].mean(dim=-1)], dim=-1
                )
            text_probs = img_logits.softmax(dim=-1)[:, 1]      # scalar anomaly score

            # pixel-level anomaly maps
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx < args.feature_map_layer[0]:
                    continue

                patch_norm = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_norm, feats)
                # binary aggregate: normal_prob vs 1-normal_prob
                sim_patches = similarity[:, 1:, :]              # (B, N_patches, 1+K)
                normal_prob = sim_patches[:, :, 0:1]            # (B, N_patches, 1)
                anomaly_raw = 1.0 - normal_prob                 # (B, N_patches, 1)
                sim_binary  = torch.cat([normal_prob, anomaly_raw], dim=-1)  # (B, N_patches, 2)

                sim_map = AnomalyCLIP_lib.get_similarity_map(sim_binary, img_size)  # (B, H, W, 2)
                # same formula as original: (anom + 1 - normal) / 2
                anomaly_map = (sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            results[cls_name[0]]["pr_sp"].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([
                torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
                for i in anomaly_map.detach().cpu()
            ])
            results[cls_name[0]]["anomaly_maps"].append(anomaly_map)

    # ---------------------------------------------------------------- metrics
    table_ls          = []
    image_auroc_list  = []
    image_ap_list     = []
    pixel_auroc_list  = []
    pixel_aupro_list  = []

    for obj in obj_list:
        table = [obj]
        results[obj]["imgs_masks"]   = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()

        if args.metrics in ("image-level", "image-pixel-level"):
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap    = image_level_metrics(results, obj, "image-ap")
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        if args.metrics in ("pixel-level", "image-pixel-level"):
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)

        if args.metrics == "image-level":
            table += [f"{image_auroc*100:.1f}", f"{image_ap*100:.1f}"]
        elif args.metrics == "pixel-level":
            table += [f"{pixel_auroc*100:.1f}", f"{pixel_aupro*100:.1f}"]
        elif args.metrics == "image-pixel-level":
            table += [f"{pixel_auroc*100:.1f}", f"{pixel_aupro*100:.1f}",
                      f"{image_auroc*100:.1f}", f"{image_ap*100:.1f}"]
        table_ls.append(table)

    def _mean(lst): return np.round(np.mean(lst) * 100, 1) if lst else 0.0

    if args.metrics == "image-level":
        table_ls.append(["mean", _mean(image_auroc_list), _mean(image_ap_list)])
        result_str = tabulate(table_ls, headers=["objects", "image_auroc", "image_ap"],
                              tablefmt="pipe")
    elif args.metrics == "pixel-level":
        table_ls.append(["mean", _mean(pixel_auroc_list), _mean(pixel_aupro_list)])
        result_str = tabulate(table_ls,
                              headers=["objects", "pixel_auroc", "pixel_aupro"],
                              tablefmt="pipe")
    else:  # image-pixel-level
        table_ls.append(["mean", _mean(pixel_auroc_list), _mean(pixel_aupro_list),
                         _mean(image_auroc_list), _mean(image_ap_list)])
        result_str = tabulate(table_ls,
                              headers=["objects", "pixel_auroc", "pixel_aupro",
                                       "image_auroc", "image_ap"],
                              tablefmt="pipe")
    logger.info("\n%s", result_str)

    # -------------------------------------------------- JSON output for sweep
    if getattr(args, "metrics_out", None):
        out = {
            "cdmp_K":      args.cdmp_K,
            "pixel_auroc": float(_mean(pixel_auroc_list))  if pixel_auroc_list  else None,
            "pixel_aupro": float(_mean(pixel_aupro_list))  if pixel_aupro_list  else None,
            "image_auroc": float(_mean(image_auroc_list))  if image_auroc_list  else None,
            "image_ap":    float(_mean(image_ap_list))     if image_ap_list     else None,
        }
        vals = [v for v in (out["pixel_auroc"], out["pixel_aupro"],
                             out["image_auroc"], out["image_ap"]) if v is not None]
        out["mean"] = float(np.round(np.mean(vals), 2)) if vals else None
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"metrics written → {args.metrics_out}: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CDMP test")
    parser.add_argument("--data_path",       type=str, default="./data/visa")
    parser.add_argument("--save_path",       type=str, default="./results/cdmp")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset",         type=str, default="visa")

    parser.add_argument("--depth",            type=int, default=9)
    parser.add_argument("--n_ctx",            type=int, default=12)
    parser.add_argument("--t_n_ctx",          type=int, default=4)
    parser.add_argument("--features_list",    type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer",type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image_size",       type=int, default=518)

    parser.add_argument("--metrics",     type=str, default="image-pixel-level")
    parser.add_argument("--metrics_out", type=str, default="",
                        help="Path to write JSON summary (used by run_sweep.py).")
    parser.add_argument("--sigma",       type=int, default=4)
    parser.add_argument("--seed",        type=int, default=111)

    # CDMP
    parser.add_argument("--cdmp_K", type=int, default=1,
                        help="Must match the K used during training.")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
