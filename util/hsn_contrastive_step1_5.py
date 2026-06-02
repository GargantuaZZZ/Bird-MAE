#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HSN / Bird-MAE contrastive diagnosis pipeline.

This script implements Step 1-5:
1) Per-class AP/AUROC/T1 change analysis.
2) Pairwise score leakage / confusion analysis.
3) Embedding visualization and class-separability diagnostics.
4) Positive-pair quality and annotation-distribution analysis.
5) Strategy recommendation for contrastive learning variants.

Inputs:
    --vit_npz      npz saved from the plain VIT model, containing preds, targets, features.
    --supcon_npz   npz saved from the VIT_Contrastive model, containing preds, targets, features, optional proj.
    --annotations_csv optional BirdSet/HSN annotation CSV with columns:
        Filename, Start Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Species eBird Code
    --class_names optional txt/csv class name file. If omitted, class_0...class_C-1 are used.

Example:
python tools/hsn_contrastive_step1_5.py \
  --vit_npz outputs_vit.npz \
  --supcon_npz outputs_supcon.npz \
  --annotations_csv "annotations (4).csv" \
  --out_dir analysis_hsn_vit_vs_supcon
"""

import argparse
import itertools
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def sigmoid_if_needed(x):
    x = np.asarray(x)
    if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
        return 1.0 / (1.0 + np.exp(-x))
    return x


def load_class_names(path, num_classes):
    if path is None:
        return [f"class_{i}" for i in range(num_classes)]
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        for col in ["species", "Species eBird Code", "ebird_code", "label", "name"]:
            if col in df.columns:
                names = df[col].astype(str).tolist()
                break
        else:
            names = df.iloc[:, 0].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported class name file: {path}")

    if len(names) != num_classes:
        print(f"[WARN] class_names length={len(names)} but model num_classes={num_classes}. "
              f"Using first {num_classes} names if possible, otherwise filling class_i.")
        names = (names + [f"class_{i}" for i in range(len(names), num_classes)])[:num_classes]
    return names


def top1_acc_multilabel(preds, targets):
    top = preds.argmax(axis=1)
    return float((targets[np.arange(len(top)), top] == 1).mean())


def per_class_metrics(preds, targets, class_names):
    rows = []
    C = targets.shape[1]
    for c in range(C):
        y = targets[:, c].astype(int)
        p = preds[:, c]
        support = int(y.sum())
        if support == 0:
            ap = np.nan
            auc = np.nan
        else:
            ap = average_precision_score(y, p)
            try:
                auc = roc_auc_score(y, p)
            except ValueError:
                auc = np.nan
        rows.append({
            "class_id": c,
            "species": class_names[c],
            "support": support,
            "AP": ap,
            "AUROC": auc,
        })
    return pd.DataFrame(rows)


def global_metrics(preds, targets):
    aps, aucs = [], []
    for c in range(targets.shape[1]):
        y = targets[:, c]
        p = preds[:, c]
        if y.sum() == 0:
            continue
        aps.append(average_precision_score(y, p))
        try:
            aucs.append(roc_auc_score(y, p))
        except ValueError:
            pass
    return {
        "mAP": float(np.mean(aps)),
        "AUROC": float(np.mean(aucs)),
        "T1_Acc": top1_acc_multilabel(preds, targets),
    }


def step1_per_class_delta(vit, supcon, class_names, out_dir):
    step_dir = os.path.join(out_dir, "step1_per_class_delta")
    ensure_dir(step_dir)

    vit_df = per_class_metrics(vit["preds"], vit["targets"], class_names)
    con_df = per_class_metrics(supcon["preds"], supcon["targets"], class_names)

    cmp = vit_df.merge(
        con_df[["class_id", "AP", "AUROC"]],
        on="class_id",
        suffixes=("_VIT", "_SupCon"),
    )
    cmp["delta_AP"] = cmp["AP_SupCon"] - cmp["AP_VIT"]
    cmp["delta_AUROC"] = cmp["AUROC_SupCon"] - cmp["AUROC_VIT"]
    cmp["relative_delta_AP"] = cmp["delta_AP"] / (cmp["AP_VIT"].abs() + 1e-8)
    cmp = cmp.sort_values("delta_AP")
    cmp.to_csv(os.path.join(step_dir, "per_class_delta.csv"), index=False)

    # Bar plot
    plt.figure(figsize=(10, max(6, 0.35 * len(cmp))))
    plt.barh(cmp["species"], cmp["delta_AP"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Delta AP = SupCon - VIT")
    plt.ylabel("Species")
    plt.title("Step 1: Per-class AP change")
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, "per_class_delta_AP.png"), dpi=300)
    plt.close()

    # Support vs delta
    plt.figure(figsize=(7, 5))
    plt.scatter(cmp["support"], cmp["delta_AP"], s=40)
    for _, r in cmp.iterrows():
        plt.text(r["support"], r["delta_AP"], str(r["species"]), fontsize=7)
    plt.axhline(0, linewidth=1)
    plt.xscale("symlog")
    plt.xlabel("Class support in test targets")
    plt.ylabel("Delta AP")
    plt.title("Support vs. SupCon AP change")
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, "support_vs_delta_AP.png"), dpi=300)
    plt.close()

    return cmp


def score_leakage_matrix(preds, targets, class_names, exclude_true_cooccur=True):
    C = targets.shape[1]
    M = np.zeros((C, C), dtype=float)
    for c in range(C):
        for d in range(C):
            if c == d:
                M[c, d] = np.nan
                continue
            if exclude_true_cooccur:
                idx = (targets[:, c] == 1) & (targets[:, d] == 0)
            else:
                idx = targets[:, c] == 1
            M[c, d] = np.nan if idx.sum() == 0 else float(preds[idx, d].mean())
    return pd.DataFrame(M, index=class_names, columns=class_names)


def step2_pairwise_leakage(vit, supcon, class_names, out_dir):
    step_dir = os.path.join(out_dir, "step2_pairwise_leakage")
    ensure_dir(step_dir)

    M_vit = score_leakage_matrix(vit["preds"], vit["targets"], class_names)
    M_con = score_leakage_matrix(supcon["preds"], supcon["targets"], class_names)
    M_delta = M_con - M_vit

    M_vit.to_csv(os.path.join(step_dir, "score_leakage_vit.csv"))
    M_con.to_csv(os.path.join(step_dir, "score_leakage_supcon.csv"))
    M_delta.to_csv(os.path.join(step_dir, "score_leakage_delta.csv"))

    pairs = []
    C = len(class_names)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            lv, lc = M_vit.iloc[i, j], M_con.iloc[i, j]
            if np.isnan(lv) or np.isnan(lc):
                continue
            pairs.append({
                "true_species": class_names[i],
                "confused_as": class_names[j],
                "leakage_VIT": lv,
                "leakage_SupCon": lc,
                "delta_leakage": lc - lv,
            })
    pair_df = pd.DataFrame(pairs).sort_values("leakage_SupCon", ascending=False)
    pair_df.to_csv(os.path.join(step_dir, "top_confused_pairs_supcon.csv"), index=False)
    pair_df.sort_values("delta_leakage", ascending=False).to_csv(
        os.path.join(step_dir, "pairs_made_worse_by_supcon.csv"), index=False
    )
    pair_df.sort_values("delta_leakage", ascending=True).to_csv(
        os.path.join(step_dir, "pairs_improved_by_supcon.csv"), index=False
    )

    for name, M in [("supcon", M_con), ("delta", M_delta)]:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(M.values, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(C), class_names, rotation=90, fontsize=6)
        plt.yticks(range(C), class_names, fontsize=6)
        plt.title(f"Step 2: Score leakage matrix ({name})")
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, f"score_leakage_{name}_heatmap.png"), dpi=300)
        plt.close()

    return pair_df


def get_single_label_indices(targets):
    return np.where(targets.sum(axis=1) == 1)[0]


def plot_2d_embedding(features, targets, class_names, out_path, title, method="tsne", max_points=3000):
    idx = get_single_label_indices(targets)
    if len(idx) < 10:
        print(f"[WARN] Too few single-label samples for embedding plot: {len(idx)}")
        return

    x = features[idx]
    y = targets[idx].argmax(axis=1)

    if len(idx) > max_points:
        rng = np.random.default_rng(1)
        sub = rng.choice(len(idx), size=max_points, replace=False)
        x = x[sub]
        y = y[sub]

    # Reduce first for speed/stability if needed
    if x.shape[1] > 50:
        x_for_plot = PCA(n_components=50, random_state=1).fit_transform(x)
    else:
        x_for_plot = x

    if method == "tsne":
        emb = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(x_for_plot) // 20)),
            init="pca",
            learning_rate="auto",
            random_state=1,
        ).fit_transform(x_for_plot)
    else:
        emb = PCA(n_components=2, random_state=1).fit_transform(x_for_plot)

    plt.figure(figsize=(10, 8))
    for c in np.unique(y):
        pts = emb[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.65, label=class_names[c])
    plt.legend(fontsize=6, ncol=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def class_centroid_distances(features, targets, class_names):
    idx = get_single_label_indices(targets)
    x = features[idx]
    y = targets[idx].argmax(axis=1)
    rows = []
    centroids = {}
    for c in np.unique(y):
        centroids[c] = x[y == c].mean(axis=0)
    for a, b in itertools.combinations(sorted(centroids.keys()), 2):
        dist = np.linalg.norm(centroids[a] - centroids[b])
        rows.append({
            "species_a": class_names[a],
            "species_b": class_names[b],
            "centroid_distance": dist,
        })
    return pd.DataFrame(rows).sort_values("centroid_distance")


def step3_embedding_visualization(vit, supcon, class_names, out_dir):
    step_dir = os.path.join(out_dir, "step3_embedding_visualization")
    ensure_dir(step_dir)

    if "features" in vit:
        plot_2d_embedding(
            vit["features"], vit["targets"], class_names,
            os.path.join(step_dir, "vit_features_tsne.png"),
            "VIT pooled features, single-label samples",
            method="tsne",
        )
        class_centroid_distances(vit["features"], vit["targets"], class_names).to_csv(
            os.path.join(step_dir, "vit_nearest_class_centroids.csv"), index=False
        )

    if "features" in supcon:
        plot_2d_embedding(
            supcon["features"], supcon["targets"], class_names,
            os.path.join(step_dir, "supcon_features_tsne.png"),
            "SupCon model pooled features, single-label samples",
            method="tsne",
        )
        class_centroid_distances(supcon["features"], supcon["targets"], class_names).to_csv(
            os.path.join(step_dir, "supcon_nearest_class_centroids.csv"), index=False
        )

    if "proj" in supcon:
        plot_2d_embedding(
            supcon["proj"], supcon["targets"], class_names,
            os.path.join(step_dir, "supcon_projection_tsne.png"),
            "SupCon projection features, single-label samples",
            method="tsne",
        )
        class_centroid_distances(supcon["proj"], supcon["targets"], class_names).to_csv(
            os.path.join(step_dir, "supcon_projection_nearest_class_centroids.csv"), index=False
        )


def positive_pair_stats(targets):
    targets = targets.astype(int)
    N, C = targets.shape
    overlap = targets @ targets.T
    pos = overlap > 0
    np.fill_diagonal(pos, False)

    union = ((targets[:, None, :] + targets[None, :, :]) > 0).sum(axis=-1)
    jaccard = overlap / np.maximum(union, 1)
    pos_jaccard = jaccard[pos]

    label_count = targets.sum(axis=1)
    return {
        "num_samples": int(N),
        "num_classes": int(C),
        "positive_pair_ratio": float(pos.sum() / max(N * (N - 1), 1)),
        "num_positive_pairs": int(pos.sum()),
        "mean_positive_jaccard": float(pos_jaccard.mean()) if len(pos_jaccard) else 0.0,
        "median_positive_jaccard": float(np.median(pos_jaccard)) if len(pos_jaccard) else 0.0,
        "single_label_ratio": float((label_count == 1).mean()),
        "multi_label_ratio": float((label_count > 1).mean()),
        "no_call_ratio": float((label_count == 0).mean()),
        "mean_labels_per_sample": float(label_count.mean()),
    }


def annotation_distribution_analysis(path, out_dir):
    step_dir = os.path.join(out_dir, "step4_positive_pair_and_data_distribution")
    ensure_dir(step_dir)

    df = pd.read_csv(path)
    required = ["Filename", "Start Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Species eBird Code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation CSV missing columns: {missing}")

    df["duration"] = df["End Time (s)"] - df["Start Time (s)"]
    df["bandwidth"] = df["High Freq (Hz)"] - df["Low Freq (Hz)"]
    counts = df["Species eBird Code"].value_counts()
    dist = pd.DataFrame({"species": counts.index, "events": counts.values})
    dist["event_ratio"] = dist["events"] / len(df)
    dist["cum_event_ratio"] = dist["event_ratio"].cumsum()
    dist.to_csv(os.path.join(step_dir, "annotation_species_event_distribution.csv"), index=False)

    # 5-second segment statistics
    df["seg5"] = np.floor(df["Start Time (s)"] / 5).astype(int)
    seg = df.groupby(["Filename", "seg5"]).agg(
        species_set=("Species eBird Code", lambda x: sorted(set(x))),
        n_events=("Species eBird Code", "size"),
    ).reset_index()
    seg["n_species"] = seg["species_set"].apply(len)
    seg_summary = pd.DataFrame([{
        "num_files": df["Filename"].nunique(),
        "num_events": len(df),
        "num_species": counts.size,
        "num_annotated_5s_segments": len(seg),
        "multi_event_segment_ratio": float((seg["n_events"] > 1).mean()),
        "multi_species_segment_ratio": float((seg["n_species"] > 1).mean()),
        "top1_event_ratio": float(counts.iloc[0] / len(df)),
        "top3_event_ratio": float(counts.iloc[:3].sum() / len(df)),
        "top5_event_ratio": float(counts.iloc[:5].sum() / len(df)),
        "bottom10_event_ratio": float(counts.iloc[-10:].sum() / len(df)) if len(counts) >= 10 else np.nan,
        "mean_duration": float(df["duration"].mean()),
        "median_duration": float(df["duration"].median()),
        "mean_low_freq": float(df["Low Freq (Hz)"].mean()),
        "mean_high_freq": float(df["High Freq (Hz)"].mean()),
    }])
    seg_summary.to_csv(os.path.join(step_dir, "annotation_summary.csv"), index=False)

    # Co-occurrence pairs in 5s segments
    pair_counter = Counter()
    for sp_list in seg["species_set"]:
        for a, b in itertools.combinations(sp_list, 2):
            pair_counter[(a, b)] += 1
    co_rows = [{"species_a": a, "species_b": b, "cooccur_5s_segments": n}
               for (a, b), n in pair_counter.most_common()]
    pd.DataFrame(co_rows).to_csv(os.path.join(step_dir, "annotation_top_5s_cooccurrence_pairs.csv"), index=False)

    # Temporal overlaps
    def ov(a1, a2, b1, b2):
        return max(0.0, min(a2, b2) - max(a1, b1))
    temporal_counter = Counter()
    total_temporal_pairs = 0
    for _, g in df.groupby("Filename"):
        arr = g[["Start Time (s)", "End Time (s)", "Species eBird Code"]].values
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                overlap = ov(arr[i][0], arr[i][1], arr[j][0], arr[j][1])
                if overlap > 0:
                    total_temporal_pairs += 1
                    a, b = sorted([arr[i][2], arr[j][2]])
                    if a != b:
                        temporal_counter[(a, b)] += 1
    temp_rows = [{"species_a": a, "species_b": b, "temporal_overlap_pairs": n}
                 for (a, b), n in temporal_counter.most_common()]
    pd.DataFrame(temp_rows).to_csv(os.path.join(step_dir, "annotation_temporal_overlap_pairs.csv"), index=False)

    # Frequency ranges
    freq = df.groupby("Species eBird Code").agg(
        events=("Species eBird Code", "size"),
        files=("Filename", "nunique"),
        low_mean=("Low Freq (Hz)", "mean"),
        high_mean=("High Freq (Hz)", "mean"),
        low_median=("Low Freq (Hz)", "median"),
        high_median=("High Freq (Hz)", "median"),
        low_q10=("Low Freq (Hz)", lambda x: np.quantile(x, 0.1)),
        high_q90=("High Freq (Hz)", lambda x: np.quantile(x, 0.9)),
        duration_mean=("duration", "mean"),
        bandwidth_mean=("bandwidth", "mean"),
    ).sort_values("events", ascending=False)
    freq.to_csv(os.path.join(step_dir, "annotation_species_frequency_stats.csv"))

    # Frequency-overlap pairs by q10-q90 range
    rows = []
    for a, b in itertools.combinations(freq.index, 2):
        ar, br = freq.loc[a], freq.loc[b]
        inter = max(0.0, min(ar["high_q90"], br["high_q90"]) - max(ar["low_q10"], br["low_q10"]))
        union = max(ar["high_q90"], br["high_q90"]) - min(ar["low_q10"], br["low_q10"])
        ratio = inter / union if union > 0 else 0.0
        rows.append({
            "species_a": a,
            "species_b": b,
            "freq_overlap_ratio_q10_q90": ratio,
            "freq_overlap_hz": inter,
            "events_a": int(ar["events"]),
            "events_b": int(br["events"]),
        })
    pd.DataFrame(rows).sort_values("freq_overlap_ratio_q10_q90", ascending=False).to_csv(
        os.path.join(step_dir, "annotation_frequency_overlap_pairs.csv"), index=False
    )

    return seg_summary, dist


def step4_positive_pair_and_distribution(vit, out_dir, annotations_csv=None):
    step_dir = os.path.join(out_dir, "step4_positive_pair_and_data_distribution")
    ensure_dir(step_dir)

    stats = positive_pair_stats(vit["targets"])
    pd.DataFrame([stats]).to_csv(os.path.join(step_dir, "positive_pair_quality_from_model_targets.csv"), index=False)

    if annotations_csv is not None:
        annotation_distribution_analysis(annotations_csv, out_dir)

    return stats


def step5_strategy_recommendation(out_dir, positive_stats, annotation_summary=None):
    step_dir = os.path.join(out_dir, "step5_strategy_recommendation")
    ensure_dir(step_dir)

    recs = []
    recs.append("# Step 5: Contrastive strategy recommendations\n")

    recs.append("## Current failure hypothesis\n")
    recs.append("- If mAP improves slightly while T1-Acc drops, the contrastive branch may improve ranking but hurt the top-confident class decision.\n")
    recs.append("- If positive-pair Jaccard is low, label-sharing positives are noisy in multi-label soundscapes.\n")
    recs.append("- If class imbalance is severe, majority classes dominate positive pairs and rare classes receive weak contrastive supervision.\n\n")

    recs.append("## Recommended experiment grid\n")
    recs.append("| Variant | Loss object | Positive definition | Suggested YAML |\n")
    recs.append("|---|---|---|---|\n")
    recs.append("| baseline | none | none | `name: VIT` |\n")
    recs.append("| current | clip-level SupCon | shared label | `type: multilabel, weight: 0.05/0.1` |\n")
    recs.append("| safer multi-label | clip-level weighted SupCon | Jaccard label overlap | `type: jaccard, weight: 0.05` |\n")
    recs.append("| low-noise positive | clip-level SupCon | only single-label samples | `type: single_label, weight: 0.05` |\n")
    recs.append("| event-oriented | patch/prototype SupCon | top-k event patches | `VIT_ppnet_Contrastive, type: jaccard, topk: 16` |\n")
    recs.append("| two-view | NT-Xent | same recording under two augmentations | requires dataloader/two-view augmentation |\n\n")

    recs.append("## Automatic recommendation from positive-pair statistics\n")
    recs.append(f"- positive_pair_ratio = {positive_stats.get('positive_pair_ratio', np.nan):.4f}\n")
    recs.append(f"- mean_positive_jaccard = {positive_stats.get('mean_positive_jaccard', np.nan):.4f}\n")
    recs.append(f"- multi_label_ratio = {positive_stats.get('multi_label_ratio', np.nan):.4f}\n")
    recs.append(f"- no_call_ratio = {positive_stats.get('no_call_ratio', np.nan):.4f}\n")

    if positive_stats.get("mean_positive_jaccard", 1) < 0.5:
        recs.append("\n**Recommendation:** use Jaccard-weighted SupCon, because many current positive pairs share only a small fraction of labels.\n")
    if positive_stats.get("multi_label_ratio", 0) > 0.1:
        recs.append("\n**Recommendation:** test single-label-only SupCon to avoid multi-label positive-pair pollution.\n")
    if positive_stats.get("no_call_ratio", 0) > 0.1:
        recs.append("\n**Recommendation:** do not use no-call clips as positive pairs. Use them for classification/no-call auxiliary losses instead.\n")

    recs.append("\n## Minimal YAML settings\n")
    recs.append("```yaml\n")
    recs.append("contrastive:\n")
    recs.append("  weight: 0.05\n")
    recs.append("  temperature: 0.07\n")
    recs.append("  out_dim: 128\n")
    recs.append("  hidden_dim: 512\n")
    recs.append("  type: jaccard\n")
    recs.append("```\n")

    with open(os.path.join(step_dir, "strategy_recommendations.md"), "w", encoding="utf-8") as f:
        f.write("".join(recs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_npz", required=True, help="Plain VIT model outputs .npz")
    parser.add_argument("--supcon_npz", required=True, help="SupCon model outputs .npz")
    parser.add_argument("--annotations_csv", default=None, help="Optional HSN annotation CSV")
    parser.add_argument("--class_names", default=None, help="Optional class names txt/csv")
    parser.add_argument("--out_dir", default="analysis_hsn_vit_vs_supcon")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    vit = load_npz(args.vit_npz)
    supcon = load_npz(args.supcon_npz)

    vit["preds"] = sigmoid_if_needed(vit["preds"])
    supcon["preds"] = sigmoid_if_needed(supcon["preds"])

    class_names = load_class_names(args.class_names, vit["targets"].shape[1])

    # Overall summary
    summary = pd.DataFrame([
        {"model": "VIT", **global_metrics(vit["preds"], vit["targets"])},
        {"model": "SupCon", **global_metrics(supcon["preds"], supcon["targets"])},
    ])
    summary.to_csv(os.path.join(args.out_dir, "00_overall_summary.csv"), index=False)
    print("\n[Overall summary]")
    print(summary)

    print("\n[Step 1] Per-class AP/AUROC delta")
    step1_per_class_delta(vit, supcon, class_names, args.out_dir)

    print("\n[Step 2] Pairwise score leakage")
    step2_pairwise_leakage(vit, supcon, class_names, args.out_dir)

    print("\n[Step 3] Embedding visualization")
    step3_embedding_visualization(vit, supcon, class_names, args.out_dir)

    print("\n[Step 4] Positive-pair quality and annotation distribution")
    pos_stats = step4_positive_pair_and_distribution(vit, args.out_dir, args.annotations_csv)

    print("\n[Step 5] Strategy recommendation")
    step5_strategy_recommendation(args.out_dir, pos_stats)

    print(f"\nDone. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
