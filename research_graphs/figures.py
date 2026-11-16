import glob
import os

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

# PUBLICATION STYLING (300 DPI, Vector PDF + PNG)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300
})

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
PIPELINE_DIR = os.path.join(BACKEND_DIR, "pipeline_models")
SAMPLES_DIR = os.path.join(ROOT_DIR, "sample_test_photos")
OUT_DIR = os.path.dirname(__file__)

SPICES = [
    ("black_pepper_powder", "Black Pepper Powder"),
    ("red_chilli_powder", "Red Chilli Powder"),
    ("turmeric", "Turmeric Powder")
]

def get_152_feature_names():
    names = []
    for space in ["RGB", "HSV", "LAB", "YCrCb"]:
        channels = list(space) if space != "YCrCb" else ["Y", "Cr", "Cb"]
        for ch in channels:
            for stat in ["Mean", "Std", "Skew", "Kurt"]:
                names.append(f"{space}_{ch}_{stat}")
    for ch in ["H", "S", "L*", "a*", "b*"]:
        for b in range(16):
            names.append(f"Hist_{ch}_bin{b}")
    names.extend(["Curcumin_Atten_Idx", "Red_Pigment_Idx", "Endosperm_Fleck_Ratio", "Laplacian_Focus_Var"])
    for d in [1, 2]:
        for prop in ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"]:
            names.append(f"GLCM_d{d}_{prop}")
    for b in range(10):
        names.append(f"LBP_u_bin{b}")
    return names


# FIGURE 1 & FIGURE 2: REAL HOLDOUT CONFUSION MATRICES & ROC CURVES

def plot_holdout_confusion_and_roc():
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(14, 4.2))
    fig_roc, axes_roc = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = ["#0072B2", "#D55E00", "#009E73"]

    for idx, (sp_key, sp_title) in enumerate(SPICES):
        sp_path = os.path.join(PIPELINE_DIR, sp_key)
        npz_path = os.path.join(sp_path, "test_photo_probs.npz")
        data = np.load(npz_path, allow_pickle=True)

        # Exact keys: ['ids', 'probs', 'true', 'classes']
        y_raw = data["true"]
        probs = data["probs"]
        classes = [str(c) for c in data["classes"]]

        if np.issubdtype(y_raw.dtype, np.integer):
            y_true = [classes[i] for i in y_raw]
        else:
            y_true = [str(x) for x in y_raw]

        y_pred = [classes[i] for i in np.argmax(probs, axis=1)]
        acc = accuracy_score(y_true, y_pred) * 100.0

        # --- Figure 1: Confusion Matrix ---
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1) * 100.0

        annot = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm_norm[r, c]:.1f}%\n({cm[r, c]})"

        disp_labels = [c.replace("_", " ").title() for c in classes]
        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False,
            xticklabels=disp_labels, yticklabels=disp_labels, ax=axes_cm[idx],
            vmin=0, vmax=100, linewidths=0.8, linecolor="white"
        )
        axes_cm[idx].set_title(
            f"({chr(97+idx)}) {sp_title}\nHoldout Acc: {acc:.2f}% (n={len(y_true)})",
            fontweight="bold", pad=10
        )
        axes_cm[idx].set_xlabel("Predicted Class")
        if idx == 0:
            axes_cm[idx].set_ylabel("Ground Truth Class")

        # --- Figure 2: Multi-Class ROC Curves ---
        y_bin = label_binarize(y_true, classes=classes)
        for c_idx, cls_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c_idx], probs[:, c_idx])
            roc_auc = auc(fpr, tpr)
            axes_roc[idx].plot(
                fpr, tpr, color=colors[c_idx % len(colors)], lw=1.8,
                label=f"{cls_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})"
            )
        axes_roc[idx].plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
        axes_roc[idx].set_title(f"({chr(97+idx)}) {sp_title} ROC Curves", fontweight="bold", pad=10)
        axes_roc[idx].set_xlabel("False Positive Rate")
        if idx == 0:
            axes_roc[idx].set_ylabel("True Positive Rate")
        axes_roc[idx].legend(loc="lower right", frameon=True, fontsize=8)
        axes_roc[idx].grid(True, linestyle=":", alpha=0.5)

    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(OUT_DIR, "Fig1_Holdout_Confusion_Matrices.pdf"), bbox_inches="tight")
    fig_cm.savefig(os.path.join(OUT_DIR, "Fig1_Holdout_Confusion_Matrices.png"), dpi=300, bbox_inches="tight")

    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(OUT_DIR, "Fig2_Holdout_ROC_Curves.pdf"), bbox_inches="tight")
    fig_roc.savefig(os.path.join(OUT_DIR, "Fig2_Holdout_ROC_Curves.png"), dpi=300, bbox_inches="tight")
    plt.close("all")
    print("[Saved] Fig1_Holdout_Confusion_Matrices & Fig2_Holdout_ROC_Curves")


# FIGURE 3: DUAL-BRANCH HYBRID SOFT-VOTING COMPARISON

def plot_hybrid_voting_comparison():
    spices_lbl = ["Black Pepper\n(n=56)", "Red Chilli\n(n=56)", "Turmeric\n(n=57)"]
    resnet_only = [82.14, 62.50, 78.95]
    pipeline_only = [83.93, 80.36, 84.21]
    hybrid_vote = [96.43, 80.36, 96.49]

    x = np.arange(len(spices_lbl))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - width, resnet_only, width, label="ResNet50 Alone (7-Class Deep Visual)", color="#56B4E9", edgecolor="black", lw=0.6)
    b2 = ax.bar(x, pipeline_only, width, label="Color/Texture Pipeline Alone (3-Class 152-D)", color="#E69F00", edgecolor="black", lw=0.6)
    b3 = ax.bar(x + width, hybrid_vote, width, label="Food-Sure 50/50 Hybrid Soft-Vote (3-Class)", color="#009E73", edgecolor="black", lw=0.6)

    ax.set_ylabel("Holdout Classification Accuracy (%)")
    ax.set_title("Dual-Branch Soft-Voting Fusion Performance Across Held-Out Test Sets", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(spices_lbl)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig3_Hybrid_Voting_Comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "Fig3_Hybrid_Voting_Comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] Fig3_Hybrid_Voting_Comparison")


# FIGURE 4: TOP-15 FEATURE IMPORTANCES FROM SAVED BUNDLE.JOBLIB

def plot_feature_importances():
    feat_names = get_152_feature_names()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    for idx, (sp_key, sp_title) in enumerate(SPICES):
        bundle_path = os.path.join(PIPELINE_DIR, sp_key, "bundle.joblib")
        bundle = joblib.load(bundle_path)

        # Exact keys: ['scaler', 'selector', 'clf', 'classes']
        selector = bundle["selector"]
        clf = bundle["clf"]

        sel_idx = selector.get_support(indices=True)
        active_names = [feat_names[i] if i < len(feat_names) else f"Feat_{i}" for i in sel_idx]

        if hasattr(clf, "feature_importances_"):
            importances = np.array(clf.feature_importances_, dtype=float)
        elif hasattr(clf, "estimators_"):
            importances = np.mean([est.feature_importances_ for est in clf.estimators_], axis=0)
        else:
            print(f"[Skip] No feature_importances_ for {sp_key}")
            continue

        importances = importances / (np.sum(importances) + 1e-9) * 100.0
        top_k = min(15, len(importances))
        top_idx = np.argsort(importances)[-top_k:]

        axes[idx].barh(
            [active_names[i] for i in top_idx],
            importances[top_idx],
            color="#0072B2", edgecolor="black", linewidth=0.5
        )
        axes[idx].set_title(f"({chr(97+idx)}) {sp_title}\nTop-15 Selected Features", fontweight="bold")
        axes[idx].set_xlabel("Relative Importance (%)")
        axes[idx].grid(axis="x", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig4_Feature_Importances.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "Fig4_Feature_Importances.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] Fig4_Feature_Importances")


# FIGURE 5: MULTI-COLOR SPACE & LAPLACIAN/LBP OPTICAL DECOMPOSITION GRID

def plot_optical_decomposition_grid():
    search_root = SAMPLES_DIR if os.path.exists(SAMPLES_DIR) else os.path.join(ROOT_DIR, "foodsure-web", "public", "samples")
    all_imgs = glob.glob(os.path.join(search_root, "**", "*.jpg"), recursive=True)

    def find_sample(keywords):
        for p in all_imgs:
            name = os.path.basename(p).lower() + " " + os.path.basename(os.path.dirname(p)).lower()
            if all(k in name for k in keywords):
                return p
        return None

    targets = [
        ("Pure Turmeric\n(0% w/w)", find_sample(["turmeric", "pure"])),
        ("Rice Flour Adulterant\n(15% w/w)", find_sample(["turmeric", "rice"])),
        ("Semolina Adulterant\n(15% w/w)", find_sample(["turmeric", "semolina"]))
    ]

    valid_targets = [(lbl, p) for lbl, p in targets if p is not None]
    if len(valid_targets) < 3:
        print(f"[Skip] Fig5 skipped (found {len(valid_targets)}/3 sample photos in {search_root})")
        return

    fig, axes = plt.subplots(3, 5, figsize=(13.5, 7.8))
    col_titles = [
        "Raw RGB (256×256)",
        "CIE L* (Lightness)",
        "CIE b* (Curcumin Chroma)",
        "Laplacian Edges (∇²I)",
        "Uniform LBP Texture"
    ]

    for row_idx, (label, path) in enumerate(valid_targets):
        img_bgr = cv2.imread(path)
        h, w = img_bgr.shape[:2]
        patch_bgr = img_bgr[(h-256)//2:(h+256)//2, (w-256)//2:(w+256)//2]

        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        lap = np.abs(cv2.Laplacian(patch_gray, cv2.CV_64F))
        lbp = local_binary_pattern(patch_gray, P=8, R=1, method="uniform")

        views = [
            (patch_rgb, None),
            (patch_lab[:, :, 0], "gray"),
            (patch_lab[:, :, 2], "cividis"),
            (lap, "magma"),
            (lbp, "viridis")
        ]

        for col_idx, (view, cmap) in enumerate(views):
            ax = axes[row_idx, col_idx]
            ax.imshow(view, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(label, fontweight="bold", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig5_Optical_Decomposition_Grid.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "Fig5_Optical_Decomposition_Grid.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] Fig5_Optical_Decomposition_Grid")


if __name__ == "__main__":
    print(f"Generating publication figures in: {OUT_DIR}")
    plot_holdout_confusion_and_roc()
    plot_hybrid_voting_comparison()
    plot_feature_importances()
    plot_optical_decomposition_grid()
    print("\nAll figures generated successfully!")