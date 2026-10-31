"""
Multimodal Pattern Recognition Benchmark & Model Registry
==========================================================
Comprehensive benchmark and model serialization across three pattern recognition paradigms:
1. Subspace Generative Bayesian Family (GaussianNB, LDA Shrinkage, Regularized QDA)
2. Early Feature-Level Fusion Family (SVM-Linear, SVM-RBF, Random Forest, Gradient Boosting)
3. Late Decision-Level Fusion Family (Stacking Meta-LR, Soft Voting, Product of Experts)
4. Baselines (Heuristic Weighted Fusion, Deep DeceptionMLP)

Generates:
- Dual-protocol evaluation: Official Holdout (109 Train / 12 Test) + 5-Fold Stratified Cross-Validation
- Modality Ablation Study across unimodal, bimodal, and trimodal representations
- Serialized Model Registry (outputs/trained_models.joblib) for real-time inference
- Diagnostic Visualizations (outputs/plots/):
    * subspace_projection.png
    * roc_curves.png
    * confusion_matrices.png
    * feature_importance.png
- Metrics CSVs (outputs/benchmark_metrics.csv, outputs/modality_ablation.csv)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from pr_pipeline import (
    ALL_FEATURE_COLS,
    DOMAIN_FEATURE_COLS,
    MultimodalPreprocessor,
    extract_modality_data,
    load_and_bundle_dataset,
)
from pr_solutions import (
    EarlyFusionDiscriminativeBenchmark,
    EvaluationMetrics,
    GenerativeBayesianBenchmark,
    LateFusionStackingClassifier,
    compute_metrics,
    evaluate_baseline_heuristic,
)
from telemetry import prewarm_telemetry_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Modern sleek styling for diagnostic plots
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "font.family": "sans-serif",
})

COLOR_TRUTH = "#2563eb"     # Royal blue
COLOR_LIE = "#dc2626"       # Crimson red
COLOR_ACCENT = "#10b981"    # Emerald green
COLOR_ORANGE = "#f59e0b"    # Amber orange
COLOR_PURPLE = "#8b5cf6"    # Purple


def run_5fold_cross_validation(
    X_all_full: np.ndarray,
    y_all: np.ndarray,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Executes 5-fold stratified cross validation with strict fold-level preprocessing.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_fold_metrics: dict[str, list[EvaluationMetrics]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all_full, y_all)):
        X_tr_raw, y_tr = X_all_full[train_idx], y_all[train_idx]
        X_te_raw, y_te = X_all_full[test_idx], y_all[test_idx]

        # Fit preprocessor strictly on training fold
        prep = MultimodalPreprocessor(pca_text_components=10, random_state=42)
        prep.fit(X_tr_raw, y_tr)
        X_tr_compact = prep.transform(X_tr_raw)
        X_te_compact = prep.transform(X_te_raw)

        # Baseline: Heuristic
        m_heur = evaluate_baseline_heuristic(X_te_raw, y_te)
        model_fold_metrics.setdefault("Heuristic-Weighted", []).append(m_heur)

        # Baseline: Deep-DeceptionMLP (5-Fold Stratified CV on standardized features)
        mlp_cv = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=250, random_state=42)
        mlp_cv.fit(X_tr_compact, y_tr)
        prob_mlp = mlp_cv.predict_proba(X_te_compact)[:, 1]
        model_fold_metrics.setdefault("Deep-DeceptionMLP", []).append(compute_metrics(y_te, prob_mlp))

        # Family 1: Subspace Generative Bayesian
        gen_benchmark = GenerativeBayesianBenchmark(random_state=42)
        res_gen = gen_benchmark.fit_and_evaluate(X_tr_compact, y_tr, X_te_compact, y_te)
        for name, m in res_gen.items():
            model_fold_metrics.setdefault(name, []).append(m)

        # Family 2: Early Fusion Margin & Ensembles (including EarlyFusion-ExtraTrees)
        early_benchmark = EarlyFusionDiscriminativeBenchmark(random_state=42)
        res_early = early_benchmark.fit_and_evaluate(X_tr_compact, y_tr, X_te_compact, y_te, tune_svm=False)
        for name, m in res_early.items():
            model_fold_metrics.setdefault(name, []).append(m)

        # Family 3: Late Fusion Stacking
        late_stack = LateFusionStackingClassifier(base_estimator_type="lr", n_splits=5, random_state=42)
        late_stack.fit(X_tr_compact, y_tr)

        prob_stack = late_stack.predict_proba(X_te_compact)[:, 1]
        model_fold_metrics.setdefault("LateFusion-StackingLR", []).append(compute_metrics(y_te, prob_stack))

        prob_soft = late_stack.soft_vote_predict_proba(X_te_compact)[:, 1]
        model_fold_metrics.setdefault("LateFusion-SoftVoting", []).append(compute_metrics(y_te, prob_soft))

        prob_poe = late_stack.product_of_experts_predict_proba(X_te_compact)[:, 1]
        model_fold_metrics.setdefault("LateFusion-BayesianPoE", []).append(compute_metrics(y_te, prob_poe))

    # Summarize mean +- std
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for name, metrics_list in model_fold_metrics.items():
        summary[name] = {
            "accuracy": (
                float(np.mean([m.accuracy for m in metrics_list])),
                float(np.std([m.accuracy for m in metrics_list])),
            ),
            "precision": (
                float(np.mean([m.precision for m in metrics_list])),
                float(np.std([m.precision for m in metrics_list])),
            ),
            "recall": (
                float(np.mean([m.recall for m in metrics_list])),
                float(np.std([m.recall for m in metrics_list])),
            ),
            "f1": (
                float(np.mean([m.f1 for m in metrics_list])),
                float(np.std([m.f1 for m in metrics_list])),
            ),
            "roc_auc": (
                float(np.mean([m.roc_auc for m in metrics_list])),
                float(np.std([m.roc_auc for m in metrics_list])),
            ),
        }
    return summary


def run_modality_ablation(X_all_full: np.ndarray, y_all: np.ndarray) -> pd.DataFrame:
    """Evaluates unimodal, bimodal, and trimodal configurations under 5-Fold Stratified CV."""
    modalities = [
        ("Text-only", "text"),
        ("Audio-only", "audio"),
        ("Vision-only", "vision"),
        ("Audio + Vision", "bimodal_av"),
        ("Text + Audio", "bimodal_ta"),
        ("Text + Vision", "bimodal_tv"),
        ("Trimodal (All)", "all"),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    for label_name, mod_key in modalities:
        accs, f1s, aucs = [], [], []

        for train_idx, test_idx in skf.split(X_all_full, y_all):
            X_tr_raw, y_tr = X_all_full[train_idx], y_all[train_idx]
            X_te_raw, y_te = X_all_full[test_idx], y_all[test_idx]

            prep = MultimodalPreprocessor(pca_text_components=10, random_state=42)
            prep.fit(X_tr_raw, y_tr)
            X_tr_compact = prep.transform(X_tr_raw)
            X_te_compact = prep.transform(X_te_raw)

            X_tr_sub = extract_modality_data(X_tr_compact, mod_key, mode="compact")
            X_te_sub = extract_modality_data(X_te_compact, mod_key, mode="compact")

            clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)
            clf.fit(X_tr_sub, y_tr)
            prob = clf.predict_proba(X_te_sub)[:, 1]
            m = compute_metrics(y_te, prob)

            accs.append(m.accuracy)
            f1s.append(m.f1)
            aucs.append(m.roc_auc)

        records.append({
            "Configuration": label_name,
            "Accuracy_Mean": float(np.mean(accs)),
            "Accuracy_Std": float(np.std(accs)),
            "F1_Mean": float(np.mean(f1s)),
            "F1_Std": float(np.std(f1s)),
            "ROC_AUC_Mean": float(np.mean(aucs)),
            "ROC_AUC_Std": float(np.std(aucs)),
        })

    return pd.DataFrame(records)


def generate_subspace_projection_plot(
    X_compact: np.ndarray,
    y: np.ndarray,
    save_path: Path,
) -> None:
    """Generates 2D PCA & Fisher's LDA scatter projections."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.patch.set_facecolor('#0b0f19')
    ax1.set_facecolor('#131b2e')
    ax2.set_facecolor('#131b2e')

    # 1. 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_compact)
    var_exp = pca_2d.explained_variance_ratio_

    for label, color, name in [(0, COLOR_TRUTH, "Truthful (y=0)"), (1, COLOR_LIE, "Deceptive (y=1)")]:
        mask = (y == label)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.85, edgecolors="#1e293b", s=55)

    ax1.set_title(f"Unsupervised PCA Subspace\n(PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}% Variance)", color='#f8fafc', fontweight='bold')
    ax1.set_xlabel("Principal Component 1", color='#cbd5e1')
    ax1.set_ylabel("Principal Component 2", color='#cbd5e1')
    ax1.tick_params(colors='#94a3b8')
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    ax1.grid(True, linestyle="--", color='#334155', alpha=0.5)
    ax1.legend(loc="upper right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

    # 2. Fisher's Linear Discriminant Analysis Projection
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_compact, y)

    # Orientation check: ensure deceptive cluster is on the right (> 0)
    if np.mean(X_lda[y == 1]) < np.mean(X_lda[y == 0]):
        X_lda = -1.0 * X_lda

    rng = np.random.RandomState(42)
    jitter = rng.normal(0, 0.08, size=len(X_lda))

    for label, color, name in [(0, COLOR_TRUTH, "Truthful (y=0)"), (1, COLOR_LIE, "Deceptive (y=1)")]:
        mask = (y == label)
        ax2.scatter(X_lda[mask, 0], jitter[mask], c=color, label=name, alpha=0.85, edgecolors="#1e293b", s=55)

    ax2.axvline(x=0.0, color="#f8fafc", linestyle="--", lw=1.8, label="Optimal Fisher Boundary")
    ax2.set_title("Supervised Fisher's LDA Subspace\n(Maximum Between-to-Within Scatter Separation)", color='#f8fafc', fontweight='bold')
    ax2.set_xlabel("Fisher Canonical Projection (w^T x)", color='#cbd5e1')
    ax2.set_yticks([])
    ax2.set_ylabel("Density / Jitter Spread", color='#cbd5e1')
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    ax2.grid(True, linestyle="--", color='#334155', alpha=0.5)
    ax2.legend(loc="upper right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    logger.info("Saved Subspace Projection to %s", save_path)


def generate_roc_curves_plot(
    roc_curves_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    save_path: Path,
) -> None:
    """Generates overlaid ROC curves comparing baselines and pattern recognition models."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#131b2e')

    palette = [
        ("#94a3b8", ":", 1.8),   # Heuristic
        ("#ec4899", "--", 2.0),  # DeceptionMLP
        (COLOR_TRUTH, "-", 2.2), # Subspace-LDA
        (COLOR_ACCENT, "-", 2.2), # EarlyFusion-SVM-Linear
        (COLOR_LIE, "-", 2.5),   # LateFusion-StackingLR
    ]

    for (name, (fpr, tpr, roc_auc)), (col, lstyle, lwidth) in zip(roc_curves_data.items(), palette):
        ax.plot(fpr, tpr, color=col, linestyle=lstyle, lw=lwidth, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", lw=1.2, label="Chance Level (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color='#cbd5e1')
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", color='#cbd5e1')
    ax.set_title("ROC Curves Comparison Across Classification Paradigms", color='#f8fafc', fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, linestyle="--", color='#334155', alpha=0.5)
    ax.legend(loc="lower right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    logger.info("Saved ROC Curves to %s", save_path)


def generate_confusion_matrices_plot(
    cms_data: list[tuple[str, np.ndarray]],
    save_path: Path,
) -> None:
    """Generates 1x3 grid of confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    fig.patch.set_facecolor('#0b0f19')
    labels = ["Truthful", "Deceptive"]

    for ax, (title, cm) in zip(axes, cms_data):
        ax.set_facecolor('#131b2e')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(title, fontsize=11, fontweight="bold", color='#f8fafc')
        ax.set_xlabel("Predicted Label", color='#cbd5e1')
        ax.set_ylabel("True Label", color='#cbd5e1')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    logger.info("Saved Confusion Matrices to %s", save_path)


def generate_feature_importance_plot(
    domain_feature_names: list[str],
    rf_feature_importances: np.ndarray,
    learned_weights: np.ndarray,
    heuristic_weights: tuple[float, float, float],
    save_path: Path,
) -> None:
    """Generates feature importances & learned late-fusion weights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('#0b0f19')
    ax1.set_facecolor('#131b2e')
    ax2.set_facecolor('#131b2e')

    # 1. Feature importances of 12 domain-engineered features
    sorted_idx = np.argsort(rf_feature_importances)
    ax1.barh(
        range(len(sorted_idx)),
        rf_feature_importances[sorted_idx],
        color=COLOR_TRUTH,
        edgecolor="#38bdf8",
        alpha=0.85,
    )
    ax1.set_yticks(range(len(sorted_idx)))
    ax1.set_yticklabels([domain_feature_names[i] for i in sorted_idx], color='#cbd5e1')
    ax1.set_xlabel("Random Forest Gini Feature Importance", color='#cbd5e1')
    ax1.set_title("Domain Feature Importance (12 Engineered Signals)", color='#f8fafc', fontweight='bold')
    ax1.tick_params(colors='#94a3b8')
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    ax1.grid(axis="x", linestyle="--", color='#334155', alpha=0.5)

    # 2. Modality weights comparison: Learned vs Heuristic
    modalities = ["Text", "Audio", "Vision"]
    norm_learned = np.abs(learned_weights) / np.sum(np.abs(learned_weights))

    x = np.arange(len(modalities))
    width = 0.35

    ax2.bar(x - width/2, heuristic_weights, width, label="Heuristic Baseline (0.3/0.3/0.4)", color="#64748b", edgecolor="#94a3b8")
    ax2.bar(x + width/2, norm_learned, width, label="Learned Late-Fusion Meta-Weights", color=COLOR_ORANGE, edgecolor="#fbbf24")

    ax2.set_xticks(x)
    ax2.set_xticklabels(modalities, fontweight="bold", color='#f8fafc')
    ax2.set_ylabel("Normalized Modality Contribution", color='#cbd5e1')
    ax2.set_ylim(0, 0.6)
    ax2.set_title("Modality Fusion Weights: Heuristic vs. Data-Driven Meta-Learner", color='#f8fafc', fontweight='bold')
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    ax2.grid(axis="y", linestyle="--", color='#334155', alpha=0.5)
    ax2.legend(loc="upper right", facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    logger.info("Saved Feature and Modality Weights to %s", save_path)


def main() -> None:
    csv_path = Path("outputs/features_full.csv")
    if not csv_path.exists():
        logger.error("Dataset not found at %s. Please run feature extraction first.", csv_path)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("MULTIMODAL PATTERN RECOGNITION BENCHMARK SUITE")
    logger.info("=" * 70)

    # 1. Load Dataset Bundle
    bundle = load_and_bundle_dataset(csv_path, pca_text_components=10)
    logger.info(
        "Dataset loaded: Train=%d, Test=%d, Total=%d",
        len(bundle.df_train), len(bundle.df_test), len(bundle.df_all),
    )

    # Preprocess with Leakage-Free Pipeline
    preprocessor = MultimodalPreprocessor(pca_text_components=10, random_state=42)
    preprocessor.fit(bundle.X_train_full, bundle.y_train)

    X_train_compact = preprocessor.transform(bundle.X_train_full)
    X_test_compact = preprocessor.transform(bundle.X_test_full)
    X_all_compact = preprocessor.transform(bundle.X_all_full)

    logger.info(
        "Subspace compression complete: 396 dims -> %d compact dims",
        X_train_compact.shape[1],
    )

    # 2. Evaluate On Official Holdout Split (109 Train / 12 Test)
    logger.info("--- Evaluating on Official Holdout Split (N_test = %d) ---", len(bundle.y_test))
    holdout_results: dict[str, EvaluationMetrics] = {}

    # Baseline Heuristic
    holdout_results["Heuristic-Weighted"] = evaluate_baseline_heuristic(bundle.X_test_full, bundle.y_test)

    # Baseline DeceptionMLP (if trained checkpoint exists)
    model_pt_path = Path("outputs/model_full.pt")
    if model_pt_path.exists():
        try:
            import torch
            from model import DeceptionMLP
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(model_pt_path, map_location=dev, weights_only=False)
            in_dim = int(ckpt.get("input_dim", bundle.X_test_full.shape[1]))
            mlp_model = DeceptionMLP(input_dim=in_dim).to(dev)
            mlp_model.load_state_dict(ckpt["model_state_dict"])
            mlp_model.eval()
            s_mean = np.array(ckpt.get("scaler_mean", [0.0] * in_dim), dtype=np.float32)
            s_scale = np.array(ckpt.get("scaler_scale", [1.0] * in_dim), dtype=np.float32)
            safe_s = np.where(s_scale == 0.0, 1.0, s_scale)
            X_scaled = (bundle.X_test_full - s_mean) / safe_s
            with torch.no_grad():
                logits = mlp_model(torch.from_numpy(X_scaled).to(dev)).cpu().numpy()
            mlp_prob = 1.0 / (1.0 + np.exp(-logits))
            holdout_results["Deep-DeceptionMLP"] = compute_metrics(bundle.y_test, mlp_prob)
            logger.info("Loaded and evaluated Deep-DeceptionMLP on holdout.")
        except Exception as exc:
            logger.warning("Could not evaluate DeceptionMLP: %s", exc)

    # Family 1: Subspace Generative Bayesian
    gen_benchmark = GenerativeBayesianBenchmark(random_state=42)
    res_gen = gen_benchmark.fit_and_evaluate(X_train_compact, bundle.y_train, X_test_compact, bundle.y_test)
    for k, v in res_gen.items():
        holdout_results[k] = v

    # Family 2: Early Feature Fusion (Margin + Ensembles)
    early_benchmark = EarlyFusionDiscriminativeBenchmark(random_state=42)
    res_early = early_benchmark.fit_and_evaluate(X_train_compact, bundle.y_train, X_test_compact, bundle.y_test, tune_svm=True)
    for k, v in res_early.items():
        holdout_results[k] = v

    # Family 3: Late Decision Fusion (Stacking)
    late_stack = LateFusionStackingClassifier(base_estimator_type="lr", n_splits=5, random_state=42)
    late_stack.fit(X_train_compact, bundle.y_train)

    prob_stack = late_stack.predict_proba(X_test_compact)[:, 1]
    holdout_results["LateFusion-StackingLR"] = compute_metrics(bundle.y_test, prob_stack)

    prob_soft = late_stack.soft_vote_predict_proba(X_test_compact)[:, 1]
    holdout_results["LateFusion-SoftVoting"] = compute_metrics(bundle.y_test, prob_soft)

    prob_poe = late_stack.product_of_experts_predict_proba(X_test_compact)[:, 1]
    holdout_results["LateFusion-BayesianPoE"] = compute_metrics(bundle.y_test, prob_poe)

    # 3. Serialize Trained Models & Preprocessor to outputs/trained_models.joblib
    registry_models: dict[str, any] = {}
    # Subspace PCA for generative models
    pca_sub = gen_benchmark.pca_subspace_
    for k, m in gen_benchmark.fitted_models.items():
        registry_models[k] = {"model": m, "subspace_pca": pca_sub}

    for k, m in early_benchmark.fitted_models.items():
        registry_models[k] = {"model": m, "subspace_pca": None}

    registry_models["LateFusion-StackingLR"] = {"model": late_stack, "type": "late_stack"}
    registry_models["LateFusion-SoftVoting"] = {"model": late_stack, "type": "late_soft"}
    registry_models["LateFusion-BayesianPoE"] = {"model": late_stack, "type": "late_poe"}

    trained_artifact = {
        "preprocessor": preprocessor,
        "models": registry_models,
        "feature_importances": early_benchmark.feature_importances,
        "feature_names_full": bundle.feature_names_full,
        "compact_feature_names": bundle.compact_feature_names,
        "domain_feature_names": DOMAIN_FEATURE_COLS,
    }
    models_joblib_path = OUTPUTS_DIR / "trained_models.joblib"
    joblib.dump(trained_artifact, models_joblib_path)
    logger.info("Serialized trained model registry to %s", models_joblib_path)

    # 4. Evaluate 5-Fold Stratified Cross Validation across all N=121 samples
    logger.info("--- Evaluating 5-Fold Stratified Cross-Validation across All Videos (N = %d) ---", len(bundle.y_all))
    cv_summary = run_5fold_cross_validation(bundle.X_all_full, bundle.y_all)

    # 5. Compile Combined Benchmark Results Table
    table_rows = []
    for model_name, holdout_m in holdout_results.items():
        cv_m = cv_summary.get(model_name, {})
        table_rows.append({
            "Model": model_name,
            "Holdout_Accuracy": holdout_m.accuracy,
            "Holdout_Precision": holdout_m.precision,
            "Holdout_Recall": holdout_m.recall,
            "Holdout_F1": holdout_m.f1,
            "Holdout_ROC_AUC": holdout_m.roc_auc,
            "CV_Accuracy": f"{cv_m.get('accuracy', (0,0))[0]:.3f} ± {cv_m.get('accuracy', (0,0))[1]:.3f}",
            "CV_Precision": f"{cv_m.get('precision', (0,0))[0]:.3f} ± {cv_m.get('precision', (0,0))[1]:.3f}",
            "CV_Recall": f"{cv_m.get('recall', (0,0))[0]:.3f} ± {cv_m.get('recall', (0,0))[1]:.3f}",
            "CV_F1": f"{cv_m.get('f1', (0,0))[0]:.3f} ± {cv_m.get('f1', (0,0))[1]:.3f}",
            "CV_ROC_AUC": f"{cv_m.get('roc_auc', (0,0))[0]:.3f} ± {cv_m.get('roc_auc', (0,0))[1]:.3f}",
        })

    benchmark_df = pd.DataFrame(table_rows)
    benchmark_csv_path = OUTPUTS_DIR / "benchmark_metrics.csv"
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    logger.info("Exported benchmark metrics to %s", benchmark_csv_path)

    # 6. Modality Ablation Study
    logger.info("--- Running Modality Ablation Experiments ---")
    ablation_df = run_modality_ablation(bundle.X_all_full, bundle.y_all)
    ablation_csv_path = OUTPUTS_DIR / "modality_ablation.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)
    logger.info("Exported modality ablation metrics to %s", ablation_csv_path)

    # 7. Generate Diagnostic Plots
    logger.info("--- Generating Modern Diagnostic Plots in outputs/plots/ ---")

    # Plot 1: Subspace Projection (PCA & LDA)
    generate_subspace_projection_plot(X_all_compact, bundle.y_all, PLOTS_DIR / "subspace_projection.png")

    # Plot 2: ROC Curve Comparisons
    roc_dict = {}
    roc_dict["Heuristic-Weighted"] = (
        *roc_curve(bundle.y_test, holdout_results["Heuristic-Weighted"].y_prob)[:2],
        holdout_results["Heuristic-Weighted"].roc_auc,
    )
    if "Deep-DeceptionMLP" in holdout_results:
        roc_dict["Deep-DeceptionMLP"] = (
            *roc_curve(bundle.y_test, holdout_results["Deep-DeceptionMLP"].y_prob)[:2],
            holdout_results["Deep-DeceptionMLP"].roc_auc,
        )
    roc_dict["Subspace-LDA"] = (
        *roc_curve(bundle.y_test, holdout_results["Subspace-LDA"].y_prob)[:2],
        holdout_results["Subspace-LDA"].roc_auc,
    )
    roc_dict["EarlyFusion-SVM-Linear"] = (
        *roc_curve(bundle.y_test, holdout_results["EarlyFusion-SVM-Linear"].y_prob)[:2],
        holdout_results["EarlyFusion-SVM-Linear"].roc_auc,
    )
    roc_dict["LateFusion-StackingLR"] = (
        *roc_curve(bundle.y_test, holdout_results["LateFusion-StackingLR"].y_prob)[:2],
        holdout_results["LateFusion-StackingLR"].roc_auc,
    )
    generate_roc_curves_plot(roc_dict, PLOTS_DIR / "roc_curves.png")

    # Plot 3: Confusion Matrices
    cms = [
        ("Subspace-LDA", confusion_matrix(bundle.y_test, holdout_results["Subspace-LDA"].y_pred)),
        ("EarlyFusion-SVM-Linear", confusion_matrix(bundle.y_test, holdout_results["EarlyFusion-SVM-Linear"].y_pred)),
        ("LateFusion-StackingLR", confusion_matrix(bundle.y_test, holdout_results["LateFusion-StackingLR"].y_pred)),
    ]
    generate_confusion_matrices_plot(cms, PLOTS_DIR / "confusion_matrices.png")

    # Plot 4: Feature & Modality Weights
    domain_indices = [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 20, 21]
    rf_feat_imps = early_benchmark.feature_importances["EarlyFusion-RandomForest"][domain_indices]
    learned_weights = late_stack.learned_weights_ if late_stack.learned_weights_ is not None else np.array([0.33, 0.33, 0.33])
    generate_feature_importance_plot(
        DOMAIN_FEATURE_COLS,
        rf_feat_imps,
        learned_weights,
        (0.3, 0.3, 0.4),
        PLOTS_DIR / "feature_importance.png",
    )

    # 8. Pre-warm Persistent Telemetry Disk Cache (outputs/telemetry_cache.joblib)
    logger.info("--- Pre-warming Persistent Telemetry Disk Cache ---")
    demo_clips = list(Path(".").glob("trial_*.mp4"))
    test_dir1 = Path("Real-life Deception Detection Dataset With Train Test/Test")
    test_dir2 = Path("Test")
    if test_dir1.exists():
        demo_clips.extend(list(test_dir1.glob("*.mp4")))
    if test_dir2.exists():
        demo_clips.extend(list(test_dir2.glob("*.mp4")))
    unique_clips = sorted(list({p.resolve() for p in demo_clips}))
    prewarm_telemetry_cache(unique_clips)

    # Print summary table
    print("\n" + "=" * 80)
    print("PATTERN RECOGNITION BENCHMARK (Holdout Test vs. 5-Fold Stratified CV)")
    print("=" * 80)
    print(benchmark_df[["Model", "Holdout_Accuracy", "Holdout_F1", "Holdout_ROC_AUC", "CV_Accuracy", "CV_F1", "CV_ROC_AUC"]].to_string(index=False))
    print("\n" + "=" * 80)
    print("MODALITY ABLATION SUMMARY (5-Fold Stratified CV)")
    print("=" * 80)
    print(ablation_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
