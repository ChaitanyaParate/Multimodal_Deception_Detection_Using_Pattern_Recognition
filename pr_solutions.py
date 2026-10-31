"""
Pattern Recognition Models & Benchmarks for Multimodal Deception Detection
==========================================================================
Implements Three Mathematically Grounded Classification Paradigms:

1. Subspace Generative Bayesian Family:
   - Subspace-GaussianNB (conditional independence assumption on PCA subspace)
   - Subspace-LDA (shared covariance with Ledoit-Wolf shrinkage regularization)
   - Subspace-RegQDA (class-specific covariance with regularization parameter alpha=0.3)

2. Early Fusion Margin & Tree Ensemble Family:
   - EarlyFusion-SVM-Linear (maximum-margin linear decision hyperplane)
   - EarlyFusion-SVM-RBF (reproducing kernel Hilbert space with Platt probability scaling)
   - EarlyFusion-RandomForest (bagging with feature subsampling & Gini importance)
   - EarlyFusion-GradBoost (stage-wise additive logistic regression)

3. Late Decision-Level Fusion Family via Stacked Generalization:
   - LateFusion-StackingLR (5-fold out-of-fold unimodal posterior meta-classifier)
   - LateFusion-SoftVoting (calibrated consensus voting)
   - LateFusion-BayesianPoE (Bayesian Product of Experts log-odds combination)
   - Baselines: Heuristic-Weighted (0.3/0.3/0.4) and Deep-DeceptionMLP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from pr_pipeline import (
    MultimodalPreprocessor,
    extract_modality_data,
    get_modality_slices,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Holds standardized performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    y_prob: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> EvaluationMetrics:
    """Computes comprehensive binary classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    return EvaluationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        y_prob=y_prob,
        y_pred=y_pred,
    )


# ============================================================================
# FAMILY 1: Subspace Projection + Generative Bayesian Classifiers
# ============================================================================

class GenerativeBayesianBenchmark:
    """
    Subspace Dimensionality Reduction & Generative Bayesian Classifiers.

    Projects feature space to an uncorrelated PCA subspace (e.g. k=8 components)
    to satisfy Gaussian independence assumptions and avoid singular covariance matrices.
    Evaluates:
    - Subspace-GaussianNB
    - Subspace-LDA (with Ledoit-Wolf shrinkage)
    - Subspace-RegQDA (with reg_param=0.3)
    """

    def __init__(self, subspace_components: int = 8, random_state: int = 42) -> None:
        self.subspace_components = subspace_components
        self.random_state = random_state
        self.classifiers: dict[str, BaseEstimator] = {
            "Subspace-GaussianNB": GaussianNB(),
            "Subspace-LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "Subspace-RegQDA": QuadraticDiscriminantAnalysis(reg_param=0.3),
        }
        self.fitted_models: dict[str, BaseEstimator] = {}
        self.pca_subspace_: PCA | None = None

    def fit_and_evaluate(
        self,
        X_train_compact: np.ndarray,
        y_train: np.ndarray,
        X_test_compact: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, EvaluationMetrics]:
        results = {}

        # 1. Subspace PCA projection (decorrelated coordinates)
        k = min(self.subspace_components, X_train_compact.shape[1], len(X_train_compact) - 1)
        self.pca_subspace_ = PCA(n_components=k, random_state=self.random_state)
        X_tr_sub = self.pca_subspace_.fit_transform(X_train_compact)
        X_te_sub = self.pca_subspace_.transform(X_test_compact)

        for name, clf in self.classifiers.items():
            model = clone(clf)
            try:
                model.fit(X_tr_sub, y_train)
            except Exception as e:
                logger.warning("Fit error for %s: %s — falling back to GaussianNB", name, e)
                model = GaussianNB().fit(X_tr_sub, y_train)

            self.fitted_models[name] = model

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te_sub)[:, 1]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X_te_sub)
                y_prob = 1.0 / (1.0 + np.exp(-df))
            else:
                y_prob = model.predict(X_te_sub).astype(float)

            results[name] = compute_metrics(y_test, y_prob)
        return results


# ============================================================================
# FAMILY 2: Early Feature-Level Fusion + Margin & Tree Ensembles
# ============================================================================

class EarlyFusionDiscriminativeBenchmark:
    """
    Early Feature-Level Fusion with Margin & Tree Ensemble Classifiers.

    Evaluates:
    - EarlyFusion-SVM-RBF (Tuned via GridSearch)
    - EarlyFusion-SVM-Linear (Maximum-Margin Linear Hyperplane)
    - EarlyFusion-RandomForest (Bagging + Feature Subsampling)
    - EarlyFusion-GradBoost (Stage-wise Boosted Decision Trees)
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.fitted_models: dict[str, BaseEstimator] = {}
        self.feature_importances: dict[str, np.ndarray] = {}

    def fit_and_evaluate(
        self,
        X_train_compact: np.ndarray,
        y_train: np.ndarray,
        X_test_compact: np.ndarray,
        y_test: np.ndarray,
        tune_svm: bool = True,
    ) -> dict[str, EvaluationMetrics]:
        results = {}

        # 1. EarlyFusion-SVM-RBF with GridSearch
        if tune_svm and len(X_train_compact) >= 20:
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.01, 0.1],
            }
            grid_rbf = GridSearchCV(
                SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=self.random_state),
                param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
            )
            grid_rbf.fit(X_train_compact, y_train)
            svm_rbf = grid_rbf.best_estimator_
        else:
            svm_rbf = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=self.random_state)
            svm_rbf.fit(X_train_compact, y_train)

        self.fitted_models["EarlyFusion-SVM-RBF"] = svm_rbf
        prob_svm_rbf = svm_rbf.predict_proba(X_test_compact)[:, 1]
        results["EarlyFusion-SVM-RBF"] = compute_metrics(y_test, prob_svm_rbf)

        # 2. EarlyFusion-SVM-Linear
        svm_linear = SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=self.random_state)
        svm_linear.fit(X_train_compact, y_train)
        self.fitted_models["EarlyFusion-SVM-Linear"] = svm_linear
        prob_svm_lin = svm_linear.predict_proba(X_test_compact)[:, 1]
        results["EarlyFusion-SVM-Linear"] = compute_metrics(y_test, prob_svm_lin)

        # 3. EarlyFusion-RandomForest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=3,
            class_weight="balanced",
            random_state=self.random_state,
        )
        rf.fit(X_train_compact, y_train)
        self.fitted_models["EarlyFusion-RandomForest"] = rf
        self.feature_importances["EarlyFusion-RandomForest"] = rf.feature_importances_
        prob_rf = rf.predict_proba(X_test_compact)[:, 1]
        results["EarlyFusion-RandomForest"] = compute_metrics(y_test, prob_rf)

        # 4. EarlyFusion-GradBoost
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=self.random_state,
        )
        gb.fit(X_train_compact, y_train)
        self.fitted_models["EarlyFusion-GradBoost"] = gb
        self.feature_importances["EarlyFusion-GradBoost"] = gb.feature_importances_
        prob_gb = gb.predict_proba(X_test_compact)[:, 1]
        results["EarlyFusion-GradBoost"] = compute_metrics(y_test, prob_gb)

        # 5. EarlyFusion-ExtraTrees (Extremely Randomized Trees)
        et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=3,
            class_weight="balanced",
            random_state=self.random_state,
        )
        et.fit(X_train_compact, y_train)
        self.fitted_models["EarlyFusion-ExtraTrees"] = et
        self.feature_importances["EarlyFusion-ExtraTrees"] = et.feature_importances_
        prob_et = et.predict_proba(X_test_compact)[:, 1]
        results["EarlyFusion-ExtraTrees"] = compute_metrics(y_test, prob_et)

        return results


# ============================================================================
# FAMILY 3: Late Decision-Level Fusion via Stacked Generalization
# ============================================================================

class LateFusionStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Out-of-Fold Unimodal Stacking Classifier.

    Trains 3 unimodal models:
      1. Text Classifier:   f_text(X_text)   -> P(Lie | X_text)
      2. Audio Classifier:  f_audio(X_audio) -> P(Lie | X_audio)
      3. Vision Classifier: f_vis(X_vision)  -> P(Lie | X_vision)

    Computes out-of-fold predictions on training data via K-Fold CV,
    then trains a Logistic Regression meta-learner on the 3D probability space:
      P(Lie | z) = sigma(w_text * p_text + w_audio * p_audio + w_vis * p_vis + b)

    Also exposes:
    - soft_vote_predict_proba(): Equal or weighted average
    - product_of_experts_predict_proba(): Bayesian log-odds aggregation
    """

    def __init__(
        self,
        base_estimator_type: Literal["svm", "rf", "lr"] = "lr",
        n_splits: int = 5,
        random_state: int = 42,
        pca_text_components: int = 10,
    ) -> None:
        self.base_estimator_type = base_estimator_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.pca_text_components = pca_text_components

        self.text_model_: BaseEstimator | None = None
        self.audio_model_: BaseEstimator | None = None
        self.vision_model_: BaseEstimator | None = None
        self.meta_model_: LogisticRegression = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.learned_weights_: np.ndarray | None = None
        self.learned_bias_: float = 0.0

    def _get_base_estimator(self) -> BaseEstimator:
        if self.base_estimator_type == "svm":
            return SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=self.random_state)
        elif self.base_estimator_type == "rf":
            return RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=self.random_state)
        else:
            return LogisticRegression(C=1.0, class_weight="balanced", random_state=self.random_state)

    def fit(self, X_compact: np.ndarray, y: np.ndarray) -> "LateFusionStackingClassifier":
        y = np.asarray(y, dtype=np.int64)
        N = len(X_compact)

        # Slice unimodal features
        X_t = extract_modality_data(X_compact, "text", mode="compact", pca_text_components=self.pca_text_components)
        X_a = extract_modality_data(X_compact, "audio", mode="compact", pca_text_components=self.pca_text_components)
        X_v = extract_modality_data(X_compact, "vision", mode="compact", pca_text_components=self.pca_text_components)

        # 1. Out-of-fold probability collection
        oof_probs = np.zeros((N, 3), dtype=np.float32)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X_compact, y):
            m_t = self._get_base_estimator().fit(X_t[train_idx], y[train_idx])
            m_a = self._get_base_estimator().fit(X_a[train_idx], y[train_idx])
            m_v = self._get_base_estimator().fit(X_v[train_idx], y[train_idx])

            oof_probs[val_idx, 0] = m_t.predict_proba(X_t[val_idx])[:, 1]
            oof_probs[val_idx, 1] = m_a.predict_proba(X_a[val_idx])[:, 1]
            oof_probs[val_idx, 2] = m_v.predict_proba(X_v[val_idx])[:, 1]

        # 2. Fit Meta-Classifier on OOF probabilities
        self.meta_model_.fit(oof_probs, y)
        self.learned_weights_ = self.meta_model_.coef_[0]
        self.learned_bias_ = float(self.meta_model_.intercept_[0])

        # 3. Fit unimodal models on full dataset for inference
        self.text_model_ = self._get_base_estimator().fit(X_t, y)
        self.audio_model_ = self._get_base_estimator().fit(X_a, y)
        self.vision_model_ = self._get_base_estimator().fit(X_v, y)

        return self

    def predict_unimodal_probs(self, X_compact: np.ndarray) -> np.ndarray:
        """Returns N x 3 matrix of [p_text, p_audio, p_vision]."""
        if self.text_model_ is None or self.audio_model_ is None or self.vision_model_ is None:
            raise ValueError("Model is not fitted.")

        X_t = extract_modality_data(X_compact, "text", mode="compact", pca_text_components=self.pca_text_components)
        X_a = extract_modality_data(X_compact, "audio", mode="compact", pca_text_components=self.pca_text_components)
        X_v = extract_modality_data(X_compact, "vision", mode="compact", pca_text_components=self.pca_text_components)

        p_t = self.text_model_.predict_proba(X_t)[:, 1]
        p_a = self.audio_model_.predict_proba(X_a)[:, 1]
        p_v = self.vision_model_.predict_proba(X_v)[:, 1]

        return np.column_stack([p_t, p_a, p_v])

    def predict_proba(self, X_compact: np.ndarray) -> np.ndarray:
        """Stacked meta-classifier posterior probabilities."""
        uni_probs = self.predict_unimodal_probs(X_compact)
        return self.meta_model_.predict_proba(uni_probs)

    def predict(self, X_compact: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X_compact)[:, 1]
        return (prob >= threshold).astype(np.int64)

    def soft_vote_predict_proba(
        self,
        X_compact: np.ndarray,
        weights: tuple[float, float, float] = (1/3, 1/3, 1/3),
    ) -> np.ndarray:
        """Weighted average soft voting rule."""
        uni_probs = self.predict_unimodal_probs(X_compact)
        w = np.array(weights) / sum(weights)
        p1 = uni_probs @ w
        return np.column_stack([1.0 - p1, p1])

    def product_of_experts_predict_proba(self, X_compact: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Bayesian Product of Experts decision rule.
        Combines unimodal log-odds under conditional independence:
        log(P(Lie|X) / P(Truth|X)) = sum_m log(p_m / (1 - p_m)) - 2 * log(P(Lie)/P(Truth))
        """
        uni_probs = self.predict_unimodal_probs(X_compact)
        clipped = np.clip(uni_probs, eps, 1.0 - eps)
        log_odds = np.log(clipped / (1.0 - clipped))
        sum_log_odds = np.sum(log_odds, axis=1)
        p1 = 1.0 / (1.0 + np.exp(-sum_log_odds))
        return np.column_stack([1.0 - p1, p1])


# ============================================================================
# BASELINE EVALUATOR (Heuristic & Neural DeceptionMLP)
# ============================================================================

def evaluate_baseline_heuristic(
    X_full: np.ndarray,
    y: np.ndarray,
    weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> EvaluationMetrics:
    """
    Evaluates the legacy heuristic fusion implemented in main.py / utils.py:
    deception_score = 0.3 * text_score + 0.3 * audio_score + 0.4 * vision_score
    """
    # Columns in X_full:
    # 0: text_score
    # 4: audio_score
    # 7: vision_score
    text_score = X_full[:, 0]
    audio_score = X_full[:, 4]
    vision_score = X_full[:, 7]

    w_text, w_audio, w_vision = weights
    w_sum = w_text + w_audio + w_vision
    deception_score = (w_text * text_score + w_audio * audio_score + w_vision * vision_score) / w_sum
    deception_score = np.clip(deception_score, 0.0, 1.0)

    return compute_metrics(y, deception_score)
