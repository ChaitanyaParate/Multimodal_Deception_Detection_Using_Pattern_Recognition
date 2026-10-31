"""
Multimodal Pattern Recognition Pipeline
=========================================
Feature partitioning, leakage-free preprocessing, and subspace dimensionality
reduction module for multimodal deception detection.

Mathematical Foundations:
- High-Dimensionality Small-Sample Regime: N = 121 samples, d = 396 features (d >> N).
- Curse of dimensionality mitigated via modality-aware Subspace Projection:
  - 384-dimensional dense semantic embeddings are compressed via Principal Component Analysis (PCA)
    to k=10 orthogonal components explaining dominant textual semantic variance.
  - 12 domain-engineered features (4 text, 3 audio, 5 visual) are standardized and preserved.
  - Resulting compact multimodal vector has dimension d_compact = 22 (14 text, 3 audio, 5 visual),
    re-establishing a statistically well-conditioned manifold (N / d_compact ≈ 5.5).
- Strict, Leakage-Free Design: All statistics (StandardScaler mean/std, PCA eigenvectors)
  are fitted strictly on training folds and applied out-of-sample to validation/testing folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature column definitions
ENGINEERED_TEXT_COLS: list[str] = [
    "text_score",
    "hesitation_score",
    "transcript_len",
    "has_transcript",
]
EMBEDDING_TEXT_COLS: list[str] = [f"emb_{i}" for i in range(384)]
TEXT_COLS: list[str] = ENGINEERED_TEXT_COLS + EMBEDDING_TEXT_COLS  # 388 dims

AUDIO_COLS: list[str] = [
    "audio_score",
    "pitch_score",
    "silence_score",
]  # 3 dims

VISION_COLS: list[str] = [
    "vision_score",
    "blink_rate_per_min",
    "eye_movement_std",
    "blink_score",
    "eye_movement_score",
]  # 5 dims

DOMAIN_FEATURE_COLS: list[str] = ENGINEERED_TEXT_COLS + AUDIO_COLS + VISION_COLS  # 12 dims
ALL_FEATURE_COLS: list[str] = DOMAIN_FEATURE_COLS + EMBEDDING_TEXT_COLS  # 396 dims


@dataclass
class DatasetBundle:
    """Encapsulates dataset partitions, splits, and modality indices."""
    X_train_full: np.ndarray          # (N_train, 396) raw
    y_train: np.ndarray               # (N_train,)
    X_test_full: np.ndarray           # (N_test, 396) raw
    y_test: np.ndarray                # (N_test,)
    X_all_full: np.ndarray            # (N_all, 396) raw
    y_all: np.ndarray                 # (N_all,)
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_all: pd.DataFrame
    feature_names_full: list[str]
    compact_feature_names: list[str]


class MultimodalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free Multimodal Preprocessor and Subspace Transformer.

    1. Separates engineered domain features (12 dims) and text embeddings (384 dims).
    2. Standardizes engineered features using StandardScaler.
    3. Standardizes text embeddings, then fits PCA(n_components=pca_text_components)
       to reduce 384 embedding dimensions to compact, decorrelated semantic coordinates.
    4. Concatenates into a balanced 22-dimensional compact multimodal vector.
    5. Optionally applies Fisher's Linear Discriminant Analysis (LDA) or Mutual Information
       selection for supervised 1D/low-dimensional projection.
    """

    def __init__(
        self,
        pca_text_components: int = 10,
        apply_lda_subspace: bool = False,
        lda_components: int = 1,
        apply_mutual_info_k: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.pca_text_components = pca_text_components
        self.apply_lda_subspace = apply_lda_subspace
        self.lda_components = lda_components
        self.apply_mutual_info_k = apply_mutual_info_k
        self.random_state = random_state

        # Fitted components
        self.scaler_engineered_ = StandardScaler()
        self.scaler_emb_ = StandardScaler()
        self.pca_text_ = PCA(n_components=self.pca_text_components, random_state=self.random_state)
        self.lda_projector_: LinearDiscriminantAnalysis | None = None
        self.mi_selector_: SelectKBest | None = None

        self.compact_feature_names_: list[str] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | None = None) -> "MultimodalPreprocessor":
        if isinstance(X, pd.DataFrame):
            X_arr = X[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        # Slice engineered (0..12) and embeddings (12..396)
        X_eng = X_arr[:, :12]
        X_emb = X_arr[:, 12:]

        # Fit scalers
        self.scaler_engineered_.fit(X_eng)
        X_emb_scaled = self.scaler_emb_.fit_transform(X_emb)

        # Fit PCA on embeddings
        self.pca_text_.fit(X_emb_scaled)

        # Construct compact feature names
        text_eng = ENGINEERED_TEXT_COLS
        pca_names = [f"pca_text_{i+1}" for i in range(self.pca_text_components)]
        self.compact_feature_names_ = text_eng + pca_names + AUDIO_COLS + VISION_COLS

        # Optional supervised subspace transforms
        if y is not None:
            X_compact = self._transform_to_compact(X_arr)
            if self.apply_lda_subspace:
                self.lda_projector_ = LinearDiscriminantAnalysis(n_components=self.lda_components)
                self.lda_projector_.fit(X_compact, y)
            if self.apply_mutual_info_k is not None:
                self.mi_selector_ = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.apply_mutual_info_k, X_compact.shape[1]),
                )
                self.mi_selector_.fit(X_compact, y)

        return self

    def _transform_to_compact(self, X_arr: np.ndarray) -> np.ndarray:
        X_eng = X_arr[:, :12]
        X_emb = X_arr[:, 12:]

        X_eng_scaled = self.scaler_engineered_.transform(X_eng)
        X_emb_scaled = self.scaler_emb_.transform(X_emb)
        X_pca = self.pca_text_.transform(X_emb_scaled)

        # Re-partition into structured compact blocks:
        # Text compact: 4 engineered text + 10 PCA text = 14 dims
        # Audio: 3 engineered audio = 3 dims
        # Vision: 5 engineered vision = 5 dims
        # Order matches: text_eng (cols 0..3), pca (10 dims), audio (cols 4..6), vision (cols 7..11)
        text_eng_scaled = X_eng_scaled[:, :4]
        audio_scaled = X_eng_scaled[:, 4:7]
        vision_scaled = X_eng_scaled[:, 7:12]

        X_compact = np.hstack([text_eng_scaled, X_pca, audio_scaled, vision_scaled]).astype(np.float32)
        return X_compact

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_compact = self._transform_to_compact(X_arr)

        if self.lda_projector_ is not None:
            return self.lda_projector_.transform(X_compact).astype(np.float32)
        if self.mi_selector_ is not None:
            return self.mi_selector_.transform(X_compact).astype(np.float32)

        return X_compact


def get_modality_slices(
    mode: Literal["full", "compact"] = "compact",
    pca_text_components: int = 10,
) -> dict[str, slice | list[int]]:
    """
    Returns column slice/index ranges for unimodal partitioning.

    For 'compact' (22 dims total):
      - text:   indices 0..(4 + pca_text_components) -> 0..14 (14 dims)
      - audio:  indices (4 + pca_text_components)..(7 + pca_text_components) -> 14..17 (3 dims)
      - vision: indices (7 + pca_text_components)..22 -> 17..22 (5 dims)

    For 'full' (396 dims total):
      - text:   indices [0, 1, 2, 3] + list(range(12, 396)) -> (388 dims)
      - audio:  indices [4, 5, 6] -> (3 dims)
      - vision: indices [7, 8, 9, 10, 11] -> (5 dims)
    """
    if mode == "compact":
        n_text = 4 + pca_text_components
        n_audio = 3
        n_vision = 5
        return {
            "text": slice(0, n_text),
            "audio": slice(n_text, n_text + n_audio),
            "vision": slice(n_text + n_audio, n_text + n_audio + n_vision),
        }
    else:
        text_indices = [0, 1, 2, 3] + list(range(12, 396))
        audio_indices = [4, 5, 6]
        vision_indices = [7, 8, 9, 10, 11]
        return {
            "text": text_indices,
            "audio": audio_indices,
            "vision": vision_indices,
        }


def extract_modality_data(
    X: np.ndarray,
    modality: Literal["text", "audio", "vision", "bimodal_av", "bimodal_ta", "bimodal_tv", "all"],
    mode: Literal["compact", "full"] = "compact",
    pca_text_components: int = 10,
) -> np.ndarray:
    """Slices a feature matrix by modality."""
    slices = get_modality_slices(mode=mode, pca_text_components=pca_text_components)

    if modality == "text":
        return X[:, slices["text"]]
    elif modality == "audio":
        return X[:, slices["audio"]]
    elif modality == "vision":
        return X[:, slices["vision"]]
    elif modality == "bimodal_av":
        if mode == "compact":
            return np.hstack([X[:, slices["audio"]], X[:, slices["vision"]]])
        else:
            return X[:, slices["audio"] + slices["vision"]]
    elif modality == "bimodal_ta":
        if mode == "compact":
            return np.hstack([X[:, slices["text"]], X[:, slices["audio"]]])
        else:
            return X[:, slices["text"] + slices["audio"]]
    elif modality == "bimodal_tv":
        if mode == "compact":
            return np.hstack([X[:, slices["text"]], X[:, slices["vision"]]])
        else:
            return X[:, slices["text"] + slices["vision"]]
    elif modality == "all":
        return X
    else:
        raise ValueError(f"Unknown modality: {modality}")


def load_and_bundle_dataset(
    csv_path: str | Path = "outputs/features_full.csv",
    pca_text_components: int = 10,
) -> DatasetBundle:
    """
    Loads outputs/features_full.csv, validates columns, and packages into DatasetBundle.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Feature CSV not found: {p}. Ensure features have been extracted.")

    df = pd.read_csv(p)
    logger.info("Loaded dataset from %s (Total rows: %d, cols: %d)", p, len(df), len(df.columns))

    # Verify all feature columns are present
    missing_cols = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required feature columns: {missing_cols[:5]}... (total {len(missing_cols)})")

    # Split train vs test
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)

    if df_train.empty:
        raise ValueError("No samples with split=='train' found.")

    X_train_full = df_train[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy(dtype=np.int64)

    if df_test.empty:
        # Fallback if no explicit test split
        logger.warning("No explicit split=='test' found. Splitting from train (80/20).")
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
        X_train_full = df_train[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = df_train["label"].to_numpy(dtype=np.int64)

    X_test_full = df_test[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int64)

    X_all_full = df[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy(dtype=np.int64)

    text_eng = ENGINEERED_TEXT_COLS
    pca_names = [f"pca_text_{i+1}" for i in range(pca_text_components)]
    compact_feature_names = text_eng + pca_names + AUDIO_COLS + VISION_COLS

    return DatasetBundle(
        X_train_full=X_train_full,
        y_train=y_train,
        X_test_full=X_test_full,
        y_test=y_test,
        X_all_full=X_all_full,
        y_all=y_all,
        df_train=df_train,
        df_test=df_test,
        df_all=df,
        feature_names_full=ALL_FEATURE_COLS,
        compact_feature_names=compact_feature_names,
    )
