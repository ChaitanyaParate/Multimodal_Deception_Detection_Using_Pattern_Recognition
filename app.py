"""
AI Lie Detector // Multimodal Pattern Recognition Studio
========================================================
Interactive Multimodal Inspection Studio & Multi-Model Inference Platform.
Provides real-time telemetry, model consensus, ocular/acoustic deep dives,
synchronized behavioral anomaly timeline, local feature attribution,
latent subspace projection, and OpenCV HUD video generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# Ensure offline operation for HuggingFace / Transformers to prevent peer resets
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from model import DeceptionMLP
from predict import extract_single_video_features
from pr_pipeline import ALL_FEATURE_COLS, DOMAIN_FEATURE_COLS, MultimodalPreprocessor
from telemetry import get_or_compute_telemetry, get_video_metadata
from visualizer import HESITATION_WORDS, get_analysis_for_video, render_annotated_video

logger = logging.getLogger(__name__)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="VERITAS // Multimodal Deception Intelligence Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Sleek Dark Glassmorphism CSS with Google Fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #080c14 !important;
        color: #f1f5f9 !important;
    }

    .stApp {
        background: radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 88% 22%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%),
                    #080c14 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0d121f !important;
        border-right: 1px solid rgba(56, 189, 248, 0.18) !important;
    }

    /* Card Components */
    .metric-card {
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
    }

    /* Animated Pulsing Status Indicators */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .live-dot-green {
        background-color: #10b981;
        animation: pulse-green 2s infinite;
    }
    .live-dot-red {
        background-color: #ef4444;
        animation: pulse-red 2s infinite;
    }

    /* Glowing Decision HUD Cards */
    .decision-hud-deceptive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(30, 27, 46, 0.85) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 22px rgba(239, 68, 68, 0.25);
    }
    .decision-hud-truthful {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 22px rgba(16, 185, 129, 0.25);
    }

    .forensic-stat-chip {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 8px 14px;
        text-align: center;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
    }

    /* Tab Headers */
    [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 10px !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2) !important;
        padding-bottom: 6px !important;
    }
    [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px 8px 0px 0px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-bottom: none !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
    }
    [data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
        background-color: rgba(30, 41, 59, 0.85) !important;
    }
    [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(37, 99, 235, 0.3) 0%, rgba(30, 41, 59, 0.9) 100%) !important;
        color: #ffffff !important;
        border-top: 2px solid #38bdf8 !important;
        border-left: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.3) !important;
    }

    .transcript-box {
        background: rgba(19, 27, 46, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 14px;
        margin-bottom: 12px;
    }
    .hesitation-pill {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def render_image(container: Any, image_path: str, caption: str | None = None) -> None:
    """Renders image compatibly across all Streamlit versions (<1.40 and >=1.40)."""
    try:
        container.image(image_path, caption=caption, use_column_width=True)
    except TypeError:
        container.image(image_path, caption=caption, use_container_width=True)


@st.cache_resource
def load_all_models_and_preprocessor() -> dict[str, Any] | None:
    """Loads serialized classical model registry and preprocessor."""
    models_path = Path("outputs/trained_models.joblib")
    if not models_path.exists():
        return None
    try:
        return joblib.load(models_path)
    except Exception as e:
        logger.error("Error loading trained models joblib: %s", e)
        return None


@st.cache_resource
def load_deception_mlp_model() -> tuple[DeceptionMLP | None, np.ndarray, np.ndarray]:
    """Loads trained PyTorch DeceptionMLP baseline model."""
    ckpt_path = Path("outputs/model_full.pt")
    if not ckpt_path.exists():
        return None, np.zeros(396), np.ones(396)
    try:
        dev = torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        in_dim = int(ckpt.get("input_dim", 396))
        model = DeceptionMLP(input_dim=in_dim).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        s_mean = np.array(ckpt.get("scaler_mean", [0.0] * in_dim), dtype=np.float32)
        s_scale = np.array(ckpt.get("scaler_scale", [1.0] * in_dim), dtype=np.float32)
        safe_scale = np.where(s_scale == 0.0, 1.0, s_scale)
        return model, s_mean, safe_scale
    except Exception as e:
        logger.warning("Could not load DeceptionMLP: %s", e)
        return None, np.zeros(396), np.ones(396)


@st.cache_data
def load_features_dataset() -> pd.DataFrame:
    """Loads feature matrix dataframe."""
    p = Path("outputs/features_full.csv")
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def get_video_metadata(video_path: str) -> tuple[float, float, int]:
    """Extracts video duration (sec), FPS, and total frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 10.0, 30.0, 300
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    duration = total / fps if (total > 0 and fps > 0) else 10.0
    return float(duration), float(fps), total


@st.cache_data
def extract_uploaded_video_features(video_path_str: str) -> dict[str, Any]:
    """Runs genuine multimodal feature extraction on custom uploaded videos."""
    vpath = Path(video_path_str)
    return extract_single_video_features(
        video_path=vpath,
        work_dir=Path("inference_artifacts"),
        frame_step=8,
        whisper_model="tiny",
        whisper_device="auto",
        max_audio_seconds=30.0,
        vision_max_frames=120,
    )


def discover_available_videos() -> list[dict[str, str]]:
    """Discovers all dataset videos and root demo clips."""
    videos = []
    base_dir = Path("Real-life Deception Detection Dataset With Train Test")

    # Root demo clips
    root_clips = sorted(Path(".").glob("trial_*.mp4"))
    for p in root_clips:
        label_hint = "Deceptive" if "lie" in p.name.lower() else "Truthful"
        videos.append({
            "path": str(p),
            "display": f"⭐ [Root Sample] {p.name} ({label_hint})",
            "name": p.name,
        })

    # Test set
    test_dir = base_dir / "Test"
    if test_dir.exists():
        for p in sorted(test_dir.glob("*.mp4")):
            label_hint = "Deceptive" if "lie" in p.name.lower() else "Truthful"
            videos.append({
                "path": str(p),
                "display": f"🧪 [Holdout Test] {p.name} ({label_hint})",
                "name": p.name,
            })

    # Train set (sample first 15)
    train_dir = base_dir / "Train"
    if train_dir.exists():
        for p in sorted(train_dir.glob("*.mp4"))[:15]:
            label_hint = "Deceptive" if "lie" in p.name.lower() else "Truthful"
            videos.append({
                "path": str(p),
                "display": f"📚 [Training Split] {p.name} ({label_hint})",
                "name": p.name,
            })

    return videos


def compute_all_model_probabilities(
    X_raw: np.ndarray,
    artifact: dict[str, Any],
    mlp_model: DeceptionMLP | None,
    mlp_mean: np.ndarray,
    mlp_scale: np.ndarray,
    text_score: float,
    audio_score: float,
    vision_score: float,
) -> dict[str, float]:
    """Computes prediction probabilities across all model backends and ensemble consensus."""
    probs = {}
    prep = artifact["preprocessor"]
    models = artifact["models"]
    X_compact = prep.transform(X_raw)

    # Heuristic Baseline
    probs["Heuristic-Weighted"] = float(np.clip(0.3 * text_score + 0.3 * audio_score + 0.4 * vision_score, 0.0, 1.0))

    # DeceptionMLP
    if mlp_model is not None:
        try:
            X_scaled = (X_raw - mlp_mean) / mlp_scale
            with torch.no_grad():
                logit = mlp_model(torch.from_numpy(X_scaled.astype(np.float32))).item()
            probs["Deep-DeceptionMLP"] = float(1.0 / (1.0 + np.exp(-logit)))
        except Exception:
            probs["Deep-DeceptionMLP"] = 0.50

    # Classical Models
    for name, entry in models.items():
        clf = entry["model"]
        mtype = entry.get("type")
        try:
            if mtype == "late_soft":
                p = float(clf.soft_vote_predict_proba(X_compact)[0, 1])
            elif mtype == "late_poe":
                p = float(clf.product_of_experts_predict_proba(X_compact)[0, 1])
            elif hasattr(clf, "predict_proba"):
                sub_pca = entry.get("subspace_pca")
                if sub_pca is not None:
                    X_sub = sub_pca.transform(X_compact)
                    p = float(clf.predict_proba(X_sub)[0, 1])
                else:
                    p = float(clf.predict_proba(X_compact)[0, 1])
            else:
                p = 0.50
            probs[name] = float(np.clip(p, 0.0, 1.0))
        except Exception:
            probs[name] = 0.50

    # Compute top-performing ensemble consensus
    top_models = [
        "EarlyFusion-GradBoost",
        "EarlyFusion-RandomForest",
        "EarlyFusion-ExtraTrees",
        "EarlyFusion-SVM-Linear",
        "Subspace-RegQDA",
        "LateFusion-BayesianPoE",
        "Deep-DeceptionMLP",
    ]
    consensus_candidates = [probs[m] for m in top_models if m in probs]
    if not consensus_candidates:
        consensus_candidates = list(probs.values())
    ensemble_prob = float(np.mean(consensus_candidates))
    probs["★ ENSEMBLE CONSENSUS"] = ensemble_prob

    return probs


@st.cache_data
def extract_ear_curve_and_blinks(video_path: str, max_frames: int = 400) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Retrieves cached or extracts frame-by-frame EAR curve, blinks, and horizontal saccade std."""
    _, _, _, ear_arr, blinks_idx, thresh, saccade_std, _, _, _, _, _ = get_or_compute_telemetry(video_path)
    return ear_arr, blinks_idx, thresh, saccade_std


@st.cache_data
def extract_audio_pitch_track(video_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Retrieves cached or extracts downsampled waveform, fundamental frequency F0, silence mask, and sampling rate."""
    _, _, _, _, _, _, _, y_downsampled, times, f0_clean, sr, silence_mask = get_or_compute_telemetry(video_path)
    return y_downsampled, times, f0_clean, sr, silence_mask


# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main() -> None:
    # Sidebar Header Badge
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;">
        <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
            <span class="live-dot live-dot-green"></span>
            <span>VERITAS STUDIO</span>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 5px; line-height: 1.4;">
            13 Models Active • 22-Dim Subspace Manifold • Live Telemetry
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Forensic Controls")
    st.sidebar.markdown("**Pattern Recognition Forensics Platform**")
    st.sidebar.markdown("---")

    # Load resources
    artifact = load_all_models_and_preprocessor()
    mlp_model, mlp_mean, mlp_scale = load_deception_mlp_model()
    df_features = load_features_dataset()

    if artifact is None:
        st.sidebar.error("⚠️ Model registry not found. Run `python benchmark.py` to train & serialize models.")

    # Video Selection
    available_videos = discover_available_videos()
    video_labels = [v["display"] for v in available_videos]

    selected_display = st.sidebar.selectbox("Select Video Clip:", video_labels, index=0)
    selected_video_item = next(v for v in available_videos if v["display"] == selected_display)
    video_path = selected_video_item["path"]
    is_custom_upload = False

    # Custom Video Upload option
    uploaded_file = st.sidebar.file_uploader("Or Upload Custom MP4 Video:", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        is_custom_upload = True
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_up:
            tmp_up.write(uploaded_file.read())
            video_path = tmp_up.name
            st.sidebar.success(f"Loaded custom video: {uploaded_file.name}")

    # Model Backend Selector
    model_choices = [
        "All Models Consensus",
        "LateFusion-BayesianPoE",
        "LateFusion-StackingLR",
        "LateFusion-SoftVoting",
        "EarlyFusion-SVM-Linear",
        "EarlyFusion-SVM-RBF",
        "EarlyFusion-RandomForest",
        "EarlyFusion-ExtraTrees",
        "EarlyFusion-GradBoost",
        "Subspace-RegQDA",
        "Subspace-LDA",
        "Deep-DeceptionMLP",
        "Heuristic-Weighted",
    ]
    active_model = st.sidebar.selectbox("Active Inference Model:", model_choices, index=0)

    # HUD Generation Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live HUD Video Annotator")
    gen_hud_btn = st.sidebar.button("🎬 Generate Annotated HUD Video", use_container_width=True)

    # Top Hero Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: #38bdf8; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                <span class="live-dot live-dot-green"></span> FORENSIC TELEMETRY ENGINE • V2.4 ACTIVE
            </div>
            <h1 style="margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 50%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                VERITAS // Multimodal Deception Intelligence Studio
            </h1>
            <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 4px;">
                Cross-Paradigm Pattern Recognition Benchmark • Ocular & Acoustic Synchronization • Subspace Manifold Forensics
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div class="forensic-stat-chip">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Manifold</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #38bdf8;">22-Dim PCA</div>
            </div>
            <div class="forensic-stat-chip">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Architectures</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #a855f7;">13 Models</div>
            </div>
            <div class="forensic-stat-chip">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Holdout Top</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #10b981;">83.3% F1</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2 = st.tabs(["🎥 Live Video Inspector & Multi-Model Predictor", "📊 Architecture & Benchmark Explorer"])

    # Load analysis data for selected video
    analysis_data = get_analysis_for_video(video_path, model_name=active_model if active_model != "All Models Consensus" else "LateFusion-BayesianPoE")

    # Match row in dataset if available
    matched_row = None
    if not df_features.empty and not is_custom_upload:
        m = df_features[df_features["video_path"].str.endswith(Path(video_path).name)]
        if not m.empty:
            matched_row = m.iloc[0]

    # Pre-extract video metadata, dense ocular tracking, and audio contour for Live Telemetry Sync
    vid_dur, vid_fps, vid_frames = get_video_metadata(video_path)
    ear_curve, blinks_idx, baseline_thresh, live_saccade_std = extract_ear_curve_and_blinks(video_path)
    y_wave, times_wave, f0_track, sr_rate, silence_mask = extract_audio_pitch_track(video_path)

    # Compute dense live ocular rates
    live_blink_rate = (len(blinks_idx) / max(vid_dur, 0.1)) * 60.0 if len(blinks_idx) > 0 else 0.0
    live_blink_score = float(np.clip((live_blink_rate - 8.0) / (35.0 - 8.0), 0.0, 1.0))
    live_saccade_score = float(np.clip((live_saccade_std - 1.5) / (10.0 - 1.5), 0.0, 1.0))
    live_vision_score = float((live_blink_score + live_saccade_score) / 2.0)

    # Compute predictions across all models with live telemetry sync
    X_raw = None
    if artifact is not None:
        if matched_row is not None:
            X_raw = matched_row[ALL_FEATURE_COLS].to_numpy(dtype=np.float32).reshape(1, -1)

            # Preserve dataset's 8-frame saccade displacement if valid (> 0)
            orig_blink_rate = float(matched_row.get("blink_rate_per_min", 0.0))
            orig_eye_std = float(matched_row.get("eye_movement_std", 0.0))
            orig_eye_score = float(matched_row.get("eye_movement_score", 0.5))

            if orig_blink_rate == 0.0 and len(blinks_idx) > 0:
                if "blink_rate_per_min" in ALL_FEATURE_COLS:
                    X_raw[0, ALL_FEATURE_COLS.index("blink_rate_per_min")] = live_blink_rate
                if "blink_score" in ALL_FEATURE_COLS:
                    X_raw[0, ALL_FEATURE_COLS.index("blink_score")] = live_blink_score
                cur_vision_score = float(0.5 * live_blink_score + 0.5 * orig_eye_score)
                if "vision_score" in ALL_FEATURE_COLS:
                    X_raw[0, ALL_FEATURE_COLS.index("vision_score")] = cur_vision_score
                analysis_data["features"]["blink_rate_per_min"] = live_blink_rate
                analysis_data["modality_scores"]["vision"] = cur_vision_score
            else:
                cur_eye_std = orig_eye_std if orig_eye_std > 0 else live_saccade_std
                analysis_data["features"]["blink_rate_per_min"] = orig_blink_rate if orig_blink_rate > 0 else live_blink_rate
                analysis_data["features"]["eye_movement_std"] = cur_eye_std
                analysis_data["modality_scores"]["vision"] = float(matched_row.get("vision_score", live_vision_score))

        elif is_custom_upload:
            with st.spinner("Extracting real multimodal features from uploaded video (Whisper + MediaPipe + Librosa)..."):
                uploaded_feats = extract_uploaded_video_features(video_path)
                if len(blinks_idx) > 0:
                    uploaded_feats["blink_rate_per_min"] = live_blink_rate
                    uploaded_feats["blink_score"] = live_blink_score
                    uploaded_feats["eye_movement_std"] = live_saccade_std
                    uploaded_feats["eye_movement_score"] = live_saccade_score
                    uploaded_feats["vision_score"] = live_vision_score

                X_raw = np.array([uploaded_feats.get(c, 0.0) for c in ALL_FEATURE_COLS], dtype=np.float32).reshape(1, -1)
                analysis_data["transcript"] = uploaded_feats.get("transcript", "")
                analysis_data["modality_scores"] = {
                    "text": float(uploaded_feats.get("text_score", 0.5)),
                    "audio": float(uploaded_feats.get("audio_score", 0.5)),
                    "vision": live_vision_score,
                }
                analysis_data["features"]["blink_rate_per_min"] = live_blink_rate
                analysis_data["features"]["eye_movement_std"] = live_saccade_std

        if X_raw is not None:
            model_probs = compute_all_model_probabilities(
                X_raw, artifact, mlp_model, mlp_mean, mlp_scale,
                analysis_data["modality_scores"]["text"],
                analysis_data["modality_scores"]["audio"],
                analysis_data["modality_scores"]["vision"],
            )
        else:
            p_base = analysis_data["deception_prob"]
            model_probs = {"LateFusion-BayesianPoE": p_base, "EarlyFusion-SVM-Linear": p_base, "Heuristic-Weighted": p_base}
    else:
        p_base = analysis_data["deception_prob"]
        model_probs = {"LateFusion-BayesianPoE": p_base, "EarlyFusion-SVM-Linear": p_base, "Heuristic-Weighted": p_base}

    # Synchronize verdict, probability, and model name BEFORE HUD video is rendered or displayed (Fix Task 2.1)
    if active_model == "All Models Consensus":
        pred_prob = model_probs.get("★ ENSEMBLE CONSENSUS", model_probs.get("LateFusion-BayesianPoE", 0.50))
        active_display_name = "Ensemble Consensus"
    else:
        pred_prob = model_probs.get(active_model, analysis_data.get("deception_prob", 0.50))
        active_display_name = active_model

    verdict = "DECEPTIVE" if pred_prob >= 0.50 else "TRUTHFUL"
    conf = pred_prob if verdict == "DECEPTIVE" else (1.0 - pred_prob)

    analysis_data["deception_prob"] = float(pred_prob)
    analysis_data["verdict"] = verdict
    analysis_data["model_name"] = active_display_name

    annotated_video_path = Path("outputs") / f"{Path(video_path).stem}_annotated.mp4"

    # Handle HUD Generation click
    if gen_hud_btn:
        with st.spinner("Processing video frames with MediaPipe FaceMesh & rendering OpenCV HUD..."):
            annotated_video_path = render_annotated_video(video_path, analysis_data=analysis_data, output_path=annotated_video_path)
            st.sidebar.success("HUD Video Ready!")

    # Sidebar: 1-Click "Export Forensic Report (JSON)" Button (Task 4.2)
    forensic_report_data = {
        "metadata": {
            "video_name": Path(video_path).name,
            "video_duration_seconds": round(float(vid_dur), 2),
            "video_fps": round(float(vid_fps), 2),
            "total_frames": int(vid_frames),
            "ground_truth": analysis_data.get("ground_truth", "UNLABELED"),
            "active_model": active_display_name,
            "primary_verdict": verdict,
            "deception_probability": round(float(pred_prob), 4),
            "confidence": round(float(conf), 4),
        },
        "modality_scores": {
            "linguistic": round(float(analysis_data["modality_scores"].get("text", 0.5)), 4),
            "acoustic": round(float(analysis_data["modality_scores"].get("audio", 0.5)), 4),
            "ocular": round(float(analysis_data["modality_scores"].get("vision", 0.5)), 4),
        },
        "all_model_probabilities": {
            k: round(float(v), 4) for k, v in model_probs.items()
        },
        "ocular_telemetry": {
            "blink_count": int(len(blinks_idx)),
            "blink_rate_per_min": round(float(live_blink_rate), 2),
            "saccade_dispersion_std_px": round(float(live_saccade_std), 4),
            "ear_baseline_threshold": round(float(baseline_thresh), 4),
            "blink_event_frame_indices": [int(x) for x in blinks_idx],
        },
        "vocal_telemetry": {
            "mean_f0_hz": round(float(np.nanmean(f0_track)), 1) if not np.all(np.isnan(f0_track)) else 0.0,
            "silence_ratio": round(float(np.mean(silence_mask)), 4) if len(silence_mask) > 0 else 0.0,
            "sampling_rate_hz": int(sr_rate),
        },
        "transcript": {
            "content": analysis_data.get("transcript", ""),
            "hesitation_markers": analysis_data.get("hesitation_tokens", []),
        },
    }
    st.sidebar.markdown("---")
    st.sidebar.subheader("Forensic Data Export")
    st.sidebar.download_button(
        label="📥 Export Forensic Report (JSON)",
        data=json.dumps(forensic_report_data, indent=2),
        file_name=f"{Path(video_path).stem}_forensic_report.json",
        mime="application/json",
        use_container_width=True,
    )

    # ========================================================================
    # TAB 1: LIVE INSPECTOR & MULTI-MODEL PREDICTOR
    # ========================================================================
    with tab1:
        col_video, col_consensus = st.columns([1.1, 1.3])

        with col_video:
            st.subheader("Video Playback")
            video_view_mode = st.radio(
                "Display Stream:",
                ["Raw Source Video", "Annotated Telemetry HUD Video"] if annotated_video_path.exists() else ["Raw Source Video"],
                horizontal=True,
            )

            if video_view_mode == "Annotated Telemetry HUD Video" and annotated_video_path.exists():
                st.video(str(annotated_video_path))
            else:
                st.video(str(video_path))

            # Verdict Badges & Decision Summary
            ground_truth = analysis_data.get("ground_truth", "UNLABELED")

            # High-Impact Decision HUD
            is_dec = (verdict == "DECEPTIVE")
            hud_class = "decision-hud-deceptive" if is_dec else "decision-hud-truthful"
            verdict_color = "#ef4444" if is_dec else "#10b981"
            verdict_glow = "rgba(239, 68, 68, 0.45)" if is_dec else "rgba(16, 185, 129, 0.45)"

            st.markdown(f"""
            <div class="{hud_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 8px; margin-bottom: 10px;">
                    <span style="font-size: 0.74rem; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; font-weight: 700;">
                        CLASSIFICATION DECISION
                    </span>
                    <span style="font-size: 0.78rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 2px 10px; font-weight: 600;">
                        {active_display_name}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.76rem; color: #94a3b8;">PREDICTED VERDICT</div>
                        <div style="font-size: 1.75rem; font-weight: 900; color: {verdict_color}; text-shadow: 0 0 16px {verdict_glow};">
                            {verdict}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.76rem; color: #94a3b8;">DECEPTION PROBABILITY</div>
                        <div style="font-size: 1.75rem; font-weight: 900; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">
                            {pred_prob * 100:.1f}%
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.8rem; color: #94a3b8;">
                    <div>Ground Truth: <strong style="color: {'#ef4444' if ground_truth == 'DECEPTIVE' else ('#10b981' if ground_truth == 'TRUTHFUL' else '#f59e0b')};">{ground_truth}</strong></div>
                    <div>•</div>
                    <div>Confidence: <strong style="color: #f8fafc;">{conf * 100:.1f}%</strong></div>
                    <div>•</div>
                    <div>Status: <strong style="color: {'#10b981' if (verdict == ground_truth) else '#ef4444'};">{'CONCORDANT ✅' if (verdict == ground_truth) else ('DISCORDANT ⚠️' if ground_truth != 'UNLABELED' else 'UNVERIFIED')}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_consensus:
            st.subheader("Multi-Model Consensus")
            st.caption("Comparison of predicted deception probability across all pattern recognition paradigms.")

            # Create Horizontal Bar Chart of All Models with ★ ENSEMBLE CONSENSUS highlighted
            plot_records = [{"Model": k, "Deception Probability": v} for k, v in model_probs.items()]
            df_plot = pd.DataFrame(plot_records).sort_values("Deception Probability", ascending=True)

            # Move ★ ENSEMBLE CONSENSUS to top
            if "★ ENSEMBLE CONSENSUS" in df_plot["Model"].values:
                ens_row = df_plot[df_plot["Model"] == "★ ENSEMBLE CONSENSUS"]
                other_rows = df_plot[df_plot["Model"] != "★ ENSEMBLE CONSENSUS"]
                df_plot = pd.concat([other_rows, ens_row], ignore_index=True)

            fig_bar, ax_bar = plt.subplots(figsize=(7, 4.4), dpi=150)
            fig_bar.patch.set_facecolor('#0b0f19')
            ax_bar.set_facecolor('#131b2e')

            bar_colors = []
            edge_colors = []
            line_widths = []
            for m_name, p_val in zip(df_plot["Model"], df_plot["Deception Probability"]):
                if m_name == "★ ENSEMBLE CONSENSUS":
                    bar_colors.append('#f59e0b')
                    edge_colors.append('#ffffff')
                    line_widths.append(1.8)
                elif p_val >= 0.50:
                    bar_colors.append('#ef4444')
                    edge_colors.append('#334155')
                    line_widths.append(1.0)
                else:
                    bar_colors.append('#10b981')
                    edge_colors.append('#334155')
                    line_widths.append(1.0)

            bars = ax_bar.barh(df_plot["Model"], df_plot["Deception Probability"],
                               color=bar_colors, edgecolor=edge_colors, linewidth=line_widths, height=0.65)
            ax_bar.axvline(0.50, color='#38bdf8', linestyle='--', linewidth=1.5, label='Decision Threshold (50%)')

            ax_bar.set_xlim(0, 1.05)
            ax_bar.set_xlabel("Deception Probability", color='#94a3b8', fontsize=9.5)
            ax_bar.tick_params(colors='#cbd5e1', labelsize=8.5)
            for spine in ax_bar.spines.values():
                spine.set_color('#334155')
            ax_bar.grid(axis='x', linestyle=':', color='#334155', alpha=0.7)
            ax_bar.legend(loc='lower right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)

            for bar in bars:
                w = bar.get_width()
                ax_bar.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%",
                            va='center', color='#ffffff', fontsize=8, fontweight='bold')

            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            # Modality Radar / Tri-Axial Signature & Behavioral Risk Index (Task 4.1)
            t_sc = float(analysis_data["modality_scores"].get("text", 0.50))
            a_sc = float(analysis_data["modality_scores"].get("audio", 0.50))
            v_sc = float(analysis_data["modality_scores"].get("vision", 0.50))

            scores_map = {
                "Linguistic Modality (Hesitations & Semantics)": t_sc,
                "Acoustic Modality (Pitch Modulation & Silence)": a_sc,
                "Ocular Modality (Blink Frequency & Saccades)": v_sc,
            }
            primary_driver_full = max(scores_map, key=scores_map.get)
            primary_driver_name = primary_driver_full.split(" (")[0]
            max_val = scores_map[primary_driver_full]

            if max_val >= 0.58:
                risk_tag = "CRITICAL DECEPTIVE DRIVER"
                badge_bg = "rgba(239, 68, 68, 0.16)"
                badge_border = "rgba(239, 68, 68, 0.45)"
                badge_color = "#f87171"
                pulse_cls = "live-dot-red"
            elif max_val >= 0.45:
                risk_tag = "ELEVATED ANOMALY"
                badge_bg = "rgba(245, 158, 11, 0.16)"
                badge_border = "rgba(245, 158, 11, 0.45)"
                badge_color = "#fbbf24"
                pulse_cls = "live-dot-red"
            else:
                risk_tag = "BASELINE HOMEOSTASIS"
                badge_bg = "rgba(16, 185, 129, 0.16)"
                badge_border = "rgba(16, 185, 129, 0.45)"
                badge_color = "#34d399"
                pulse_cls = "live-dot-green"

            st.markdown(f"""
            <div style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 10px; padding: 10px 16px; margin: 12px 0 14px 0; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="live-dot {pulse_cls}"></span>
                    <span style="font-size: 0.74rem; font-weight: 800; letter-spacing: 1px; color: #94a3b8; text-transform: uppercase;">Behavioral Risk Index:</span>
                    <span style="font-size: 0.85rem; font-weight: 800; color: #f8fafc;">{primary_driver_name}</span>
                </div>
                <div style="font-size: 0.72rem; font-weight: 800; color: {badge_color}; letter-spacing: 0.5px; text-transform: uppercase; background: rgba(0,0,0,0.35); padding: 3px 10px; border-radius: 6px; border: 1px solid {badge_border};">
                    {risk_tag} ({max_val:.2f})
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sleek 3-Card Progress Gauge Row
            m1, m2, m3 = st.columns(3)
            with m1:
                t_color = "#ef4444" if t_sc >= 0.55 else ("#38bdf8" if t_sc <= 0.40 else "#fbbf24")
                st.markdown(f"""
                <div class="metric-card" style="padding: 12px 14px; margin-bottom: 0px; border-left: 3px solid {t_color};">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">💬 Linguistic</span>
                        <span style="font-size: 1.05rem; font-weight: 800; color: {t_color}; font-family: 'JetBrains Mono', monospace;">{t_sc:.2f}</span>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.8); border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.2);">
                        <div style="background: {t_color}; width: {min(100.0, max(0.0, t_sc*100)):.1f}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="font-size: 0.68rem; color: #64748b; margin-top: 5px;">Hesitation & Embeddings</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                a_color = "#ef4444" if a_sc >= 0.55 else ("#38bdf8" if a_sc <= 0.40 else "#fbbf24")
                st.markdown(f"""
                <div class="metric-card" style="padding: 12px 14px; margin-bottom: 0px; border-left: 3px solid {a_color};">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">🎙️ Acoustic</span>
                        <span style="font-size: 1.05rem; font-weight: 800; color: {a_color}; font-family: 'JetBrains Mono', monospace;">{a_sc:.2f}</span>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.8); border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.2);">
                        <div style="background: {a_color}; width: {min(100.0, max(0.0, a_sc*100)):.1f}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="font-size: 0.68rem; color: #64748b; margin-top: 5px;">Pitch F0 & Vocal Silence</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                v_color = "#ef4444" if v_sc >= 0.55 else ("#38bdf8" if v_sc <= 0.40 else "#fbbf24")
                st.markdown(f"""
                <div class="metric-card" style="padding: 12px 14px; margin-bottom: 0px; border-left: 3px solid {v_color};">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                        <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">👁️ Ocular</span>
                        <span style="font-size: 1.05rem; font-weight: 800; color: {v_color}; font-family: 'JetBrains Mono', monospace;">{v_sc:.2f}</span>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.8); border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.2);">
                        <div style="background: {v_color}; width: {min(100.0, max(0.0, v_sc*100)):.1f}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="font-size: 0.68rem; color: #64748b; margin-top: 5px;">Blinks & Saccade Jitter</div>
                </div>
                """, unsafe_allow_html=True)

        # ====================================================================
        # FORENSIC TELEMETRY STAT CHIPS ROW
        # ====================================================================
        st.markdown("---")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            b_rate_val = live_blink_rate if len(blinks_idx) > 0 else 18.0
            st.markdown(f"""
            <div class="metric-card" style="padding: 12px 16px; margin-bottom: 0px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">👁️ Eye Blink Rate</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {'#ef4444' if b_rate_val > 25.0 else '#38bdf8'}; font-family: 'JetBrains Mono', monospace;">
                    {b_rate_val:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">/min</span>
                </div>
                <div style="font-size: 0.7rem; color: {'#ef4444' if b_rate_val > 25.0 else '#10b981'}; font-weight: 600;">
                    {'⚠️ Elevated Blink Stress' if b_rate_val > 25.0 else '✓ Baseline Ocular Motion'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sc2:
            st.markdown(f"""
            <div class="metric-card" style="padding: 12px 16px; margin-bottom: 0px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">🎯 Saccade Dispersion</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">
                    {live_saccade_std:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">px</span>
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8;">
                    Horizontal eye jitter std
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sc3:
            silence_pct = float(np.mean(silence_mask) * 100) if len(silence_mask) > 0 else 25.0
            st.markdown(f"""
            <div class="metric-card" style="padding: 12px 16px; margin-bottom: 0px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">⏸️ Vocal Silence Ratio</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {'#f59e0b' if silence_pct > 35.0 else '#10b981'}; font-family: 'JetBrains Mono', monospace;">
                    {silence_pct:.1f}%
                </div>
                <div style="font-size: 0.7rem; color: {'#f59e0b' if silence_pct > 35.0 else '#94a3b8'}; font-weight: 600;">
                    {'⚠️ Hesitant Pause Pattern' if silence_pct > 35.0 else '✓ Fluid Speech Rhythm'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sc4:
            valid_f0_all = f0_track[~np.isnan(f0_track)]
            f0_mean_disp = float(np.mean(valid_f0_all)) if len(valid_f0_all) > 5 else 180.0
            st.markdown(f"""
            <div class="metric-card" style="padding: 12px 16px; margin-bottom: 0px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">🎙️ Mean Vocal Pitch F0</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #a855f7; font-family: 'JetBrains Mono', monospace;">
                    {f0_mean_disp:.0f} <span style="font-size: 0.8rem; color: #94a3b8;">Hz</span>
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8;">
                    Fundamental frequency tracking
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================================
        # SYNCHRONIZED BEHAVIORAL EVENT TIMELINE STRIP (TASK 4)
        # ====================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⏱️ Synchronized Behavioral Anomaly Timeline")
        st.caption("Temporal alignment of ocular blinks, vocal pauses, and acoustic pitch stress. Scrub the video player above to these exact timestamps.")

        fig_time, ax_time = plt.subplots(figsize=(10, 1.5), dpi=150)
        fig_time.patch.set_facecolor('#0b0f19')
        ax_time.set_facecolor('#131b2e')

        # Track 1: Ocular Blinks (Y=2)
        blink_times = [float(idx * (vid_dur / max(len(ear_curve), 1))) for idx in blinks_idx]
        if blink_times:
            ax_time.scatter(blink_times, [2.0] * len(blink_times), color='#ef4444', marker='d', s=65, zorder=6, label=f'Ocular Blinks ({len(blink_times)})')

        # Track 2: Vocal Silence / Pauses (Y=1)
        spans_silence = []
        in_s = False
        s_start = 0.0
        for t_val, is_s in zip(times_wave, silence_mask):
            if is_s and not in_s:
                in_s = True
                s_start = float(t_val)
            elif not is_s and in_s:
                in_s = False
                dur_s = float(t_val - s_start)
                if dur_s >= 0.12:
                    spans_silence.append((s_start, dur_s))
        if in_s and (times_wave[-1] - s_start >= 0.12):
            spans_silence.append((s_start, float(times_wave[-1] - s_start)))
        if spans_silence:
            ax_time.broken_barh(spans_silence, (0.76, 0.48), facecolors='#f59e0b', edgecolors='#b45309', alpha=0.9, zorder=5, label=f'Vocal Pauses ({len(spans_silence)})')

        # Track 3: Pitch Stress Spikes (Y=0)
        valid_f0 = f0_track[~np.isnan(f0_track)]
        spans_stress = []
        if len(valid_f0) > 5:
            f0_mean = float(np.mean(valid_f0))
            f0_std = float(np.std(valid_f0))
            pitch_stress_mask = (~np.isnan(f0_track)) & (f0_track > (f0_mean + 0.85 * f0_std))
            in_st = False
            st_start = 0.0
            for t_val, is_st in zip(times_wave, pitch_stress_mask):
                if is_st and not in_st:
                    in_st = True
                    st_start = float(t_val)
                elif not is_st and in_st:
                    in_st = False
                    dur_st = float(t_val - st_start)
                    if dur_st >= 0.08:
                        spans_stress.append((st_start, dur_st))
            if in_st and (times_wave[-1] - st_start >= 0.08):
                spans_stress.append((st_start, float(times_wave[-1] - st_start)))
        if spans_stress:
            ax_time.broken_barh(spans_stress, (-0.24, 0.48), facecolors='#06b6d4', edgecolors='#0891b2', alpha=0.9, zorder=5, label=f'Pitch Stress ({len(spans_stress)})')

        # Timeline Layout
        ax_time.set_xlim(0, max(vid_dur, 1.0))
        ax_time.set_ylim(-0.6, 2.6)
        ax_time.set_yticks([0, 1, 2])
        ax_time.set_yticklabels(['Pitch Stress', 'Vocal Silence', 'Eye Blinks'], color='#cbd5e1', fontsize=8.5, fontweight='bold')
        ax_time.set_xlabel("Time Elapsed in Video (seconds) — Scrub video player above to these anomalous timestamps", color='#94a3b8', fontsize=8)
        ax_time.tick_params(colors='#cbd5e1', labelsize=8)
        for spine in ax_time.spines.values():
            spine.set_color('#334155')
        ax_time.grid(axis='x', linestyle=':', color='#334155', alpha=0.7)
        ax_time.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=7.5)
        fig_time.tight_layout()
        st.pyplot(fig_time)
        plt.close(fig_time)

        # ====================================================================
        # LOCAL FEATURE ATTRIBUTION ("WHY THIS VERDICT?")
        # ====================================================================
        if artifact is not None and X_raw is not None:
            st.markdown("---")
            st.subheader("💡 Local Feature Attribution (Why This Verdict?)")
            st.caption("Decomposition of margin contributions: bars pushing the prediction toward DECEPTIVE point right (crimson), while signals pushing toward TRUTHFUL point left (emerald).")

            prep = artifact["preprocessor"]
            X_compact = prep.transform(X_raw)

            # Extract weights from EarlyFusion-SVM-Linear
            svm_entry = artifact["models"].get("EarlyFusion-SVM-Linear")
            if svm_entry is not None:
                svm_weights = svm_entry["model"].coef_[0]
            else:
                svm_weights = np.ones(22)

            feature_mapping = [
                ("Blink Rate (/min)", 18),
                ("Horizontal Saccade", 19),
                ("Vision Score", 17),
                ("Pitch F0 Deviation", 15),
                ("Vocal Silence Ratio", 16),
                ("Audio Score", 14),
                ("Lexical Hesitation", 1),
                ("Semantic Deception", 0),
                ("Semantic PCA Component 1", 4),
                ("Semantic PCA Component 2", 5),
            ]

            contrib_records = []
            for fname, f_idx in feature_mapping:
                w_j = float(svm_weights[f_idx])
                x_j = float(X_compact[0, f_idx])
                contrib_records.append({"Feature": fname, "Contribution": w_j * x_j})

            df_contrib = pd.DataFrame(contrib_records).sort_values("Contribution", ascending=True)

            fig_att, ax_att = plt.subplots(figsize=(10, 2.8), dpi=150)
            fig_att.patch.set_facecolor('#0b0f19')
            ax_att.set_facecolor('#131b2e')

            att_colors = ['#ef4444' if c >= 0 else '#10b981' for c in df_contrib["Contribution"]]
            ax_att.barh(df_contrib["Feature"], df_contrib["Contribution"], color=att_colors, edgecolor='#334155', height=0.6)
            ax_att.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.2)

            ax_att.set_xlabel("Margin Contribution (w_j * x_j)", color='#94a3b8', fontsize=9)
            ax_att.tick_params(colors='#cbd5e1', labelsize=8)
            for spine in ax_att.spines.values():
                spine.set_color('#334155')
            ax_att.grid(axis='x', linestyle=':', color='#334155', alpha=0.5)

            # Annotate directions
            max_c = max(np.max(np.abs(df_contrib["Contribution"])), 0.5)
            ax_att.set_xlim(-max_c * 1.25, max_c * 1.25)
            ax_att.text(-max_c * 1.15, len(df_contrib) - 0.7, "← Pushes TRUTHFUL", color='#10b981', fontsize=8.5, fontweight='bold')
            ax_att.text(max_c * 0.4, len(df_contrib) - 0.7, "Pushes DECEPTIVE →", color='#ef4444', fontsize=8.5, fontweight='bold')

            fig_att.tight_layout()
            st.pyplot(fig_att)
            plt.close(fig_att)

        # ====================================================================
        # MODALITY DEEP-DIVE ROW (3 INTERACTIVE PLOTS)
        # ====================================================================
        st.markdown("---")
        st.subheader("🔬 Multimodal Signal Telemetry & Deep-Dive Analysis")

        d_col1, d_col2, d_col3 = st.columns(3)

        # 1. Acoustic Waveform & Pitch (F0) Track
        with d_col1:
            st.markdown("**1. Acoustic Waveform & Pitch ($F_0$) Contour**")

            fig_a, ax_a = plt.subplots(figsize=(5, 3.2), dpi=150)
            fig_a.patch.set_facecolor('#0b0f19')
            ax_a.set_facecolor('#131b2e')

            # Plot normalized waveform in background
            norm_wave = y_wave / (np.max(np.abs(y_wave)) + 1e-6)
            t_wave = np.linspace(0, len(y_wave) / sr_rate, len(y_wave))
            ax_a.plot(t_wave, norm_wave * 100 + 220, color='#38bdf8', alpha=0.28, label='Audio Waveform')

            # Shaded Vocal Silence intervals (Task 3)
            if np.any(silence_mask):
                ax_a.fill_between(times_wave, 80, 420, where=silence_mask, color='#ef4444', alpha=0.15, label='Silence / Pause')

            # Clean F0 contour without drop-off cliffs
            valid_f0 = ~np.isnan(f0_track)
            if np.any(valid_f0):
                ax_a.plot(times_wave, f0_track, color='#f59e0b', linewidth=2.0, label='Pitch F0 (Hz)')
                f0_mean_val = float(np.nanmean(f0_track))
                ax_a.axhline(f0_mean_val, color='#38bdf8', linestyle=':', linewidth=1.0, alpha=0.7, label=f'Mean F0 ({f0_mean_val:.0f}Hz)')

            ax_a.set_title("Vocal Pitch Modulation ($F_0$)", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_a.set_xlabel("Time (s)", color='#94a3b8', fontsize=8.5)
            ax_a.set_ylabel("Frequency (Hz)", color='#94a3b8', fontsize=8.5)
            ax_a.set_ylim(80, 420)
            ax_a.tick_params(colors='#cbd5e1', labelsize=8)
            for spine in ax_a.spines.values():
                spine.set_color('#334155')
            ax_a.grid(True, linestyle=':', color='#334155', alpha=0.6)
            ax_a.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=7.0)
            fig_a.tight_layout()
            st.pyplot(fig_a)
            plt.close(fig_a)

        # 2. Ocular Dynamics (EAR Timeline)
        with d_col2:
            st.markdown("**2. Ocular Dynamics & Blink Timeline (EAR)**")

            fig_e, ax_e = plt.subplots(figsize=(5, 3.2), dpi=150)
            fig_e.patch.set_facecolor('#0b0f19')
            ax_e.set_facecolor('#131b2e')

            if len(ear_curve) > 0:
                ax_e.plot(ear_curve, color='#10b981', linewidth=1.5, label='Eye Aspect Ratio (EAR)')
                ax_e.axhline(baseline_thresh, color='#ef4444', linestyle='--', linewidth=1.2, label=f'Blink Threshold ({baseline_thresh:.2f})')
                if len(blinks_idx) > 0:
                    ax_e.scatter(blinks_idx, ear_curve[blinks_idx], color='#ef4444', s=35, zorder=5, label=f'Blinks ({len(blinks_idx)})')

            ax_e.set_title("Dynamic Eye Aspect Ratio (EAR)", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_e.set_xlabel("Sampled Frames", color='#94a3b8', fontsize=8.5)
            ax_e.set_ylabel("EAR Value", color='#94a3b8', fontsize=8.5)
            ax_e.set_ylim(0.08, 0.45)
            ax_e.tick_params(colors='#cbd5e1', labelsize=8)
            for spine in ax_e.spines.values():
                spine.set_color('#334155')
            ax_e.grid(True, linestyle=':', color='#334155', alpha=0.6)
            ax_e.legend(loc='lower right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=7.5)
            fig_e.tight_layout()
            st.pyplot(fig_e)
            plt.close(fig_e)

        # 3. Linguistic & Semantic Breakdown (Dynamic Scaling)
        with d_col3:
            st.markdown("**3. Linguistic & Feature Deviations**")
            transcript_text = analysis_data.get("transcript", "No transcript extracted.")
            hesitation_tokens = analysis_data.get("hesitation_tokens", [])

            st.markdown(f"""
            <div class="transcript-box">
                <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:700; letter-spacing:0.5px;">Extracted Transcript:</span><br>
                <span style="color:#e2e8f0; font-size:0.88rem; font-style:italic; line-height:1.4;">"{transcript_text}"</span>
            </div>
            """, unsafe_allow_html=True)

            if hesitation_tokens:
                st.markdown("**Detected Hesitation Tokens:** " + "".join([f"<span class='hesitation-pill'>🗣️ {tok}</span>" for tok in hesitation_tokens]), unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#10b981; font-size:0.85rem;'>✅ No hesitation or disfluency markers detected.</span>", unsafe_allow_html=True)

            # Normalized Feature Deviations Bar Chart with Dynamic Dataset Quantile Bounds
            if not df_features.empty:
                feats = ["blink_rate_per_min", "eye_movement_std", "hesitation_score", "pitch_score", "silence_score"]
                feat_labels = ["Blink", "Saccade", "Hesitation", "Pitch", "Silence"]

                raw_sample = [
                    live_blink_rate if len(blinks_idx) > 0 else (float(matched_row["blink_rate_per_min"]) if matched_row is not None else float(analysis_data["features"].get("blink_rate_per_min", 18.0))),
                    live_saccade_std if matched_row is None else float(matched_row["eye_movement_std"]),
                    float(matched_row["hesitation_score"]) if matched_row is not None else float(analysis_data["features"].get("hesitation_score", 0.15)),
                    float(matched_row["pitch_score"]) if matched_row is not None else float(analysis_data["features"].get("pitch_score", 0.5)),
                    float(matched_row["silence_score"]) if matched_row is not None else float(analysis_data["features"].get("silence_score", 0.4)),
                ]

                raw_truth = [float(df_features[df_features["label"] == 0][f].mean()) for f in feats]
                raw_lie = [float(df_features[df_features["label"] == 1][f].mean()) for f in feats]

                # Dynamic bounds preventing clipping
                norm_truth = []
                norm_sample = []
                norm_lie = []
                for idx_f, f in enumerate(feats):
                    q95 = float(df_features[f].quantile(0.95)) if f in df_features else 1.0
                    f_max = max(q95, raw_truth[idx_f] * 1.3, raw_lie[idx_f] * 1.3, raw_sample[idx_f] * 1.15, 1e-3)
                    f_min = 0.0
                    norm_truth.append(np.clip((raw_truth[idx_f] - f_min) / (f_max - f_min), 0.0, 1.0))
                    norm_sample.append(np.clip((raw_sample[idx_f] - f_min) / (f_max - f_min), 0.0, 1.0))
                    norm_lie.append(np.clip((raw_lie[idx_f] - f_min) / (f_max - f_min), 0.0, 1.0))

                fig_f, ax_f = plt.subplots(figsize=(5, 2.2), dpi=150)
                fig_f.patch.set_facecolor('#0b0f19')
                ax_f.set_facecolor('#131b2e')

                x = np.arange(len(feats))
                w = 0.25
                ax_f.bar(x - w, norm_truth, w, label="Truth Mean", color='#2563eb', alpha=0.85)
                cur_bars = ax_f.bar(x, norm_sample, w, label="Current Video", color='#f59e0b')
                ax_f.bar(x + w, norm_lie, w, label="Lie Mean", color='#ef4444', alpha=0.85)

                # Add raw value text labels above current video bars
                for b, r_val, f in zip(cur_bars, raw_sample, feats):
                    if f == "blink_rate_per_min":
                        txt = f"{r_val:.0f}/m"
                    elif f == "eye_movement_std":
                        txt = f"{r_val:.1f}px"
                    else:
                        txt = f"{r_val:.2f}"
                    ax_f.text(b.get_x() + b.get_width()/2, min(b.get_height() + 0.04, 0.95), txt,
                              ha='center', va='bottom', color='#ffffff', fontsize=7, fontweight='bold')

                ax_f.set_ylim(0, 1.15)
                ax_f.set_xticks(x)
                ax_f.set_xticklabels(feat_labels, color='#cbd5e1', fontsize=7.5)
                ax_f.tick_params(colors='#cbd5e1', labelsize=7)
                for spine in ax_f.spines.values():
                    spine.set_color('#334155')
                ax_f.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=6.5)
                fig_f.tight_layout()
                st.pyplot(fig_f)
                plt.close(fig_f)

    # ========================================================================
    # TAB 2: PATTERN RECOGNITION ARCHITECTURE & BENCHMARK EXPLORER
    # ========================================================================
    with tab2:
        st.subheader("Dual-Protocol Benchmark Evaluation (N=121)")
        st.caption("Rigorous comparison between Official Holdout Split (N=12) and 5-Fold Stratified Cross-Validation.")

        # Load metrics CSVs
        csv_bench = Path("outputs/benchmark_metrics.csv")
        csv_ablation = Path("outputs/modality_ablation.csv")

        b_left, b_right = st.columns([1.4, 1.0])

        with b_left:
            if csv_bench.exists():
                df_b = pd.read_csv(csv_bench)
                st.dataframe(df_b, use_container_width=True, hide_index=True)
            else:
                st.info("Run `python benchmark.py` to generate benchmark metrics table.")

        with b_right:
            if csv_ablation.exists():
                df_a = pd.read_csv(csv_ablation)
                st.markdown("**Modality Ablation Performance**")
                st.dataframe(df_a[["Configuration", "Accuracy_Mean", "F1_Mean", "ROC_AUC_Mean"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Subspace Manifold Projection with Live Sample Coordinate")
        st.caption("Visualizes where the currently selected video is positioned within the 2D PCA & Fisher LDA latent spaces relative to the decision boundary.")

        proj_col1, proj_col2 = st.columns(2)

        # Extract coordinates for current video in latent space
        if artifact is not None and not df_features.empty:
            prep = artifact["preprocessor"]
            X_all = df_features[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
            y_all = df_features["label"].to_numpy(dtype=np.int64)
            X_compact_all = prep.transform(X_all)

            pca_2d = PCA(n_components=2, random_state=42)
            X_pca_all = pca_2d.fit_transform(X_compact_all)

            lda_1d = LinearDiscriminantAnalysis(n_components=1)
            X_lda_all = lda_1d.fit_transform(X_compact_all, y_all)

            cur_pca_coord = None
            cur_lda_coord = None
            if X_raw is not None:
                cur_comp = prep.transform(X_raw)
                cur_pca_coord = pca_2d.transform(cur_comp)[0]
                cur_lda_coord = lda_1d.transform(cur_comp)[0, 0]

            # Orientation check: guarantee Deceptive cluster is on the right (> 0) and Truthful on the left (< 0)
            if np.mean(X_lda_all[y_all == 1]) < np.mean(X_lda_all[y_all == 0]):
                X_lda_all = -1.0 * X_lda_all
                if cur_lda_coord is not None:
                    cur_lda_coord = -1.0 * cur_lda_coord

            with proj_col1:
                fig_p1, ax_p1 = plt.subplots(figsize=(6, 4.2), dpi=150)
                fig_p1.patch.set_facecolor('#0b0f19')
                ax_p1.set_facecolor('#131b2e')

                ax_p1.scatter(X_pca_all[y_all == 0, 0], X_pca_all[y_all == 0, 1], c='#2563eb', label='Truthful (y=0)', alpha=0.7, edgecolors='k', s=45)
                ax_p1.scatter(X_pca_all[y_all == 1, 0], X_pca_all[y_all == 1, 1], c='#ef4444', label='Deceptive (y=1)', alpha=0.7, edgecolors='k', s=45)

                if cur_pca_coord is not None:
                    ax_p1.scatter([cur_pca_coord[0]], [cur_pca_coord[1]], c='#facc15', s=220, marker='*', edgecolors='black', linewidth=1.5, label='★ Current Video', zorder=10)

                ax_p1.set_title("Unsupervised 2D PCA Subspace", color='#f8fafc', fontsize=11, fontweight='bold')
                ax_p1.set_xlabel("Principal Component 1", color='#94a3b8', fontsize=9)
                ax_p1.set_ylabel("Principal Component 2", color='#94a3b8', fontsize=9)
                ax_p1.tick_params(colors='#cbd5e1', labelsize=8)
                for spine in ax_p1.spines.values():
                    spine.set_color('#334155')
                ax_p1.grid(True, linestyle=':', color='#334155', alpha=0.6)
                ax_p1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)
                fig_p1.tight_layout()
                st.pyplot(fig_p1)
                plt.close(fig_p1)

            with proj_col2:
                fig_p2, ax_p2 = plt.subplots(figsize=(6, 4.2), dpi=150)
                fig_p2.patch.set_facecolor('#0b0f19')
                ax_p2.set_facecolor('#131b2e')

                rng = np.random.RandomState(42)
                jitter = rng.normal(0, 0.08, size=len(X_lda_all))

                ax_p2.scatter(X_lda_all[y_all == 0, 0], jitter[y_all == 0], c='#2563eb', label='Truthful (y=0)', alpha=0.7, edgecolors='k', s=45)
                ax_p2.scatter(X_lda_all[y_all == 1, 0], jitter[y_all == 1], c='#ef4444', label='Deceptive (y=1)', alpha=0.7, edgecolors='k', s=45)

                if cur_lda_coord is not None:
                    ax_p2.scatter([cur_lda_coord], [0.0], c='#facc15', s=220, marker='*', edgecolors='black', linewidth=1.5, label='★ Current Video', zorder=10)

                ax_p2.axvline(0.0, color='#f8fafc', linestyle='--', linewidth=1.8, label='Optimal Bayes Boundary')
                ax_p2.set_title("Supervised Fisher's LDA Separation", color='#f8fafc', fontsize=11, fontweight='bold')
                ax_p2.set_xlabel("Fisher Canonical Coordinate (w^T x)", color='#94a3b8', fontsize=9)
                ax_p2.set_yticks([])
                ax_p2.tick_params(colors='#cbd5e1', labelsize=8)
                for spine in ax_p2.spines.values():
                    spine.set_color('#334155')
                ax_p2.grid(True, linestyle=':', color='#334155', alpha=0.6)
                ax_p2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)
                fig_p2.tight_layout()
                st.pyplot(fig_p2)
                plt.close(fig_p2)

        # Diagnostic Plots Showcase
        st.markdown("---")
        st.subheader("Published Diagnostic Curves & Weights")
        plot_c1, plot_c2, plot_c3 = st.columns(3)

        with plot_c1:
            p_roc = Path("outputs/plots/roc_curves.png")
            if p_roc.exists():
                render_image(st, str(p_roc), caption="Multi-Model ROC Curves Comparison")

        with plot_c2:
            p_cm = Path("outputs/plots/confusion_matrices.png")
            if p_cm.exists():
                render_image(st, str(p_cm), caption="Paradigm Confusion Matrices")

        with plot_c3:
            p_fi = Path("outputs/plots/feature_importance.png")
            if p_fi.exists():
                render_image(st, str(p_fi), caption="Domain Feature Importance & Learned Late-Fusion Weights")


if __name__ == "__main__":
    main()
