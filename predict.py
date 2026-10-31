from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Offline mode for huggingface to prevent timeout
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import numpy as np
import torch
import torch.nn as nn

from text_audio import TextAudioResult, analyze_text_audio
from utils import extract_audio, extract_frames
from vision import analyze_vision
from model import DeceptionMLP
from pr_pipeline import ALL_FEATURE_COLS


DEFAULT_FEATURE_COLS = [
    "text_score",
    "audio_score",
    "vision_score",
    "hesitation_score",
    "pitch_score",
    "silence_score",
    "blink_rate_per_min",
    "eye_movement_std",
    "blink_score",
    "eye_movement_score",
    "transcript_len",
    "has_transcript",
] + [f"emb_{i}" for i in range(384)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-Model Multimodal Deception Predictor")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model", default="all", help="Model to evaluate: 'all', 'EarlyFusion-SVM-Linear', 'LateFusion-BayesianPoE', 'Subspace-RegQDA', 'Deep-DeceptionMLP', 'Heuristic-Weighted', etc.")
    parser.add_argument("--model-path", default="outputs/model_full.pt", help="Path to saved DeceptionMLP .pt model")
    parser.add_argument("--registry-path", default="outputs/trained_models.joblib", help="Path to saved classical model registry")
    parser.add_argument("--work-dir", default="inference_artifacts", help="Directory to store extracted audio/frames")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model size")
    parser.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"], help="Whisper inference device")
    parser.add_argument("--frame-step", default=8, type=int, help="Extract every Nth frame")
    parser.add_argument("--max-audio-seconds", default=30.0, type=float, help="Analyze only first N seconds of audio (<=0 means full audio)")
    parser.add_argument("--vision-max-frames", default=120, type=int, help="Maximum sampled frames for vision analysis")
    parser.add_argument("--threshold", default=0.5, type=float, help="Decision threshold for lie class")
    parser.add_argument("--output-json", default="", help="Optional path to save prediction JSON")
    return parser


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[nn.Module, np.ndarray, np.ndarray, list[str]]:
    checkpoint: dict[str, Any] = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing model_state_dict")

    input_dim = int(checkpoint.get("input_dim", len(DEFAULT_FEATURE_COLS)))
    feature_cols = list(checkpoint.get("feature_cols", DEFAULT_FEATURE_COLS))

    scaler_mean = np.array(checkpoint.get("scaler_mean", [0.0] * input_dim), dtype=np.float32)
    scaler_scale = np.array(checkpoint.get("scaler_scale", [1.0] * input_dim), dtype=np.float32)

    if scaler_mean.shape[0] != input_dim or scaler_scale.shape[0] != input_dim:
        raise ValueError("Invalid checkpoint: scaler stats shape does not match input_dim")

    model = DeceptionMLP(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler_mean, scaler_scale, feature_cols


def extract_single_video_features(
    video_path: Path,
    work_dir: Path,
    frame_step: int,
    whisper_model: str,
    whisper_device: str,
    max_audio_seconds: float,
    vision_max_frames: int,
) -> dict[str, float]:
    sample_dir = work_dir / video_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    audio_path = sample_dir / "audio.wav"
    frames_dir = sample_dir / "frames"

    has_audio = True
    try:
        extract_audio(str(video_path), str(audio_path))
    except ValueError:
        has_audio = False

    _, fps = extract_frames(str(video_path), str(frames_dir), frame_step=frame_step)

    # Check transcript cache
    cache_path = Path("outputs/transcripts_cache.json")
    cached_transcript = None
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_transcript = json.load(f).get(video_path.name)
        except Exception:
            pass

    if has_audio:
        ta = analyze_text_audio(
            str(audio_path),
            model_size=whisper_model,
            whisper_device=whisper_device,
            max_audio_seconds=max_audio_seconds,
        )
        if cached_transcript and not ta.transcript:
            ta.transcript = cached_transcript
    else:
        ta = TextAudioResult(
            transcript=cached_transcript or "",
            hesitation_score=0.0,
            pitch_score=0.0,
            silence_score=0.0,
            text_score=0.0,
            audio_score=0.0,
        )

    vi = analyze_vision(str(frames_dir), video_fps=fps, frame_step=frame_step, max_frames=vision_max_frames)

    transcript_len = len(ta.transcript.split()) if ta.transcript else 0
    has_transcript = int(transcript_len > 0)

    result = {
        "text_score": float(ta.text_score),
        "audio_score": float(ta.audio_score),
        "vision_score": float(vi.vision_score),
        "hesitation_score": float(ta.hesitation_score),
        "pitch_score": float(ta.pitch_score),
        "silence_score": float(ta.silence_score),
        "blink_rate_per_min": float(vi.blink_rate_per_min),
        "eye_movement_std": float(vi.eye_movement_std),
        "blink_score": float(vi.blink_score),
        "eye_movement_score": float(vi.eye_movement_score),
        "transcript_len": float(transcript_len),
        "has_transcript": float(has_transcript),
        "transcript": ta.transcript,
    }

    for i, val in enumerate(ta.transcript_embedding):
        result[f"emb_{i}"] = float(val)

    return result


def main() -> None:
    args = build_parser().parse_args()

    video_path = Path(args.video_path)
    model_path = Path(args.model_path)
    registry_path = Path(args.registry_path)
    work_dir = Path(args.work_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    work_dir.mkdir(parents=True, exist_ok=True)

    feats = extract_single_video_features(
        video_path=video_path,
        work_dir=work_dir,
        frame_step=args.frame_step,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        max_audio_seconds=args.max_audio_seconds,
        vision_max_frames=args.vision_max_frames,
    )

    all_predictions: dict[str, dict[str, Any]] = {}

    # 1. Deep-DeceptionMLP Prediction
    if model_path.exists():
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, scaler_mean, scaler_scale, feature_cols = load_checkpoint(model_path, device)
            x = np.array([feats.get(col, 0.0) for col in feature_cols], dtype=np.float32)
            safe_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
            x_scaled = (x - scaler_mean) / safe_scale
            with torch.no_grad():
                logit = model(torch.from_numpy(x_scaled).unsqueeze(0).to(device)).item()
            prob_mlp = float(1.0 / (1.0 + np.exp(-logit)))
            pred_mlp = int(prob_mlp >= args.threshold)
            all_predictions["Deep-DeceptionMLP"] = {
                "prob_deceptive": prob_mlp,
                "verdict": "DECEPTIVE" if pred_mlp == 1 else "TRUTHFUL",
            }
        except Exception as e:
            logger.warning("Could not run DeceptionMLP: %s", e)

    # 2. Baseline Heuristic
    ts = feats.get("text_score", 0.5)
    as_ = feats.get("audio_score", 0.5)
    vs = feats.get("vision_score", 0.5)
    prob_heur = float(np.clip(0.3 * ts + 0.3 * as_ + 0.4 * vs, 0.0, 1.0))
    all_predictions["Heuristic-Weighted"] = {
        "prob_deceptive": prob_heur,
        "verdict": "DECEPTIVE" if prob_heur >= args.threshold else "TRUTHFUL",
    }

    # 3. Classical Pattern Recognition Models from Registry
    if registry_path.exists():
        try:
            artifact = joblib.load(registry_path)
            prep = artifact["preprocessor"]
            models = artifact["models"]
            X_raw = np.array([feats.get(col, 0.0) for col in ALL_FEATURE_COLS], dtype=np.float32).reshape(1, -1)
            X_compact = prep.transform(X_raw)

            for name, entry in models.items():
                clf = entry["model"]
                mtype = entry.get("type")
                if mtype == "late_soft":
                    prob = float(clf.soft_vote_predict_proba(X_compact)[0, 1])
                elif mtype == "late_poe":
                    prob = float(clf.product_of_experts_predict_proba(X_compact)[0, 1])
                elif hasattr(clf, "predict_proba"):
                    sub_pca = entry.get("subspace_pca")
                    if sub_pca is not None:
                        X_sub = sub_pca.transform(X_compact)
                        prob = float(clf.predict_proba(X_sub)[0, 1])
                    else:
                        prob = float(clf.predict_proba(X_compact)[0, 1])
                else:
                    prob = 0.50

                prob = float(np.clip(prob, 0.0, 1.0))
                all_predictions[name] = {
                    "prob_deceptive": prob,
                    "verdict": "DECEPTIVE" if prob >= args.threshold else "TRUTHFUL",
                }
        except Exception as e:
            logger.warning("Could not evaluate classical model registry: %s", e)

    # 4. Ensemble Consensus
    top_models = [
        "EarlyFusion-GradBoost",
        "EarlyFusion-RandomForest",
        "EarlyFusion-ExtraTrees",
        "EarlyFusion-SVM-Linear",
        "Subspace-RegQDA",
        "LateFusion-BayesianPoE",
        "Deep-DeceptionMLP",
    ]
    candidates = [all_predictions[m]["prob_deceptive"] for m in top_models if m in all_predictions]
    if candidates:
        ens_prob = float(np.mean(candidates))
        all_predictions["★ ENSEMBLE CONSENSUS"] = {
            "prob_deceptive": ens_prob,
            "verdict": "DECEPTIVE" if ens_prob >= args.threshold else "TRUTHFUL",
        }

    # Filter by model if requested
    target_model = args.model
    if target_model in ["consensus", "ensemble", "Ensemble"]:
        target_model = "★ ENSEMBLE CONSENSUS"

    if target_model != "all" and target_model in all_predictions:
        primary_pred = all_predictions[target_model]
    else:
        primary_pred = all_predictions.get("★ ENSEMBLE CONSENSUS", all_predictions.get("LateFusion-BayesianPoE", next(iter(all_predictions.values()))))

    result = {
        "video_path": str(video_path),
        "primary_model": target_model,
        "prediction": primary_pred,
        "multi_model_predictions": all_predictions,
        "transcript": feats.get("transcript", ""),
        "modality_scores": {
            "text": ts,
            "audio": as_,
            "vision": vs,
        },
        "features": {k: v for k, v in feats.items() if not k.startswith("emb_") and k != "transcript"},
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print Formatted CLI Table
    print("\n" + "=" * 76)
    print(f"MULTIMODAL DECEPTION PREDICTIONS: {video_path.name}")
    print("=" * 76)
    print(f"{'Model Architecture':<30} {'Deception Prob.':<20} {'Verdict'}")
    print("-" * 76)
    for mname, pred in all_predictions.items():
        prob_str = f"{pred['prob_deceptive'] * 100:.1f}%"
        print(f"{mname:<30} {prob_str:<20} {pred['verdict']}")
    print("=" * 76)
    if feats.get("transcript"):
        print(f"\nTranscript: \"{feats.get('transcript')}\"\n")


if __name__ == "__main__":
    main()
