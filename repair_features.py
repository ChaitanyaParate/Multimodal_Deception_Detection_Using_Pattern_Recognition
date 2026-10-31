"""
Repair Features Script
======================
Repairs missing / zero blink rates and vision scores in outputs/features_full.csv
caused by coarse frame-step sampling. Uses dense frame sampling and debounced
MediaPipe FaceMesh tracking to accurately extract:
- blink_rate_per_min
- blink_score
- eye_movement_std
- eye_movement_score
- vision_score
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from utils import minmax_normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]


def extract_dense_vision_signals(video_path: str, max_frames: int = 450) -> dict[str, float] | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return None

    step = max(1, total_frames // max_frames) if total_frames > max_frames else 1

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ear_list: list[float] = []
    pupil_x_list: list[float] = []
    last_valid_ear = 0.28
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                pts = np.array([(lm.x * w, lm.y * h) for lm in res.multi_face_landmarks[0].landmark], dtype=np.float32)
                
                # Left & Right EAR
                v1_l = np.linalg.norm(pts[160] - pts[144])
                v2_l = np.linalg.norm(pts[158] - pts[153])
                h_l = np.linalg.norm(pts[33] - pts[133])
                ear_l = (v1_l + v2_l) / (2.0 * max(h_l, 1e-5))

                v1_r = np.linalg.norm(pts[385] - pts[380])
                v2_r = np.linalg.norm(pts[387] - pts[373])
                h_r = np.linalg.norm(pts[362] - pts[263])
                ear_r = (v1_r + v2_r) / (2.0 * max(h_r, 1e-5))

                raw_ear = float((ear_l + ear_r) / 2.0)
                if raw_ear >= 0.08:
                    val = raw_ear
                    last_valid_ear = raw_ear
                else:
                    val = last_valid_ear
                ear_list.append(val)

                # Pupil x
                if len(pts) > 473:
                    pupil_x = float((pts[468][0] + pts[473][0]) / 2.0)
                else:
                    pupil_x = float((pts[33][0] + pts[362][0]) / 2.0)
                pupil_x_list.append(pupil_x)
            else:
                ear_list.append(last_valid_ear)
        frame_idx += 1

    cap.release()
    face_mesh.close()

    ear_arr = np.array(ear_list, dtype=np.float32)
    if len(ear_arr) < 5:
        return None

    baseline = float(np.median(ear_arr))
    thresh = baseline * 0.75

    # Debounce blinks
    blink_count = 0
    in_blink = False
    cur_len = 0
    for val in ear_arr:
        if val < thresh:
            if not in_blink:
                in_blink = True
                cur_len = 1
            else:
                cur_len += 1
        else:
            if in_blink:
                if 1 <= cur_len <= 18:
                    blink_count += 1
                in_blink = False

    duration_sec = total_frames / fps
    blink_rate_per_min = (blink_count / max(duration_sec, 0.1)) * 60.0

    if len(pupil_x_list) > 2:
        eye_movement_std = float(np.std(np.diff(pupil_x_list)))
    else:
        eye_movement_std = 3.5

    blink_score = float(minmax_normalize(blink_rate_per_min, 8.0, 35.0))
    eye_movement_score = float(minmax_normalize(eye_movement_std, 1.5, 10.0))
    vision_score = float((blink_score + eye_movement_score) / 2.0)

    return {
        "blink_rate_per_min": blink_rate_per_min,
        "blink_score": blink_score,
        "eye_movement_std": eye_movement_std,
        "eye_movement_score": eye_movement_score,
        "vision_score": vision_score,
    }


def main() -> None:
    csv_path = Path("outputs/features_full.csv")
    if not csv_path.exists():
        logger.error("outputs/features_full.csv not found!")
        return

    df = pd.read_csv(csv_path)
    logger.info("Total rows in features_full.csv: %d", len(df))

    # Identify rows needing repair
    need_repair = df[(df["blink_rate_per_min"] == 0.0) | (df["vision_score"] == 0.0)]
    logger.info("Rows needing repair: %d", len(need_repair))

    repaired_count = 0
    for idx, row in need_repair.iterrows():
        vpath = row["video_path"]
        if not os.path.exists(vpath):
            logger.warning("File does not exist: %s", vpath)
            continue

        logger.info("Repairing row %d: %s...", idx, Path(vpath).name)
        signals = extract_dense_vision_signals(vpath)
        if signals and signals["blink_rate_per_min"] > 0:
            for k, v in signals.items():
                df.at[idx, k] = v
            repaired_count += 1
            logger.info("  -> Done: %s: blinks/min=%.1f, vision_score=%.2f",
                        Path(vpath).name, signals["blink_rate_per_min"], signals["vision_score"])
        else:
            # Fallback to class median when face tracking fails / no blinks detected
            label = int(row["label"])
            subset = df[(df["label"] == label) & (df["blink_rate_per_min"] > 0)]
            med_blink = float(subset["blink_rate_per_min"].median() if not subset.empty else 18.0)
            med_std = float(subset["eye_movement_std"].median() if not subset.empty else 4.0)
            b_sc = float(minmax_normalize(med_blink, 8.0, 35.0))
            e_sc = float(minmax_normalize(med_std, 1.5, 10.0))
            df.at[idx, "blink_rate_per_min"] = med_blink
            df.at[idx, "eye_movement_std"] = med_std
            df.at[idx, "blink_score"] = b_sc
            df.at[idx, "eye_movement_score"] = e_sc
            df.at[idx, "vision_score"] = (b_sc + e_sc) / 2.0
            repaired_count += 1
            logger.info("  -> Fallback to class %d median: blinks=%.1f, vision_score=%.2f", label, med_blink, (b_sc + e_sc) / 2.0)

    # Save repaired dataset
    df.to_csv(csv_path, index=False)
    logger.info("Successfully updated %s! Repaired %d rows.", csv_path, repaired_count)
    logger.info("Zero blink rows remaining: %d", (df["blink_rate_per_min"] == 0.0).sum())


if __name__ == "__main__":
    main()
