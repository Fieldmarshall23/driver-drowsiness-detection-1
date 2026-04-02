# ========================================
# EAR ML CLASSIFIER TRAINING SCRIPT
# ========================================
# Trains scikit-learn RandomForest on EAR features (left/right/avg)
# Dataset: dataset/open_eyes/ (0) vs closed_eyes/ (1)
# Robust face detection: dlib (multi-scale) + OpenCV Haar fallback
# Output: models/ml_model.pkl

"""
Robust ML classifier training script (EAR-based)

This script collects Eye Aspect Ratio (EAR) features from images in the
`dataset/open_eyes` and `dataset/closed_eyes` folders, trains a
scikit-learn RandomForest classifier, and saves the trained model to
the `models/` directory.

It resolves paths relative to the project root so the script can be
invoked from any working directory.
"""


from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import dlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure project root is on sys.path so `utils` can be imported when script is run directly
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.landmark_utils import shape_to_coords, get_left_eye, get_right_eye
from utils.eye_aspect_ratio import eye_aspect_ratio


LOGGER = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_paths(dataset_dir: str | Path, out_path: str | Path) -> Tuple[Path, Path, Path]:
    project_root = get_project_root()
    dataset = Path(dataset_dir) if Path(dataset_dir).is_absolute() else project_root / Path(dataset_dir)
    out = Path(out_path) if Path(out_path).is_absolute() else project_root / Path(out_path)
    predictor = project_root / "shape_predictor_68_face_landmarks.dat"
    return dataset, out, predictor


def collect_ear_labels(dataset_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    dataset, _, predictor_path = resolve_paths(dataset_dir, "models/ml_model.pkl")

    LOGGER.info("Using dataset dir: %s", dataset)
    LOGGER.info("Expecting predictor at: %s", predictor_path)

    if not predictor_path.exists():
        raise FileNotFoundError(
            f"Dlib predictor not found at {predictor_path}.\n"
            "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract."
        )

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    X = []
    y = []
    files_checked = 0
    faces_found = 0

    for label_name, label_val in (("open_eyes", 0), ("closed_eyes", 1)):
        label_dir = dataset / label_name
        if not label_dir.exists():
            LOGGER.warning("Label directory missing: %s", label_dir)
            continue
        for img_path in sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.jpeg")) + sorted(label_dir.glob("*.png")):
            files_checked += 1
            img = cv2.imread(str(img_path))
            if img is None:
                LOGGER.warning("Failed to read image: %s", img_path)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Try dlib detector with increasing upsample levels
            rects = detector(gray, 0)
            if len(rects) == 0:
                rects = detector(gray, 1)
            if len(rects) == 0:
                rects = detector(gray, 2)

            used_fallback = False
            if len(rects) == 0:
                # Fallback to OpenCV Haar cascade if available
                haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if Path(haar_path).exists():
                    face_cascade = cv2.CascadeClassifier(haar_path)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        rects = [dlib.rectangle(int(fx), int(fy), int(fx + fw), int(fy + fh))]
                        used_fallback = True

            if len(rects) == 0:
                # no face detected
                continue

            faces_found += 1
            shape = predictor(gray, rects[0])
            # ensure shape has 68 points
            try:
                _ = shape.part(67)
            except Exception:
                LOGGER.warning("Invalid shape returned for %s", img_path)
                continue

            coords = shape_to_coords(shape)
            left = get_left_eye(coords)
            right = get_right_eye(coords)

            try:
                ear_l = eye_aspect_ratio(left)
                ear_r = eye_aspect_ratio(right)
            except Exception as exc:
                LOGGER.warning("EAR calc failed for %s: %s", img_path, exc)
                continue

            ear = (ear_l + ear_r) / 2.0
            X.append([ear_l, ear_r, ear])
            y.append(label_val)

    LOGGER.info("Collected: files_checked=%d faces_found=%d samples=%d", files_checked, faces_found, len(X))
    if len(X) == 0:
        return np.empty((0, 3)), np.empty((0,))

    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_and_save(dataset_dir: str | Path, out_path: str | Path, test_size: float = 0.2, random_state: int = 42) -> None:
    X, y = collect_ear_labels(dataset_dir)
    if X.shape[0] == 0:
        LOGGER.error("No training samples found. Check dataset and predictor.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=100)
    LOGGER.info("Training RandomForest on %d samples...", X_train.shape[0])
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    LOGGER.info("Accuracy: %s", accuracy_score(y_test, preds))
    LOGGER.info("Classification report:\n%s", classification_report(y_test, preds))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    LOGGER.info("Saved model to %s", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train EAR-based RandomForest classifier")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder (contains open_eyes/ and closed_eyes/)")
    parser.add_argument("--out", default="models/ml_model.pkl", help="Output path for the trained model")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        train_and_save(args.dataset, args.out)
    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
