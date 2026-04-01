import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe FaceMesh config
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def eye_aspect_ratio(eye_coords: List[Tuple[int, int]]) -> float:
    """Standard EAR."""
    if len(eye_coords) < 6:
        return 1.0
    p1, p2, p3, p4, p5, p6 = eye_coords
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (A + B) / (2.0 * C)
    return ear

def landmarks_to_coords(landmarks: List, image_shape: Tuple[int, int] = None) -> List[Tuple[int, int]]:
    """Convert normalized landmarks to pixel coords."""
    if image_shape is None:
        h, w = 480, 640
    else:
        h, w = image_shape[:2]
    if landmarks and isinstance(landmarks[0], tuple):
        return landmarks  # already pixel
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

def shape_to_coords(landmarks: List) -> List[Tuple[int, int]]:
    """Alias."""
    return landmarks_to_coords(landmarks)

def get_left_eye(landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Left eye."""
    return [landmarks[i] for i in LEFT_EYE_IDX]

def get_right_eye(landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Right eye."""
    return [landmarks[i] for i in RIGHT_EYE_IDX]

def crop_eye(image: np.ndarray, eye_points: List[Tuple[int, int]], margin=10, size=(64,64)) -> np.ndarray:
    """Eye crop with enhancement."""
    if len(eye_points) < 3:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    x1 = max(0, int(np.percentile(xs, 5)) - margin)
    y1 = max(0, int(np.percentile(ys, 5)) - margin)
    x2 = min(image.shape[1], int(np.percentile(xs, 95)) + margin)
    y2 = min(image.shape[0], int(np.percentile(ys, 95)) + margin)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    crop = clahe.apply(crop)
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    return crop

def eye_aspect_ratio_vertical(eye_coords: List[Tuple[int, int]]) -> float:
    """Vertical EAR variant."""
    if len(eye_coords) < 6:
        return 1.0
    ys = [p[1] for p in eye_coords]
    upper = min(ys[:3])
    lower = max(ys[3:])
    v_dist = lower - upper
    eye_h = max(ys) - min(ys)
    return v_dist / eye_h if eye_h > 0 else 1.0

def head_pose(landmarks: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """Head pose from pixel landmarks."""
    if len(landmarks) < 6:
        return 0.0, 0.0, 0.0
    idx = [1, 152, 33, 362, 61, 291]
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")
    image_points = np.array([landmarks[i] for i in idx if i < len(landmarks)], dtype="double")
    if len(image_points) < 4:
        return 0.0, 0.0, 0.0
    h, w = image_shape[:2]
    focal_length = w
    center = (w / 2., h / 2.)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))
    success, rvec, tvec = cv2.solvePnP(model_points[:len(image_points)], image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    euler = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw, roll = euler[0], euler[2], euler[1]
    return float(yaw), float(pitch), float(roll)

def eye_gaze_offset(eye_coords: List[Tuple[int, int]]) -> float:
    """Gaze offset."""
    if len(eye_coords) < 6:
        return 0.0
    cx = sum(p[0] for p in eye_coords) / len(eye_coords)
    inner = eye_coords[:3]
    px = sum(p[0] for p in inner) / len(inner)
    eye_w = max(p[0] for p in eye_coords) - min(p[0] for p in eye_coords)
    if eye_w == 0:
        return 0.0
    offset = (px - cx) / (eye_w / 2)
    return np.clip(offset, -1.0, 1.0)

def detect_landmarks(image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Main entry: detect pixel landmarks."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

