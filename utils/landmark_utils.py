# ========================================
# LANDMARK UTILITIES
# ========================================
# Core utilities for dlib/MediaPipe landmarks, eye cropping, head pose (PnP),
# gaze estimation. Supports both 68-pt dlib and full MediaPipe FaceMesh.

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

# ========================================
# MEDIAFACE MESH SETUP
# ========================================
mp_face_mesh = mp.solutions.face_mesh

# High accuracy tracker (iris refinement for gaze)
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Iris landmarks for precise gaze/eyes
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================================
# EYE LANDMARK INDICES
# ========================================
# MediaPipe full eye contours (iris + eyelid)
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def eye_aspect_ratio(eye_coords: List[Tuple[int, int]]) -> float:
    """Standard Eye Aspect Ratio (EAR) using 6-point formula."""
    if len(eye_coords) < 6:
        return 1.0
    p1, p2, p3, p4, p5, p6 = eye_coords
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (A + B) / (2.0 * C)
    return ear

def detect_landmarks(image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Detect landmarks using MediaPipe (returns pixel coords or None)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

def shape_to_coords(landmarks) -> List[Tuple[int, int]]:
    """Convert dlib shape or return pixel landmarks."""
    if hasattr(landmarks, 'num_parts') and landmarks.num_parts == 68:
        return [(int(landmarks.part(i).x), int(landmarks.part(i).y)) for i in range(68)]
    return landmarks

def get_left_eye(landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Left eye landmarks (dlib or MP indices based on len)."""
    if len(landmarks) <= 70:  # dlib ~68 pts
        idx = [36, 37, 38, 39, 40, 41]
    else:
        idx = LEFT_EYE_IDX[:6]
    return [landmarks[i] for i in idx]

def get_right_eye(landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Right eye landmarks (dlib or MP indices based on len)."""
    if len(landmarks) <= 70:  # dlib ~68 pts
        idx = [42, 43, 44, 45, 46, 47]
    else:
        idx = RIGHT_EYE_IDX[:6]
    return [landmarks[i] for i in idx]

def crop_eye(image: np.ndarray, eye_points: List[Tuple[int, int]], margin=10, size=(64,64)) -> np.ndarray:
    """Enhanced eye crop: percentile bounds, CLAHE, grayscale, cubic resize."""
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
    crop = image[y1:y2, x1:x2].copy()
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    crop = clahe.apply(crop)
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    return crop

def eye_aspect_ratio_vertical(eye_coords: List[Tuple[int, int]]) -> float:
    """Improved vertical ratio."""
    if len(eye_coords) < 6:
        return 1.0
    ys = [p[1] for p in eye_coords]
    upper = min(ys[:3])
    lower = max(ys[3:6])
    v_dist = lower - upper
    eye_h = max(ys) - min(ys)
    return v_dist / eye_h if eye_h > 0 else 1.0

def head_pose(landmarks: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Enhanced dlib-optimized PnP with 12 key points for stability.
    """
    if len(landmarks) < 12:
        return 0.0, 0.0, 0.0
    
    # Dlib 68-pt optimized indices: nose bridge, chin, eyes, mouth corners, jaw
    dlib_indices = [30, 8, 36, 45, 17, 21, 54, 62, 27, 33, 51, 57]  # Better coverage
    model_points = np.array([
        (0.0,      0.0,     0.0),      # Nose bridge
        (0.0,     -330.0,  -65.0),     # Chin
        (-225.0,  170.0,  -135.0),     # Left eye outer
        (225.0,   170.0,  -135.0),     # Right eye outer
        (-150.0, -150.0,  -125.0),     # Left mouth
        (150.0,  -150.0,  -125.0),     # Right mouth
        (-120.0,  120.0,   -20.0),     # Left jaw
        (120.0,   120.0,   -20.0),     # Right jaw
        (0.0,     100.0,   -50.0),     # Forehead
        (-80.0,   20.0,    -60.0),     # Left brow
        (-60.0,  -80.0,   -110.0),     # Left lip
        (60.0,   -80.0,   -110.0)      # Right lip
    ], dtype="double")
    
    image_points = np.array([landmarks[i % len(landmarks)] for i in dlib_indices[:len(landmarks)]], dtype="double")
    
    h, w = image_shape[:2]
    focal_length = w
    center = (w / 2., h / 2.)
    camera_matrix = np.array([
        [focal_length*0.9, 0,      center[0]],
        [0,               focal_length*0.9, center[1]],
        [0,               0,      1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))
    
    success, rvec, tvec = cv2.solvePnP(model_points[:len(image_points)], image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0
        
    rmat, _ = cv2.Rodrigues(rvec)
    euler = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw, roll = euler[0]*0.8, euler[2]*1.2, euler[1]  # Calibrated scaling
    
    # Pose stability (variance penalty)
    pose_var = abs(yaw**2 + pitch**2 + roll**2) / 1000
    yaw = yaw / (1 + pose_var*0.1)
    
    return float(yaw), float(pitch), float(roll)

def mouth_aspect_ratio(landmarks: List[Tuple[int, int]]) -> float:
    """
    Enhanced MAR using full dlib lip contour (20 pts).
    """
    if len(landmarks) < 68:
        return 0.0
    
    # Full outer lips: upper 51-57, lower 57-65 (dlib standard)
    upper_lip = landmarks[50:53] + landmarks[61:64]  # Simplified key pts
    lower_lip = landmarks[56:59] + landmarks[65:68]
    
    if len(upper_lip) < 3 or len(lower_lip) < 3:
        return 0.0
    
    upper_y = np.mean([p[1] for p in upper_lip])
    lower_y = np.mean([p[1] for p in lower_lip])
    mouth_height = abs(lower_y - upper_y)
    
    left_x = min(p[0] for p in upper_lip + lower_lip)
    right_x = max(p[0] for p in upper_lip + lower_lip)
    mouth_width = right_x - left_x
    
    mar = mouth_height / mouth_width if mouth_width > 0 else 0.0
    return mar

def landmark_quality(landmarks: List[Tuple[int, int]]) -> float:
    """
    Dlib shape fitting quality: bilateral symmetry + compactness.
    """
    if len(landmarks) < 68:
        return 0.5
    
    left_face = landmarks[0:17] + landmarks[26:36]  # Left profile
    right_face = landmarks[17:26] + landmarks[45:55]  # Right profile
    
    left_var = np.var([p[0] for p in left_face])
    right_var = np.var([p[0] for p in right_face])
    symmetry = 1.0 / (1.0 + abs(left_var - right_var)/100)
    
    # Compactness (face bounding box aspect)
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    bbox_w, bbox_h = max(xs)-min(xs), max(ys)-min(ys)
    compact = 1.0 / (1.0 + abs(bbox_w/bbox_h - 1.2))  # ~1.2 ideal
    
    return 0.6 * symmetry + 0.4 * compact

def eye_gaze_offset(eye_coords: List[Tuple[int, int]]) -> float:
    """
    Enhanced gaze offset using iris approximation (inner corner bias).
    Improved for drowsiness: large |offset| indicates distraction/drowsiness.
    """
    if len(eye_coords) < 6:
        return 0.0
    coords = np.array(eye_coords)
    cx = np.mean(coords[:, 0])
    
    # Iris proxy: weighted inner points (for dlib 6pt: p3,p2 center bias)
    inner_weights = np.array([0.1, 0.4, 0.4, 0.4, 0.4, 0.1])  # center heavy
    px = np.average(coords[:, 0], weights=inner_weights[:len(coords)])
    
    eye_w = coords[:, 0].max() - coords[:, 0].min()
    if eye_w < 1.0:
        return 0.0
    offset = (px - cx) / (eye_w / 3)
    return np.clip(offset, -1.2, 1.2)

# Disabled dlib fallback to avoid errors (requires dlib + 'shape_predictor_68_face_landmarks.dat')
# See instructions if needed.

