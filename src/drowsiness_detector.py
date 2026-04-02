# ========================================
# IMPORTS AND SETUP
# ========================================
# Core libraries for CV/ML/audio/logging
import os
import sys
import cv2
import dlib
import time
import pickle
import platform
import threading
import numpy as np
import datetime
from pathlib import Path
import warnings

# Add project root to path for utils imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TensorFlow for CNN inference
import tensorflow as tf

# Suppress non-critical TF warnings for cleaner output
warnings.filterwarnings('ignore', message='.*All `Callback`.*')
warnings.filterwarnings('ignore', message='.*glibc.*')
warnings.filterwarnings('ignore', message='.*unrecognized shape.*')

# Custom utils for landmark processing and EAR calculation
from utils.landmark_utils import (shape_to_coords, get_left_eye, get_right_eye, crop_eye, 
                                  head_pose, eye_gaze_offset, detect_landmarks)
from utils.eye_aspect_ratio import fused_ear
from utils.blink_utils import BlinkDetector
from utils.landmark_utils import mouth_aspect_ratio, landmark_quality


# Optional audio library (fallback to Windows winsound if unavailable)
try:
    from playsound import playsound
except Exception:
    playsound = None


# ========================================
# PATHS AND LOGGING SETUP
# ========================================
# Project directories (ensures cross-platform compatibility)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PREDICTOR_PATH = os.path.join(PROJECT_ROOT, 'shape_predictor_68_face_landmarks.dat')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
SAVE_DEBUG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'debug_mouth')
ALERT_LOG = os.path.join(PROJECT_ROOT, 'outputs', 'alerts.log')

# Create required directories
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(SAVE_DEBUG_DIR, exist_ok=True)

# ========================================
# LOGGING CONFIGURATION
# ========================================
# Persistent logging for alerts/debug (appends to file)
import logging
logging.basicConfig(level=logging.INFO, filename=ALERT_LOG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='a')
logger = logging.getLogger(__name__)


DROWSY_SOUND_PATH = os.path.join(ASSETS_DIR, 'drowsy_alert.mp3')
CONCENTRATE_SOUND_PATH = os.path.join(ASSETS_DIR, 'concentrate_alert.mp3')

# Global counters and state (step 4)
ear_counter = 0
eye_cnn_counter = 0
yawn_counter = 0
side_start_time = None
last_audio_alert = 0.0
frame_counter = 0
perclos_counter = 0
ear_baseline = 0.27
baseline_collected = False
ema_drowsy_prob = 0.0
ema_eye_prob = 0.5
ema_yawn_prob = 0.0
blink_detector = BlinkDetector(ear_thresh=0.22, fps=15)
heavy_process_frame = 0
last_fps_time = 0.0
fps = 0.0
# Head pose EMA (step 4)
ema_yaw = 0.0
ema_pitch = 0.0
ema_roll = 0.0
head_pose_frame = 0

# Constants (restored from original)
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 12  # TEMP LOW for debug
EYE_CNN_CLOSE_THRESH = 0.5
EYE_CNN_CONSEC = 8      # Increased
YAWN_THRESH = 0.5
YAWN_CONSEC = 3
SIDE_LOOK_THRESH = 2.0
HEAD_YAW_THRESH = 30.0  # degrees
GAZE_OFFSET_THRESH = 0.4
SAVE_DEBUG_THRESH = 0.4
ALERT_AUDIO_COOLDOWN = 3.0
DROWSY_PROB_THRESH = 0.50  # TEMP LOWERED FOR DEBUG
DROWSY_RESET_THRESH = 0.30
# New improved constants
PERCLOS_WINDOW = 300
EMA_ALPHA = 0.3
FUSION_WEIGHTS = [0.22, 0.22, 0.18, 0.13, 0.1, 0.05, 0.1]  # EAR, eye, yawn, ml, perclos, gaze, blink
EAR_BASELINE_FRAMES = 100

# Initialize directories
os.makedirs(SAVE_DEBUG_DIR, exist_ok=True)

# Load models
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(
        f"Dlib predictor file not found at {PREDICTOR_PATH}.\n"
        "Please download `shape_predictor_68_face_landmarks.dat` from http://dlib.net/files/"
        " and place it in the project root or update PREDICTOR_PATH."
    )

try:
    eye_cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'))
    print("[MODEL] Eye CNN loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load eye_cnn 'cnn_model.h5': {e}")
    eye_cnn = None
    logger.error(f"Eye CNN load failed: {e}")

# Load mouth CNN model
try:
    mouth_cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'mouth_cnn_model.h5'))
    print("[MODEL] Mouth CNN loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load mouth_cnn 'mouth_cnn_model.h5': {e}")
    mouth_cnn = None
    logger.error(f"Mouth CNN load failed: {e}")

EAR_MODEL_PATH = os.path.join(MODEL_DIR, 'ml_model.pkl')
try:
    with open(EAR_MODEL_PATH, 'rb') as f:
        ear_model = pickle.load(f)
    print("[MODEL] EAR ML model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load ear_model 'ml_model.pkl': {e}")
    ear_model = None
    logger.error(f"EAR model load failed: {e}")

# Removed TTS dependency - using beep alerts only
TTS_AVAILABLE = False

counter_lock = threading.Lock()

def play_drowsy_alert():
    """Non-blocking beep pattern for drowsy alert."""
    logger.info("Drowsy alert triggered")
    print("[AUDIO] Drowsy alert - BEEP PATTERN")
    if platform.system() == "Windows":
        try:
            import winsound
            # Triple beep: urgent pattern
            winsound.Beep(800, 300)
            time.sleep(0.1)
            winsound.Beep(700, 300)
            time.sleep(0.1)
            winsound.Beep(900, 500)
        except Exception as e:
            logger.error(f"Winsound failed: {e}")

def play_concentrate_alert():
    """Non-blocking beep for concentrate alert."""
    logger.info("Concentrate alert triggered")
    print("[AUDIO] Concentrate alert - BEEP")
    if platform.system() == "Windows":
        try:
            import winsound
            # Double high beep: attention
            winsound.Beep(1200, 400)
            time.sleep(0.1)
            winsound.Beep(1400, 400)
        except Exception as e:
            logger.error(f"Winsound failed: {e}")

# Removed safe_tts_say - no longer needed

def preprocess_rgb(img):
    """Preprocess for RGB models (eyes) - outputs (1,64,64,3)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    norm = resized.astype('float32') / 255.0
    norm = np.expand_dims(norm, axis=(0, -1))
    norm = np.repeat(norm, 3, axis=-1)  # Repeat to RGB (1,64,64,3) for eye_cnn
    return norm

def preprocess_grayscale(img):
    """Preprocess for grayscale models (mouth) - outputs (1,64,64,1)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    norm = resized.astype('float32') / 255.0
    norm = np.expand_dims(norm, axis=-1)  # Grayscale channel (64,64,1)
    norm = np.expand_dims(norm, axis=0)   # Batch dim (1,64,64,1)
    return norm

def crop_region(image, points, margin=10, size=(64,64)):
    """Crop region (mouth/eyes) safely with empty check."""
    if len(points) < 3:
        empty = np.zeros((*size, 3), dtype=np.uint8)
        return empty
        
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    h, w = image.shape[:2]
    x1 = max(0, minx - margin)
    y1 = max(0, miny - margin)
    x2 = min(w, maxx + margin)
    y2 = min(h, maxy + margin)
    
    if x2 <= x1 or y2 <= y1:
        empty = np.zeros((*size, 3), dtype=np.uint8)
        return empty
        
    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, size)
    return crop

def detect_drowsiness(frame, detector, predictor):
    global ear_counter, eye_cnn_counter, yawn_counter, side_start_time, last_audio_alert, frame_counter, ear_baseline, baseline_collected
    if frame is None:
        return None
    # Light preprocess always
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray_raw
    
    # Heavy preprocess only on CNN frames (step 3)
    if heavy_process_frame:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray_raw)
        gamma = 1.2
        gray = np.power(gray / 255.0, gamma) * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Dlib pyramid: improved small/occluded detection
    faces = detector(gray, 1)
    if len(faces) == 0:
        faces = detector(gray, 2)  # Higher pyramid upsampling (level 2) for small/occluded faces
    max_faces_scale = 1.2 if len(faces) > 0 else 1.0

    if len(faces) == 0:
        # Fast Haar fallback, no debug prints
        try:
            haar_path = os.path.join(PROJECT_ROOT, 'assets', 'haarcascade_frontalface_default.xml')
            if os.path.exists(haar_path):
                face_cascade = cv2.CascadeClassifier(haar_path)
            else:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_cv = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            for (x, y, w, h) in faces_cv:
                faces.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
        except Exception:
            pass

    alert_active = False
    
    num_faces = len(faces)
    if frame_counter % 10 == 0:  # More frequent
        if max_faces_scale >= 0:
            print(f"[DEBUG] Best detection scale: {max_faces_scale}")
        if num_faces == 0:
            print("[DEBUG] No faces")
        elif num_faces > 1:
            print(f"[DEBUG] Multiple faces ({num_faces})")
        else:
            print("[DEBUG] Single face")
    
    if num_faces == 0:
        if frame_counter % 120 == 0:
            noface_dir = os.path.join(PROJECT_ROOT, 'outputs', 'debug_no_face')
            os.makedirs(noface_dir, exist_ok=True)
            fname = os.path.join(noface_dir, f'no_face_f{frame_counter:06d}.jpg')
            if frame is not None:
                cv2.imwrite(fname, frame)
        return frame
        
    if num_faces > 1:
        face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
    else:
        face = faces[0]

    # Validate face size (too small = invalid)
    if face.width() < 50 or face.height() < 50:
        return frame

    # Dlib landmarks (primary)
    try:
        shape = predictor(gray, face)
        coords = shape_to_coords(shape)
        print(f"[DEBUG] Using dlib landmarks ({len(coords)} pts)")
    except Exception as e:
        print(f"[DEBUG] Dlib predictor failed: {e}. Falling back to MediaPipe.")
        coords = None

    left_eye = np.array(get_left_eye(coords))
    right_eye = np.array(get_right_eye(coords))

    global ema_yaw, ema_pitch, ema_roll, head_pose_frame
    # Head pose every 5th frame (step 4)
    if frame_counter % 5 == 0:
        yaw, pitch, roll = head_pose(coords, gray.shape)
        ema_yaw = EMA_ALPHA * yaw + (1 - EMA_ALPHA) * ema_yaw
        ema_pitch = EMA_ALPHA * pitch + (1 - EMA_ALPHA) * ema_pitch
        ema_roll = EMA_ALPHA * roll + (1 - EMA_ALPHA) * ema_roll
        head_pose_frame = frame_counter
    else:
        yaw = ema_yaw
        pitch = ema_pitch
        roll = ema_roll

    left_gaze = eye_gaze_offset(left_eye)
    right_gaze = eye_gaze_offset(right_eye)
    avg_gaze_offset = (left_gaze + right_gaze) / 2.0

    # Enhanced EAR with smoothing/quality (track history)
    global ear_history_l, ear_history_r
    if not hasattr(detect_drowsiness, 'ear_history_l'):
        detect_drowsiness.ear_history_l = []
        detect_drowsiness.ear_history_r = []
    
    left_ear, left_qual = fused_ear(left_eye.tolist(), detect_drowsiness.ear_history_l[-2:], ear_baseline)
    right_ear, right_qual = fused_ear(right_eye.tolist(), detect_drowsiness.ear_history_r[-2:], ear_baseline)
    ear_qual = min(left_qual, right_qual)
    ear = (left_ear + right_ear) / 2.0
    
    detect_drowsiness.ear_history_l.append(left_ear)
    detect_drowsiness.ear_history_r.append(right_ear)
    if len(detect_drowsiness.ear_history_l) > 10:
        detect_drowsiness.ear_history_l.pop(0)
        detect_drowsiness.ear_history_r.pop(0)
    
    # Enhanced blink with velocity/points
    blink_metrics = blink_detector.update(left_ear, right_ear, frame_counter,
                                        left_eye_points=left_eye.tolist(), right_eye_points=right_eye.tolist(),
                                        prev_left=detect_drowsiness.prev_left_eye if hasattr(detect_drowsiness, 'prev_left_eye') else None,
                                        prev_right=detect_drowsiness.prev_right_eye if hasattr(detect_drowsiness, 'prev_right_eye') else None)
    blink_rate = blink_metrics['blink_rate']
    freq_score = blink_metrics['freq_score']
    perclos_pct = blink_metrics.get('perclos_pct', 0.0)
    current_ear_thresh = blink_metrics.get('ear_thresh_adapt', ear_baseline * 0.8)
    
    # Store prev for next frame
    detect_drowsiness.prev_left_eye = left_eye.tolist()
    detect_drowsiness.prev_right_eye = right_eye.tolist()
    
    # Step 2: EAR baseline collection

    if not baseline_collected and frame_counter < EAR_BASELINE_FRAMES:
        if not hasattr(detect_drowsiness, 'ear_sum'):
            detect_drowsiness.ear_sum = 0.0
            detect_drowsiness.ear_count = 0
        detect_drowsiness.ear_sum += ear
        detect_drowsiness.ear_count += 1
        if detect_drowsiness.ear_count >= EAR_BASELINE_FRAMES:
            ear_baseline = detect_drowsiness.ear_sum / EAR_BASELINE_FRAMES
            baseline_collected = True
            print(f"[BASELINE] EAR baseline set to {ear_baseline:.3f}")

    eye_prob = 0.5
    yawn_prob = 0.0
    ml_pred = 0.0
    if heavy_process_frame:
        try:
            ml_pred = ear_model.predict_proba([[left_ear, right_ear, ear]])[:, 1][0]
            
            # Eye CNN
            left_eye_region = crop_eye(frame, list(left_eye))
            right_eye_region = crop_eye(frame, list(right_eye))
            left_input = preprocess_rgb(left_eye_region)
            right_input = preprocess_rgb(right_eye_region)
            left_prob = eye_cnn.predict(left_input, verbose=0)[0][0]
            right_prob = eye_cnn.predict(right_input, verbose=0)[0][0]
            eye_prob = (left_prob + right_prob) / 2.0
            
            # Mouth CNN + enhanced MAR
            mouth_region = crop_region(frame, coords[48:68])
            mouth_input = preprocess_grayscale(mouth_region)
            yawn_prob = mouth_cnn.predict(mouth_input, verbose=0)[0][0]
            
            # Enhanced full-contour MAR + quality gating
            mar = mouth_aspect_ratio(coords)
            shape_qual = landmark_quality(coords)
            mar_boost = max(0, (mar - 0.02)) * shape_qual
            yaw_adjust = max(0, abs(pitch)/45.0)  # Non-frontal penalty
            yawn_prob = max(0.0, min(1.0, yawn_prob * (1.0 + mar_boost) * (1.0 - yaw_adjust * 0.3)))
        except Exception:
            pass
    eye_closed = eye_prob >= EYE_CNN_CLOSE_THRESH
    
    # Step 3: EMA smoothing on eye/yawn (use ema in fusion)
    global ema_eye_prob, ema_yawn_prob
    ema_eye_prob = EMA_ALPHA * eye_prob + (1 - EMA_ALPHA) * ema_eye_prob
    ema_yawn_prob = EMA_ALPHA * yawn_prob + (1 - EMA_ALPHA) * ema_yawn_prob

    # Step 4: PERCLOS counter
    global perclos_counter
    if eye_closed:
        perclos_counter += 1
    else:
        perclos_counter = max(0, perclos_counter - 1)
    perclos = perclos_counter / PERCLOS_WINDOW

    # Step 4: Adaptive EAR threshold

    current_ear_thresh = ear_baseline * 0.8 if baseline_collected else EAR_THRESHOLD

    try:
        if yawn_prob >= SAVE_DEBUG_THRESH:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            fname = os.path.join(SAVE_DEBUG_DIR, f'mouth_{ts}.jpg')
            cv2.imwrite(fname, mouth_region)
            with open(ALERT_LOG, 'a', encoding='utf-8') as lf:
                lf.write(f"{ts}, MOUTH_CROP_SAVED, yawn_prob={yawn_prob:.3f}\n")
    except Exception:
        pass

    # Improved sideways detection: head yaw + gaze offset (more accurate than nose)
    side_looking = (abs(yaw) > HEAD_YAW_THRESH) or (abs(avg_gaze_offset) > GAZE_OFFSET_THRESH)
    if side_looking:
        if side_start_time is None:
            side_start_time = time.time()
        side_duration = time.time() - side_start_time
    else:
        side_start_time = None
        side_duration = 0.0

    # Drowsiness detection logic
    # EAR check
    with counter_lock:
        # Step 6: Smoother counters with decay (cap 50)
        if ear < current_ear_thresh:
            ear_counter = min(50, ear_counter * 1.02 + 1)
        else:
            ear_counter = max(0, ear_counter * 0.95)

        # Eye CNN check - smoother decay
        if eye_closed:
            eye_cnn_counter = min(50, eye_cnn_counter * 1.02 + 1)
        else:
            eye_cnn_counter = max(0, eye_cnn_counter * 0.95)

        # Yawn check - smoother decay
        if yawn_prob >= YAWN_THRESH:
            yawn_counter = min(50, yawn_counter * 1.02 + 1)
        else:
            yawn_counter = max(0, yawn_counter * 0.95)

        # Periodic reset if no recent alerts (prevent stuck loop)
        if frame_counter % 300 == 0 and not alert_active:
            ear_counter = max(0, ear_counter // 2)
            eye_cnn_counter = max(0, eye_cnn_counter // 2)
            yawn_counter = max(0, yawn_counter // 2)
            logger.info(f"Periodic counter reset: EAR={ear_counter}, EYE={eye_cnn_counter}, YAWN={yawn_counter}")

    # Step 5: High accuracy fusion with gaze (separate drowsy vs concentrate)
    ear_factor = 1.0 if ear < current_ear_thresh else 0.0
    eye_factor = ema_eye_prob
    yawn_factor = ema_yawn_prob
    ml_factor = ml_pred
    perclos_factor = perclos_pct if 'perclos_pct' in locals() else perclos
    gaze_factor = abs(avg_gaze_offset)
    
    # Drowsy prob (eyes/yawn heavy + blink + gaze)
    drowsy_factors = [ear_factor, eye_factor, yawn_factor, ml_factor, perclos_factor, gaze_factor, freq_score]
    drowsy_prob = sum(w * f for w, f in zip(FUSION_WEIGHTS, drowsy_factors))
    
    # Side/Concentrate factor
    concentrate_factor = min(1.0, side_duration / SIDE_LOOK_THRESH) * gaze_factor
    
    if frame_counter % 30 == 0:
        print(f"[DETECT] Frame{frame_counter} EAR:{ear:.2f} eye:{ema_eye_prob:.2f} yawn:{ema_yawn_prob:.2f} PERCLOS:{perclos:.3f} BLINK:{blink_rate:.2f} drowsy:{drowsy_prob:.2f} yaw:{yaw:.1f} gaze:{avg_gaze_offset:.2f} side:{side_duration:.1f}")

    # Update counters
    if drowsy_prob >= DROWSY_PROB_THRESH:
        ear_counter += 1
    elif drowsy_prob < DROWSY_RESET_THRESH:
        ear_counter = max(0, ear_counter - 1)
        eye_cnn_counter = max(0, eye_cnn_counter - 1)
        yawn_counter = max(0, yawn_counter - 1)

    # Alert conditions
    # High accuracy alerts: drowsy (0.7+ or counters), concentrate (side 2s+), focus (else green)
    # Multi-level alert escalation
    alert_level = 1
    if drowsy_prob >= 0.7 or ear_counter >= EAR_CONSEC_FRAMES * 1.5:
        alert_level = 3  # Critical: flashing, urgent audio
    elif drowsy_prob >= 0.55 or (ear_counter >= EAR_CONSEC_FRAMES or yawn_counter >= YAWN_CONSEC):
        alert_level = 2  # Urgent: thick border, multi-beep
    elif drowsy_prob >= DROWSY_PROB_THRESH or freq_score > 0.8 or side_duration > SIDE_LOOK_THRESH:
        alert_level = 1  # Warning: thin border, single beep
    
    is_drowsy = drowsy_prob >= DROWSY_PROB_THRESH or ear_counter >= EAR_CONSEC_FRAMES
    is_concentrate = side_duration >= SIDE_LOOK_THRESH
    alert_active = alert_level > 1 or is_concentrate
    
    # Escalating audio/pitch based on level/counter
    def play_escalating_alert(level, counter_total):
        if platform.system() != "Windows":
            return
        try:
            import winsound
            base_freq = 800 + (level-1)*200 + min(counter_total//10, 5)*100
            duration = 250 + level*100
            beeps = 1 + level
            for i in range(beeps):
                winsound.Beep(base_freq + i*100, duration)
                time.sleep(0.08)
        except Exception as e:
            logger.error(f"Escalating alert failed: {e}")
    
    if alert_active:
        print(f"[ALERT Lv{alert_level}] drowsy:{drowsy_prob:.2f} ear:{ear_counter} eye:{eye_cnn_counter} yawn:{yawn_counter} blink:{freq_score:.2f} side:{side_duration:.1f}s")
        
        total_counter = ear_counter + eye_cnn_counter + yawn_counter
        
        # Level-based messaging/border
        if is_drowsy and alert_level >= 2:
            drowsy_msgs = ["🚨🚨 CRITICAL DROWSY!", "🚨 PULL OVER IMMEDIATELY!", "🚨 EYES CLOSED - DANGER!"]
            msg = drowsy_msgs[frame_counter % 3]
            border_color = (0, 0, 255) if frame_counter % 10 < 5 else (50, 50, 50)  # Flash red/black
            border_thick = 10 + (alert_level-1)*2
            play_escalating_alert(alert_level, total_counter)
        elif is_concentrate:
            concentrate_msgs = ["⚠️ FOCUS ON ROAD!", "⚠️ EYES FORWARD NOW!", "⚠️ CONCENTRATE!"]
            msg = concentrate_msgs[frame_counter % 3]
            border_color = (0, 165, 255)  # Orange
            border_thick = 6
            play_escalating_alert(1, total_counter)
        else:
            msg = f"WARNING Lv{alert_level}"
            border_color = (0, 255, 255)  # Yellow
            border_thick = 4
            play_escalating_alert(1, total_counter)
        
        msg2 = f"DrowsyP:{drowsy_prob:.1f} Lv{alert_level} Cnt:{total_counter} BLK:{freq_score:.1f}"
        
        now = time.time()
        if now - last_audio_alert >= ALERT_AUDIO_COOLDOWN / alert_level:  # Shorter cooldown higher level
            last_audio_alert = now

        # Dynamic flashing border
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), border_color, border_thick)
        cv2.putText(frame, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0 + 0.2*(alert_level-1), border_color, 4)
        cv2.putText(frame, msg2, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
        
    else:
        # Green focused (eyes open, frontal gaze)
        if abs(yaw) < 15 and abs(avg_gaze_offset) < 0.3 and ear > current_ear_thresh * 0.9:
            msg = "👁️ FOCUSED ✓ ROAD WATCHED"
            cv2.putText(frame, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        else:
            msg = "ROAD FOCUS: OK"
            cv2.putText(frame, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]), (0,255,0), 2)

    # Enhanced viz with yaw/gaze
    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    
    # Optimized viz: core metrics only if changed (step 5)
    viz_y = frame.shape[0] - 160
    cv2.putText(frame, f"EAR:{ear:.2f} EyeP:{ema_eye_prob:.1f} YawnP:{ema_yawn_prob:.1f}", (20, viz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    viz_y -= 25
    cv2.putText(frame, f"Drowsy:{drowsy_prob:.1f} PERCLOS:{perclos_pct:.2f} MAR:{mar:.3f}", (20, viz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    viz_y -= 25
    cv2.putText(frame, f"Yaw:{yaw:.0f} Gaze:{avg_gaze_offset:+.2f} Side:{side_duration:.1f}s", (20, viz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

    return frame

def main():
    """Main function to run the drowsiness detector."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # Try multiple camera indices
    cap = None
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Using camera index {idx}")
            break
    if cap is None or not cap.isOpened():
        print("Error: Could not open any video source (tried 0,1). Check camera connection.")
        return
    
    # Optimize capture
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warmup camera pipeline
    print("Warming up camera...")
    for _ in range(30):
        ret, _ = cap.read()
        if ret:
            cv2.waitKey(1)
    print("Camera warmed up.")
    
    print("Drowsiness Detector started. Press 'q' to quit.")

    bad_frames = 0
    frame_skip = 1
    loop_start = time.time()
    skipped_frames = 0
    while True:
        if time.time() - loop_start > 300:
            logger.warning("Loop timeout - restarting cap")
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            loop_start = time.time()
        
        ret, frame = cap.read()
        skipped_frames += 1
        if skipped_frames % frame_skip != 0:
            continue  # Adaptive skip (step 8)
        # Robust frame validation
        if (not ret or frame is None or len(frame.shape) != 3 or 
            frame.shape[0] < 1 or frame.shape[1] < 1):
            bad_frames += 1
            logger.warning(f"Bad frame #{bad_frames}")
            if bad_frames > 50:
                logger.error("Too many bad frames - resetting camera")
                cap.release()
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                bad_frames = 0
            continue
        
        bad_frames = 0
        # Strict 640x480 always (step 6)
        h, w = frame.shape[:2]
        if h != 480 or w != 640:
            frame = cv2.resize(frame, (640, 480))
        
        global frame_counter, last_fps_time, fps
        frame_counter += 1
        now_time = time.time()
        if now_time - last_fps_time > 1.0:
            fps = frame_counter / (now_time - last_fps_time)
            print(f"[DEBUG] FPS: {fps:.1f} | Frame: {frame_counter} | Counters EAR:{ear_counter} EYE:{eye_cnn_counter} YAWN:{yawn_counter}")
            logger.info(f"FPS={fps:.1f}, bad_frames={bad_frames}, ear_counter={ear_counter}")
            sys.stdout.flush()
            frame_counter = 0
            last_fps_time = now_time
        
        frame = detect_drowsiness(frame, detector, predictor)
        
        # Validate processed frame
        if (frame is None or len(frame.shape) != 3 or frame.shape[0] < 1 or frame.shape[1] < 1):
            print(f"Warning: Processed frame invalid, shape={getattr(frame, 'shape', 'None')}")
            bad_frames += 1
            if bad_frames > 10:
                print("Too many bad processed frames. Exiting.")
                break
            continue
        
        cv2.imshow("Drowsiness Detector - Press ESC or Q to quit", frame)
        cv2.setWindowProperty("Drowsiness Detector - Press ESC or Q to quit", cv2.WND_PROP_TOPMOST, 1)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break

    global tts_engine
    if tts_engine is not None:
        try:
            tts_engine.stop()
            tts_engine = None
        except:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print("Drowsiness Detector stopped.")

if __name__ == "__main__":
    main()
