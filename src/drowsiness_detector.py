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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
warnings.filterwarnings('ignore', message='.*All `Callback`.*')
warnings.filterwarnings('ignore', message='.*glibc.*')
warnings.filterwarnings('ignore', message='.*unrecognized shape.*')
from utils.landmark_utils import shape_to_coords, get_left_eye, get_right_eye, crop_eye
from utils.eye_aspect_ratio import eye_aspect_ratio

try:
    from playsound import playsound
except Exception:
    playsound = None

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PREDICTOR_PATH = os.path.join(PROJECT_ROOT, 'shape_predictor_68_face_landmarks.dat')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
SAVE_DEBUG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'debug_mouth')
ALERT_LOG = os.path.join(PROJECT_ROOT, 'outputs', 'alerts.log')
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(SAVE_DEBUG_DIR, exist_ok=True)

# Logging setup for debugging
import logging
logging.basicConfig(level=logging.INFO, filename=ALERT_LOG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='a')
logger = logging.getLogger(__name__)

DROWSY_SOUND_PATH = os.path.join(ASSETS_DIR, 'drowsy_alert.mp3')
CONCENTRATE_SOUND_PATH = os.path.join(ASSETS_DIR, 'concentrate_alert.mp3')

# Global counters and state
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
heavy_process_frame = 0
last_fps_time = 0.0
fps = 0.0

# Constants (restored from original)
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 25  # Increased for accuracy
EYE_CNN_CLOSE_THRESH = 0.5
EYE_CNN_CONSEC = 8      # Increased
YAWN_THRESH = 0.5
YAWN_CONSEC = 3
SIDE_LOOK_THRESH = 3.0
SIDE_LEFT_RATIO = 0.35
SIDE_RIGHT_RATIO = 0.65
SAVE_DEBUG_THRESH = 0.4
ALERT_AUDIO_COOLDOWN = 3.0
DROWSY_PROB_THRESH = 0.65
DROWSY_RESET_THRESH = 0.35
# New improved constants
PERCLOS_WINDOW = 300
EMA_ALPHA = 0.3
FUSION_WEIGHTS = [0.3, 0.25, 0.2, 0.15, 0.1]  # EAR, eye, yawn, ml, perclos
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

eye_cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'))

# Load mouth CNN model
mouth_cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'mouth_cnn_model.h5'))

EAR_MODEL_PATH = os.path.join(MODEL_DIR, 'ml_model.pkl')
with open(EAR_MODEL_PATH, 'rb') as f:
    ear_model = pickle.load(f)

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
    global ear_counter, eye_cnn_counter, yawn_counter, side_start_time, last_audio_alert, frame_counter
    if frame is None:
        return None
    # Preprocess for robust detection
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray_raw)
    # Gamma correction
    gamma = 1.2
    gray = np.power(gray / 255.0, gamma) * 255.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Multi-scale dlib (0-2) for speed
    faces = []
    max_faces_scale = -1
    for scale in range(3):
        scale_faces = detector(gray, scale)
        if len(scale_faces) > 0:
            faces = scale_faces
            max_faces_scale = scale
            break

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

    global heavy_process_frame
    heavy_process_frame = frame_counter % 2 == 0
    
    alert_active = False
    
    num_faces = len(faces)
    if frame_counter % 60 == 0:
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

    shape = predictor(gray, face)
    coords = shape_to_coords(shape)

    left_eye = np.array(get_left_eye(coords))
    right_eye = np.array(get_right_eye(coords))

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    
    # Step 2: EAR baseline collection
    global ear_baseline, baseline_collected
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
            
            # Mouth CNN
            mouth = coords[48:68]
            mouth_region = crop_region(frame, mouth)
            mouth_input = preprocess_grayscale(mouth_region)
            yawn_prob = mouth_cnn.predict(mouth_input, verbose=0)[0][0]
            
            # Step 3: Mouth Aspect Ratio (MAR) for yawn validation
            # Landmarks 51-57 (upper outer lip), 61-67 (lower outer lip)
            mouth_points = coords[51:60] + coords[61:68]  # 6 upper + 6 lower
            mouth_left = np.array([mouth_points[0], mouth_points[3]])
            mouth_right = np.array([mouth_points[5], mouth_points[8]])
            mouth_top = np.array([mouth_points[1], mouth_points[2]])
            mouth_bottom = np.array([mouth_points[4], mouth_points[7]])
            mar_v1 = np.linalg.norm(mouth_top - mouth_bottom)
            mar_v2 = np.linalg.norm(mouth_left - mouth_right) * 2
            mar = mar_v1 / mar_v2 if mar_v2 > 0 else 0.0
            yawn_prob = max(0.0, min(1.0, yawn_prob * (1.0 + (mar - 0.02))))  # Boost if MAR high (>0.02 yawn)
        except Exception:
            mar = 0.0
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
    global ear_baseline, EAR_THRESHOLD
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

    x_coords = [p[0] for p in coords]
    minx, maxx = min(x_coords), max(x_coords)
    nose_x = coords[30][0]
    face_w = maxx - minx if (maxx - minx) > 0 else 1
    nose_norm = (nose_x - minx) / face_w
    side_looking = (nose_norm < SIDE_LEFT_RATIO) or (nose_norm > SIDE_RIGHT_RATIO)
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

# Step 5: Improved weighted fusion with normalization + PERCLOS
    ear_factor = 1.0 if ear < current_ear_thresh else 0.0
    eye_factor = ema_eye_prob
    yawn_factor = ema_yawn_prob
    ml_factor = ml_pred
    perclos_factor = perclos
    side_factor = min(1.0, side_duration / SIDE_LOOK_THRESH)
    
    # Normalize to [0,1]
    factors = [ear_factor, eye_factor, yawn_factor, ml_factor, perclos_factor]
    drowsy_prob = sum(w * f for w, f in zip(FUSION_WEIGHTS, factors))
    if frame_counter % 30 == 0:
        print(f"[DETECT] Frame{frame_counter} EAR:{ear:.2f} EMA-eye:{ema_eye_prob:.2f} EMA-yawn:{ema_yawn_prob:.2f} PERCLOS:{perclos:.3f} drowsy:{drowsy_prob:.2f} side:{side_duration:.1f} thresh:{current_ear_thresh:.3f}")

    # Update counters
    if drowsy_prob >= DROWSY_PROB_THRESH:
        ear_counter += 1
    elif drowsy_prob < DROWSY_RESET_THRESH:
        ear_counter = max(0, ear_counter - 1)
        eye_cnn_counter = max(0, eye_cnn_counter - 1)
        yawn_counter = max(0, yawn_counter - 1)

    # Alert conditions
    alert_active = (ear_counter >= EAR_CONSEC_FRAMES or 
                   eye_cnn_counter >= EYE_CNN_CONSEC or 
                   yawn_counter >= YAWN_CONSEC or 
                   side_duration >= SIDE_LOOK_THRESH or
                   drowsy_prob >= DROWSY_PROB_THRESH)
    if alert_active:
        print(f"[ALERT] TRIGGERED! drowsy_prob={drowsy_prob:.2f} ear_counter={ear_counter} eye_cnn={eye_cnn_counter} yawn={yawn_counter} side={side_duration:.1f}")
        # Task-specific warnings
        is_drowsy = (ear_counter >= EAR_CONSEC_FRAMES or eye_cnn_counter >= EYE_CNN_CONSEC or 
                    yawn_counter >= YAWN_CONSEC or drowsy_prob >= DROWSY_PROB_THRESH)
        is_looking_away = side_duration >= SIDE_LOOK_THRESH
        
        if is_drowsy:
            drowsy_msgs = ["PULLOVER TO REST!", "DROWSINESS DETECTED - REST NOW!", "TAKE A BREAK - SAFETY FIRST!"]
            msg = drowsy_msgs[frame_counter % len(drowsy_msgs)]
            play_drowsy_alert()
        elif is_looking_away:
            concentrate_msgs = ["PLEASE CONCENTRATE!", "EYES ON ROAD!", "FOCUS FORWARD - SAFETY!"]
            msg = concentrate_msgs[frame_counter % len(concentrate_msgs)]
            play_concentrate_alert()
        else:
            msg = "Alert Active!"
        
        msg2 = f"Drowsy: {drowsy_prob:.1f} | Side: {side_duration:.1f}s"

        now = time.time()
        if now - last_audio_alert >= ALERT_AUDIO_COOLDOWN:
            last_audio_alert = now

        # Red border + warnings
        cv2.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]), (0,0,255), 5)
        cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        cv2.putText(frame, msg2, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    else:
        # Safe: Green ROAD FOCUS ON
        cv2.putText(frame, "ROAD FOCUS: ON ✓", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]), (0,255,0), 2)

    # Draw face box, EAR, probs
    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Eye EMA: {ema_eye_prob:.2f}", (20,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Yawn EMA: {ema_yawn_prob:.2f}", (20,140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.3f}", (20,200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"NosePos: {nose_norm:.2f}", (20,170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
    if side_duration > 0:
        cv2.putText(frame, f"SideSecs: {side_duration:.1f}s", (20,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

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
    loop_start = time.time()
    while True:
        if time.time() - loop_start > 300:  # 5min timeout safety
            logger.warning("Loop timeout - restarting cap")
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            loop_start = time.time()
        
        ret, frame = cap.read()
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
        # Resize if too large
        h, w = frame.shape[:2]
        if h > 720 or w > 1280:
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
