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
ALERT_SOUND_PATH = os.path.join(ASSETS_DIR, 'siren-alert.mp3')

# Global counters and state
ear_counter = 0
eye_cnn_counter = 0
yawn_counter = 0
side_start_time = None
last_audio_alert = 0.0
frame_counter = 0
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
DROWSY_PROB_THRESH = 0.6
DROWSY_RESET_THRESH = 0.4

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

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("pyttsx3 not installed for TTS fallback. Install with: pip install pyttsx3")

def play_audio_alert():
    """Play alert: MP3 → Beep → TTS fallback."""
    def _play():
        print("[AUDIO] Alert triggered!")
        print(f"[AUDIO] MP3 exists: {os.path.exists(ALERT_SOUND_PATH)}")
        played = False
        # 1. MP3 siren
        if playsound and os.path.exists(ALERT_SOUND_PATH):
            try:
                print("[AUDIO] Playing MP3...")
                playsound(ALERT_SOUND_PATH)
                print("[AUDIO] MP3 success!")
                played = True
            except Exception as e:
                print(f"[AUDIO] MP3 failed: {e}")
        elif playsound:
            print("[AUDIO] MP3 file missing")
        # 2. Windows beep
        if not played and platform.system() == "Windows":
            try:
                import winsound
                winsound.Beep(2500, 700)
                winsound.Beep(2000, 500)
                played = True
            except Exception:
                pass
        # 3. TTS fallback
        if not played and TTS_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.say("Alert! Stay awake for safe driving!")
                engine.runAndWait()
            except Exception:
                pass

    threading.Thread(target=_play, daemon=True).start()

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    norm = resized.astype('float32') / 255.0
    norm = np.expand_dims(norm, axis=(0, -1))
    norm = np.repeat(norm, 3, axis=-1)  # Repeat to RGB (1,64,64,3) for eye_cnn
    return norm

def detect_drowsiness(frame, detector, predictor):
    global ear_counter, eye_cnn_counter, yawn_counter, side_start_time, last_audio_alert, frame_counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    num_faces = len(faces)
    if num_faces == 0:
        print(f"[DEBUG] No faces detected in frame {frame_counter}")
    elif num_faces > 1:
        print(f"[DEBUG] Multiple faces ({num_faces}) in frame {frame_counter}")
    else:
        print(f"[DEBUG] Face detected in frame {frame_counter}")
    
    for face in faces:
        shape = predictor(gray, face)
        coords = shape_to_coords(shape)
        
        left_eye = np.array(get_left_eye(coords))
        right_eye = np.array(get_right_eye(coords))
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        try:
            ml_pred = ear_model.predict([[left_ear, right_ear, ear]])[0][0]  # Get probability of closed (class 1)
        except Exception as e:
            print(f"[ERROR] ML pred failed: {e}")
            ml_pred = 0.0

        try:
            left_eye_region = crop_eye(frame, list(left_eye))
            right_eye_region = crop_eye(frame, list(right_eye))
            left_input = preprocess(left_eye_region)
            right_input = preprocess(right_eye_region)
            left_prob = eye_cnn.predict(left_input)[0][0]
            right_prob = eye_cnn.predict(right_input)[0][0]
            eye_prob = (left_prob + right_prob) / 2.0
        except Exception as e:
            print(f"[ERROR] Eye CNN failed: {e}")
            eye_prob = 0.0
        eye_closed = eye_prob >= EYE_CNN_CLOSE_THRESH

        mouth = coords[48:68]
        try:
            mouth_region = crop_region(frame, mouth)
            mouth_input = preprocess(mouth_region)
            yawn_prob = mouth_cnn.predict(mouth_input)[0][0]
        except Exception as e:
            print(f"[ERROR] Mouth CNN failed: {e}")
            yawn_prob = 0.0

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
        if ear < EAR_THRESHOLD:
            ear_counter += 1
        else:
            ear_counter = 0

        # Eye CNN check
        if eye_closed:
            eye_cnn_counter += 1
        else:
            eye_cnn_counter = 0

        # Yawn check
        if yawn_prob >= YAWN_THRESH:
            yawn_counter += 1
        else:
            yawn_counter = 0

# Fused drowsiness probability
        ear_factor = 1.0 if ear < EAR_THRESHOLD else 0.0
        eye_factor = 1.0 if eye_closed else eye_prob  # Use prob if not binary
        yawn_factor = yawn_prob
        ml_factor = ml_pred
        side_factor = min(1.0, side_duration / SIDE_LOOK_THRESH)
        drowsy_prob = 0.25 * (ear_factor + eye_factor + yawn_factor + ml_factor + side_factor)
        if frame_counter % 30 == 0:
            print(f"[DETECT] Frame{frame_counter} EAR:{ear:.2f} eye:{eye_prob:.2f} yawn:{yawn_prob:.2f} drowsy:{drowsy_prob:.2f} side:{side_duration:.1f}")

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
            if side_duration >= SIDE_LOOK_THRESH:
                msg = "LOOK AHEAD - SAFETY FIRST!"
            elif yawn_prob >= YAWN_THRESH:
                msg = "TAKE A BREAK - STAY ALERT!"
            else:
                msg = "DROWSY! SAFETY DRIVING - WAKE UP!"
            msg2 = f"Drowsy Prob: {drowsy_prob:.2f}"

            now = time.time()
            if now - last_audio_alert >= ALERT_AUDIO_COOLDOWN:
                play_audio_alert()
                last_audio_alert = now

            cv2.putText(frame, msg, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            cv2.putText(frame, msg2, (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]),
                          (0,0,255), 5)
        else:
            cv2.putText(frame, "SAFE DRIVING - Eyes Open!", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

        # Draw face box, EAR, probs
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Eye Prob: {eye_prob:.2f}", (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Yawn Prob: {yawn_prob:.2f}", (20,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"NosePos: {nose_norm:.2f}", (20,170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        if side_duration > 0:
            cv2.putText(frame, f"SideSecs: {side_duration:.1f}s", (20,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    return frame

def crop_region(image, points, margin=10, size=(64,64)):
    # Similar to crop_eye but general for region (e.g., mouth)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    h, w = image.shape[:2]
    x1 = max(0, minx - margin)
    y1 = max(0, miny - margin)
    x2 = min(w, maxx + margin)
    y2 = min(h, maxy + margin)
    crop = image[y1:y2, x1:x2]
    if crop.size > 0:
        crop = cv2.resize(crop, size)
    return crop

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
    
    print("Drowsiness Detector started. Press 'q' to quit.")

    bad_frames = 0
    while True:
        ret, frame = cap.read()
        # Robust frame validation
        if (not ret or frame is None or len(frame.shape) != 3 or 
            frame.shape[0] < 1 or frame.shape[1] < 1):
            bad_frames += 1
            if bad_frames > 100:
                print("Too many consecutive bad frames. Exiting.")
                break
            print(f"Warning: Skipping invalid frame (bad_frames={bad_frames}).")
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

    cap.release()
    cv2.destroyAllWindows()
    print("Drowsiness Detector stopped.")

if __name__ == "__main__":
    main()