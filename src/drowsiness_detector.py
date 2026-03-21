import os
import cv2
import dlib
import datetime
import pickle
import numpy as np

import tensorflow as tf
from utils.landmark_utils import shape_to_coords, get_left_eye, get_right_eye, crop_eye
from utils.eye_aspect_ratio import eye_aspect_ratio

# Paths and directories
MODEL_DIR = 'models'
SAVE_DEBUG_DIR = 'outputs'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
EAR_MODEL_PATH = os.path.join(MODEL_DIR, 'ml_model.pkl')

# Global counters
ear_counter = 0
eye_cnn_counter = 0
yawn_counter = 0
side_start_time = None

# Constants
EYE_CNN_CLOSE_THRESH = 0.5
EYE_CNN_CONSEC = 5
YAWN_THRESH = 0.6
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 3
YAWN_CONSEC = 3
SIDE_LOOK_THRESH = 5.0
SIDE_LEFT_RATIO = 0.3
SIDE_RIGHT_RATIO = 0.7
SAVE_DEBUG_THRESH = 0.7

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

with open(EAR_MODEL_PATH, 'rb') as f:
    ear_model = pickle.load(f)

def detect_drowsiness(frame, detector, predictor):
    global ear_counter, eye_cnn_counter, yawn_counter, side_start_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    for face in faces:
        shape = predictor(gray, face)
        coords = shape_to_coords(shape)
        
        left_eye = get_left_eye(coords)
        right_eye = get_right_eye(coords)
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        try:
            ml_pred = ear_model.predict([[ear]])[0]
        except Exception:
            ml_pred = None

        try:
            eye_region = crop_eye(frame, list(left_eye) + list(right_eye))
        except Exception:
            eye_region = None



        # Mouth region extraction and prediction
        mouth_pts = coords[48:68]
        mouth_region = crop_region(frame, mouth_pts)
        mouth_resized = cv2.resize(mouth_region, (64, 64))
        mouth_gray = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY)
        mouth_input = np.expand_dims(mouth_gray / 255.0, axis=(0, -1))

        yawn_prob = 0.0
        try:
            yawn_prob = mouth_cnn.predict(mouth_input, verbose=0)[0][0]
        except Exception:
            yawn_prob = 0.0

        # Eye prediction
        eye_prob = 0.5
        eye_closed = True
        try:
            if eye_region is not None:
                eye_input = np.expand_dims(eye_region / 255.0, axis=0)
                eye_prob = eye_cnn.predict(eye_input, verbose=0)[0][0]
            eye_closed = eye_prob < EYE_CNN_CLOSE_THRESH
        except Exception:
            eye_closed = True

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
        if yawn_prob > YAWN_THRESH:
            yawn_counter += 1
        else:
            yawn_counter = 0

        # Alert conditions
        drowsy = ear_counter >= EAR_CONSEC_FRAMES or eye_cnn_counter >= EYE_CNN_CONSEC
        yawning = yawn_counter >= YAWN_CONSEC
        if drowsy or yawning:
            cv2.putText(frame, "ALERT: Drowsy!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Save debug frame
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            debug_path = os.path.join(SAVE_DEBUG_DIR, f"alert_{timestamp}.jpg")
            cv2.imwrite(debug_path, frame)
        else:
            cv2.putText(frame, "Drive Safe!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw face box, EAR, probs
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Eye: {eye_prob:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawn: {yawn_prob:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
    
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Drowsiness Detector started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame = detect_drowsiness(frame, detector, predictor)
        cv2.imshow("Drowsiness Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Drowsiness Detector stopped.")

if __name__ == "__main__":
    main()