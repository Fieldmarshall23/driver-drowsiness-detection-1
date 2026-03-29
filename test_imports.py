#!/usr/bin/env python3
"""
Test the specific TensorFlow/Keras imports that are failing + utils imports
"""

print("Testing TensorFlow/Keras imports...")

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("✅ ImageDataGenerator imported successfully")
except ImportError as e:
    print(f"❌ ImageDataGenerator import failed: {e}")

try:
    from tensorflow.keras.models import Sequential
    print("✅ Sequential imported successfully")
except ImportError as e:
    print(f"❌ Sequential import failed: {e}")

try:
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    print("✅ Layers imported successfully")
except ImportError as e:
    print(f"❌ Layers import failed: {e}")

try:
    from tensorflow.keras.optimizers import Adam
    print("✅ Adam optimizer imported successfully")
except ImportError as e:
    print(f"❌ Adam optimizer import failed: {e}")

print("\nTesting utils imports...")
try:
    from utils.landmark_utils import shape_to_coords, get_left_eye, crop_eye
    print("✅ utils.landmark_utils imported successfully")
except ImportError as e:
    print(f"❌ utils.landmark_utils import failed: {e}")

try:
    from utils.eye_aspect_ratio import eye_aspect_ratio
    print("✅ utils.eye_aspect_ratio imported successfully")
except ImportError as e:
    print(f"❌ utils.eye_aspect_ratio import failed: {e}")

# Test OpenCV
try:
    import cv2
    print("✅ cv2 imported successfully")
    print(f"   OpenCV version: {cv2.__version__}")
    print(f"   VideoCapture available: {hasattr(cv2, 'VideoCapture')}")
    cap = cv2.VideoCapture(0)
    print(f"   VideoCapture instantiated: {cap.isOpened()}")
    cap.release()
except Exception as e:
    print(f"❌ cv2 test failed: {e}")

print("\nIf any imports failed above, VS Code is using the wrong Python interpreter.")
print("Check VS Code's status bar (bottom) and select the virtual environment Python.")
