import cv2

# Helpers to extract eye landmarks from the 68-point dlib predictor shape

def shape_to_coords(shape):
    # shape: dlib shape object
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return coords

def get_left_eye(coords):
    # left eye points: 42-47 (0-based indexing)
    return coords[42:48]

def get_right_eye(coords):
    # right eye points: 36-41
    return coords[36:42]

def crop_eye(image, eye_points, margin=5, size=(64,64)):
    # eye_points: list of (x,y)
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
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
    else:
        import numpy as np
        crop = np.zeros((size[1], size[0], image.shape[2] if len(image.shape)==3 else 1), dtype=image.dtype)
    return crop


def eye_aspect_ratio_vertical(eye_coords):
    """
    Additional vertical eye ratio for closure detection.
    Returns vertical_ratio (low when closed).
    """
    if len(eye_coords) != 6:
        return 1.0
    upper = min(p[1] for p in eye_coords[:3])
    lower = max(p[1] for p in eye_coords[3:])
    v_dist = lower - upper
    eye_h = max(p[1] for p in eye_coords) - min(p[1] for p in eye_coords)
    if eye_h == 0:
        return 1.0
    return v_dist / eye_h


def head_pose(coords, image_shape):
    """
    Estimate head pose (yaw, pitch, roll) using PnP with 68 landmark 3D model points.
    Returns (yaw_deg, pitch_deg, roll_deg)
    """
    import numpy as np
    h, w = image_shape[:2]
    
    # Standard 3D model points for 68 landmarks (simplified key points subset for accuracy)
    # Using 6 key points for robust PnP: nose tip(30), chin(8), eye corners, mouth corners
    model_points = np.array([
        (0.0, 0.0, 0.0),      # Nose tip (30)
        (0.0, -330.0, -65.0), # Chin (8)
        (-225.0, 170.0, -135.0),  # Left eye left corner (36)
        (225.0, 170.0, -135.0),   # Right eye right corner (45)
        (-150.0, -150.0, -125.0), # Left Mouth corner (48)
        (150.0, -150.0, -125.0)   # Right Mouth corner (54)
    ], dtype="double")
    
    # Corresponding 2D image points from coords
    indices = [30, 8, 36, 45, 48, 54]
    image_points = np.array([coords[i] for i in indices], dtype="double")
    
    # Camera internals (approximation for 640x480)
    size = (w, h)
    focal_length = size[0]
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
    )
    
    dist_coeffs = np.zeros((4,1))  # No lens distortion
    
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        return 0.0, 0.0, 0.0
    
    # Convert rotation to Euler angles (yaw, pitch, roll)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    if rotation_mat.shape != (3,3):
        return 0.0, 0.0, 0.0
    euler_angles = cv2.RQDecomp3x3(rotation_mat)[0]
    
    pitch, yaw, roll = euler_angles[0], euler_angles[2], euler_angles[1]
    return yaw, pitch, roll


def eye_gaze_offset(eye_coords):
    """
    Simple gaze offset: 'pupil' (inner eye avg) relative to eye center.
    Returns horizontal offset (-1 left to 1 right gaze).
    eye_coords: 6 points [outer, upper outer, upper inner, inner lower, lower outer, lower inner]
    """
    import numpy as np
    if len(eye_coords) != 6:
        return 0.0
    
    # Eye center: avg of all points
    cx = sum(p[0] for p in eye_coords) / 6
    cy = sum(p[1] for p in eye_coords) / 6
    
    # Pupil proxy: avg of inner points (2 upper inner, 3 lower inner)
    inner_points = [eye_coords[2], eye_coords[3]]  # upper/lower inner
    px = sum(p[0] for p in inner_points) / 2
    py = sum(p[1] for p in inner_points) / 2
    
    # Horizontal gaze offset (normalized width)
    eye_w = max(p[0] for p in eye_coords) - min(p[0] for p in eye_coords)
    if eye_w == 0:
        return 0.0
    offset = (px - cx) / (eye_w / 2)  # -1 to 1
    return np.clip(offset, -1.0, 1.0)
