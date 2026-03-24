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
        crop = np.zeros((size[1], size[0], image.shape[2] if len(image.shape)==3 else 1), dtype=image.dtype)
    return crop
