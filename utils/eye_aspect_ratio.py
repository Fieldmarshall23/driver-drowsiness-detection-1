# ========================================
# EYE ASPECT RATIO (EAR) UTILITY
# ========================================
# Standard 6-point EAR formula from "Detecting Driver Drowsiness"
# Input: 6 (x,y) landmarks clockwise from outer horizontal eye corners
# Points order: p1(top-left), p2(top), p3(top-right), p4(bottom-right), 
#               p5(bottom), p6(bottom-left)

from math import hypot
import numpy as np

def eye_aspect_ratio(eye_points):
    """
    Compute Eye Aspect Ratio (EAR) - standard horizontal.
    
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Low EAR (~0.2) indicates closed eye; high EAR (~0.3) indicates open.
    
    Args:
        eye_points: List of 6 (x,y) tuples [p1..p6] eye landmarks
        
    Returns:
        float EAR value (0.0 if degenerate)
    """
    if len(eye_points) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points[:6]
    
    # Vertical distances (eyelid separation)
    A = hypot(p2[0] - p6[0], p2[1] - p6[1])
    B = hypot(p3[0] - p5[0], p3[1] - p5[1])
    
    # Horizontal eye width (normalization)
    C = hypot(p1[0] - p4[0], p1[1] - p4[1])
    
    if C == 0:
        return 0.0
        
    return (A + B) / (2.0 * C)

def eye_aspect_ratio_vertical(eye_points):
    """
    Vertical EAR variant: eyelid separation / total eye height.
    More robust to partial occlusions.
    """
    if len(eye_points) < 6:
        return 1.0
    points = np.array(eye_points[:6])
    ys = points[:, 1]
    eyelid_sep = np.max(ys[:3]) - np.min(ys[3:])
    eye_height = np.max(ys) - np.min(ys)
    return eyelid_sep / eye_height if eye_height > 0 else 1.0

def fused_ear(eye_points, prev_ears=None, baseline=0.27):
    """
    Enhanced fused EAR with dlib shape quality, 3-frame median smoothing.
    
    Args:
        eye_points: 6 landmarks
        prev_ears: Optional list of last 2 EARs for smoothing
        baseline: Adaptive personal baseline
        
    Returns:
        tuple (ear, quality_score 0-1)
    """
    if len(eye_points) < 6:
        return 0.0, 0.0
    
    points = np.array(eye_points)
    
    # Dlib shape quality: landmark fitting consistency (low variance = good)
    eye_width = np.max(points[:,0]) - np.min(points[:,0])
    eye_height = np.max(points[:,1]) - np.min(points[:,1])
    aspect_ratio = eye_width / eye_height if eye_height > 0 else 10
    shape_quality = 1.0 / (1.0 + abs(aspect_ratio - 3.0))  # Ideal eye ~3:1
    
    # Reject poor landmarks
    if shape_quality < 0.4:
        return 0.0, shape_quality
    
    ear_h = eye_aspect_ratio(eye_points)
    ear_v = eye_aspect_ratio_vertical(eye_points)
    raw_ear = 0.6 * ear_h + 0.4 * ear_v
    
    # 3-frame median smoothing (robust to outliers)
    if prev_ears and len(prev_ears) >= 2:
        recent = sorted(prev_ears[-2:] + [raw_ear])
        smooth_ear = recent[1]  # Median
    else:
        smooth_ear = raw_ear
    
    # Adaptive normalization
    adaptive_ear = smooth_ear / max(baseline, 0.1)
    
    return adaptive_ear, shape_quality

# Enhanced API alias
def ear_with_quality(eye_points, prev_ears=None, baseline=0.27):
    ear, quality = fused_ear(eye_points, prev_ears, baseline)
    return ear  # Backward compatible

eye_aspect_ratio_fused = ear_with_quality

