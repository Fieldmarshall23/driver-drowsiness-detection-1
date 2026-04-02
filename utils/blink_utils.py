"""Blink Detection Utilities for Driver Drowsiness.

Tracks blink frequency over rolling windows using EAR transitions.
Detects blinks: rapid EAR drop (<thresh) then recovery.
Provides blink_rate (blinks per 10s), freq_score (high freq = alert).
"""

import collections
import numpy as np
from typing import List, Tuple
from utils.eye_aspect_ratio import eye_aspect_ratio, fused_ear
from math import hypot

class BlinkDetector:
    def __init__(self, ear_thresh=0.22, blink_window=10.0, fps=15):
        """
        Args:
            ear_thresh: EAR below which eye considered closed.
            blink_window: Rolling window seconds for blink rate.
            fps: Expected FPS for timing.
        """
        self.ear_thresh = ear_thresh
        self.window_size = int(blink_window * fps)  # Frames in window
        self.ear_history = collections.deque(maxlen=self.window_size)
        self.blink_history = collections.deque(maxlen=self.window_size)
        self.last_open_ear = 0.28
        self.frame_time = 1.0 / fps
        self.min_blink_dist = int(0.2 / self.frame_time)  # Min 200ms between blinks
        self.last_blink_frame = -self.min_blink_dist
        
    def update(self, left_ear: float, right_ear: float, frame_idx: int, 
                 left_eye_points=None, right_eye_points=None, prev_left=None, prev_right=None) -> dict:
        """
        Enhanced update with dlib velocity check for precise blink confirmation.
        """
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        
        # Velocity-based quality (dlib landmark movement during blink)
        eye_velocity = 0.0
        if (left_eye_points and right_eye_points and prev_left and prev_right):
            left_vel = np.mean([hypot(l[0]-p[0], l[1]-p[1]) for l,p in zip(left_eye_points, prev_left)])
            right_vel = np.mean([hypot(r[0]-p[0], r[1]-p[1]) for r,p in zip(right_eye_points, prev_right)])
            eye_velocity = (left_vel + right_vel) / 2.0
        
        is_closed = ear < self.ear_thresh
        blink_detected = False
        
        if is_closed:
            recent_ears = list(self.ear_history)[-5:]
            if len(recent_ears) >= 3 and np.mean(recent_ears[-3:]) < self.ear_thresh:
                self.last_open_ear = max(self.last_open_ear * 0.95, ear * 1.1)
        else:
            # Enhanced: sustained close + velocity spike + recovery
            was_recently_closed = len(self.ear_history) >= 5 and np.mean(list(self.ear_history)[-5:-2]) < self.ear_thresh * 1.1
            velocity_confirmed = eye_velocity > 2.0  # px/frame blink motion
            good_recovery = ear > self.ear_thresh * 1.3
            if (frame_idx - self.last_blink_frame > self.min_blink_dist and 
                was_recently_closed and velocity_confirmed and good_recovery):
                blink_detected = True
                self.last_blink_frame = frame_idx
        
        if blink_detected:
            self.blink_history.append(1)
        else:
            self.blink_history.append(0)
        
        # Enhanced metrics
        recent_blinks = sum(self.blink_history)
        blink_rate = recent_blinks / max(1, len(self.blink_history) / self.window_size * 10)
        
        # Improved freq_score: classify micro/natural blinks
        micro_blink_ratio = sum(1 for e in self.ear_history if e < self.ear_thresh * 1.2) / max(1, len(self.ear_history))
        if blink_rate > 0.5 or micro_blink_ratio > 0.3:
            freq_score = min(1.0, blink_rate / 0.4 + micro_blink_ratio)  # Drowsy
        elif blink_rate < 0.08:
            freq_score = min(0.25, blink_rate * 4)  # Distraction
        else:
            freq_score = 0.0
        
        # PERCLOS: % time below P20 baseline
        ears = np.array(self.ear_history)
        if len(ears) > 20:
            p20 = np.percentile(ears, 20)
            perclos = np.mean(ears < p20 * 1.1)  # Improved threshold
        else:
            perclos = 0.0
            
        return {
            'blink_detected': blink_detected,
            'blink_rate': blink_rate,
            'freq_score': freq_score,
            'perclos_pct': perclos,
            'is_closed': is_closed,
            'eye_velocity': eye_velocity,
            'avg_ear': np.mean(ears),
            'ear_thresh_adapt': np.percentile(ears, 20) if len(ears)>10 else self.ear_thresh,
            'ear_history_len': len(self.ear_history)
        }

# Standalone test
if __name__ == "__main__":
    detector = BlinkDetector()
    # Simulate: open-normal-closed-open
    test_ears = [(0.28, 0.27), (0.26, 0.25), (0.18, 0.19), (0.21, 0.24), (0.27, 0.28)]
    for i, (le, re) in enumerate(test_ears):
        metrics = detector.update(le, re, i)
        print(f"Frame {i}: {metrics}")

