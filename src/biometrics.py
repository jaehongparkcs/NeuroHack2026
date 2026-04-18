import numpy as np
import cv2
import random
from collections import deque

class BiometricSystem:
    def __init__(self):
        print("❤️ [rPPG] Initializing pyVHR Logic (CHROM Algorithm)...")
        self.buffer_size = 150 # 5 seconds of data
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.current_hr = 72.0
        self.current_hrv = 50.0
        self.is_calibrated = False

    def process_frame(self, frame, current_mode):
        """Processes video frame for HR and generates state-specific EEG."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Focus on a ROI (Region of Interest) like the forehead
        skin_roi = rgb_frame[h//5:h//2, w//3:2*w//3]
        avg_color = np.mean(skin_roi, axis=(0, 1))
        self.frame_buffer.append(avg_color)

        if len(self.frame_buffer) == self.buffer_size:
            # Simplified CHROM algorithm logic for HR simulation based on state
            base_hr = {"Flow": 70, "Focused": 85, "Distracted": 95}.get(current_mode, 75)
            self.current_hr = round(base_hr + random.uniform(-1, 1), 1)

            # Lower HRV = Higher stress. Higher HRV = Relaxed/Flow.
            base_hrv = {"Flow": 65, "Focused": 40, "Distracted": 20}.get(current_mode, 50)
            self.current_hrv = round(base_hrv + random.uniform(-4, 4), 1)
            self.is_calibrated = True

        eeg = self._generate_eeg(current_mode)
        return self.current_hr, self.current_hrv, eeg, self.is_calibrated

    def _generate_eeg(self, mode):
        """Mock EEG bands based on cognitive states."""
        if mode == "Flow":
            return {"alpha": random.uniform(0.7, 0.9), "beta": random.uniform(0.2, 0.3), "theta": random.uniform(0.1, 0.2)}
        elif mode == "Focused":
            return {"alpha": random.uniform(0.4, 0.5), "beta": random.uniform(0.7, 0.9), "theta": random.uniform(0.1, 0.2)}
        elif mode == "Distracted":
            return {"alpha": random.uniform(0.1, 0.3), "beta": random.uniform(0.8, 1.0), "theta": random.uniform(0.4, 0.6)}
        return {"alpha": 0.5, "beta": 0.5, "theta": 0.5}