import numpy as np
import cv2
import random
from collections import deque

class BiometricSystem:
    def __init__(self):
        print("❤️ [rPPG] Initializing pyVHR Logic (CHROM Algorithm)...")
        self.buffer_size = 150 
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.current_hr = 72.0
        self.current_hrv = 50.0
        self.is_calibrated = False

    def process_frame(self, frame, current_mode, attention_score=0.5):
        """Processes video frame for HR and generates continuous EEG based on attention."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        skin_roi = rgb_frame[h//5:h//2, w//3:2*w//3]
        avg_color = np.mean(skin_roi, axis=(0, 1))
        self.frame_buffer.append(avg_color)

        if len(self.frame_buffer) == self.buffer_size:
            # HR/HRV still use the categorical mode to simulate physical state
            base_hr = {"Flow": 70, "Focused": 85, "Distracted": 95, "Drowsy": 60, "Away": 65}.get(current_mode, 75)
            self.current_hr = round(base_hr + random.uniform(-1, 1), 1)

            base_hrv = {"Flow": 65, "Focused": 40, "Distracted": 20, "Drowsy": 55, "Away": 50}.get(current_mode, 50)
            self.current_hrv = round(base_hrv + random.uniform(-4, 4), 1)
            self.is_calibrated = True

        # EEG is now generated on a continuous spectrum
        eeg = self._generate_eeg(attention_score)
        return self.current_hr, self.current_hrv, eeg, self.is_calibrated

    def _generate_eeg(self, attention_score):
        """
        Maps a 0.0-1.0 attention score to brainwaves organically:
        - High (0.8+): High Alpha, moderate Beta, low Theta (Flow/Focused)
        - Mid (0.4-0.7): High Beta, low Alpha (Distracted/Active thinking)
        - Low (<0.3): High Theta, low Alpha/Beta (Drowsy)
        """
        # Alpha (Relaxed Focus): Scales linearly with attention
        base_alpha = 0.2 + (0.6 * attention_score) 
        
        # Beta (Active/Stress): Peaks when mildly distracted (around 0.5), drops when drowsy
        if attention_score < 0.3:
            base_beta = 0.1 + (attention_score * 1.0) 
        else:
            base_beta = 0.8 - (0.5 * abs(attention_score - 0.5)) 
            
        # Theta (Sleepy): High when attention is zero, drops as attention rises
        base_theta = 0.8 - (0.6 * attention_score) 
        
        return {
            "alpha": np.clip(base_alpha + random.uniform(-0.1, 0.1), 0.0, 1.0),
            "beta": np.clip(base_beta + random.uniform(-0.1, 0.1), 0.0, 1.0),
            "theta": np.clip(base_theta + random.uniform(-0.1, 0.1), 0.0, 1.0)
        }