import numpy as np
import cv2
import random
import threading
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import mediapipe as mp

class BiometricSystem:
    def __init__(self):
        print("❤️ [rPPG] Initializing REAL rPPG with MediaPipe FaceMesh...")
        self.fps = 30 # Assumption of typical webcam framerate
        self.window_seconds = 5
        self.buffer_size = self.fps * self.window_seconds 
        
        # Buffer for the raw green light signal
        self.signal_buffer = deque(maxlen=self.buffer_size)
        
        self.current_hr = 0.0
        self.current_hrv = 0.0
        self.is_calibrated = False
        
        self.lock = threading.Lock()
        self.is_processing = False
        
        # --- INIT MEDIAPIPE FACE MESH ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame, current_mode, attention_score=0.5):
        """Finds the face, extracts forehead light intensity, and generates EEG."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # 1. Run MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 2. Extract specific landmarks for the Forehead ROI
            # Landmark 10 is upper forehead, 9 is between eyebrows
            # Landmark 109 is left temple area, 338 is right temple area
            top_y = int(landmarks[10].y * h)
            bottom_y = int(landmarks[9].y * h)
            left_x = int(landmarks[109].x * w)
            right_x = int(landmarks[338].x * w)
            
            # Clean up coordinates to stay within frame bounds
            top_y, bottom_y = max(0, top_y), min(h, bottom_y)
            left_x, right_x = max(0, left_x), min(w, right_x)
            
            if bottom_y > top_y and right_x > left_x:
                forehead_roi = rgb_frame[top_y:bottom_y, left_x:right_x]
                
                # Extract the spatial average of the Green Channel (Best for rPPG)
                avg_g = np.mean(forehead_roi[:, :, 1]) 
                
                with self.lock:
                    self.signal_buffer.append(avg_g)
                    
                # Visual Feedback: Draw the ROI box on the actual HUD frame
                cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 255, 100), 1)
                cv2.putText(frame, "rPPG LOCK", (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

        # 3. Check if we have enough data to calculate pulse
        with self.lock:
            ready = (len(self.signal_buffer) == self.buffer_size)
            
        if ready and not self.is_processing:
            self.is_processing = True
            threading.Thread(target=self._calculate_rppg, daemon=True).start()

        # Generate continuous EEG
        eeg = self._generate_eeg(attention_score)
        return self.current_hr, self.current_hrv, eeg, self.is_calibrated

    def _calculate_rppg(self):
        """Real-time signal processing in a background thread."""
        with self.lock:
            raw_signal = np.array(self.signal_buffer)
            
        try:
            # 1. Detrending (Remove slow changes in room lighting)
            x = np.arange(len(raw_signal))
            detrended = raw_signal - np.polyval(np.polyfit(x, raw_signal, 1), x)
            
            # 2. Bandpass Filter (Keep frequencies between 0.75Hz [45 BPM] and 3.0Hz [180 BPM])
            nyq = 0.5 * self.fps
            b, a = butter(3, [0.75 / nyq, 3.0 / nyq], btype='band')
            filtered_bvp = filtfilt(b, a, detrended)
            
            # 3. Peak Detection (Find the literal heartbeats)
            peaks, _ = find_peaks(filtered_bvp, distance=self.fps/3.0) 
            
            if len(peaks) >= 3:
                # Calculate HR (Beats per minute)
                time_between_beats = np.diff(peaks) / self.fps
                hr = 60.0 / np.mean(time_between_beats)
                
                # Calculate HRV (RMSSD in milliseconds)
                diffs_ms = np.diff(time_between_beats) * 1000.0 
                hrv = np.sqrt(np.mean(diffs_ms**2))
                
                # Filter out crazy anomalies (caused by sudden heavy head movement)
                if 45 < hr < 180 and 5 < hrv < 150:
                    with self.lock:
                        # Smooth the updates slightly
                        self.current_hr = round((self.current_hr * 0.5) + (hr * 0.5) if self.is_calibrated else hr, 1)
                        self.current_hrv = round((self.current_hrv * 0.5) + (hrv * 0.5) if self.is_calibrated else hrv, 1)
                        self.is_calibrated = True

        except Exception as e:
            print(f"rPPG Math Error: {e}")
            pass
            
        self.is_processing = False

    def _generate_eeg(self, attention_score):
        """Fluid, continuous EEG generation based on attention."""
        base_alpha = 0.2 + (0.6 * attention_score) 
        if attention_score < 0.3:
            base_beta = 0.1 + (attention_score * 1.0) 
        else:
            base_beta = 0.8 - (0.5 * abs(attention_score - 0.5)) 
        base_theta = 0.8 - (0.6 * attention_score) 
        
        return {
            "alpha": np.clip(base_alpha + random.uniform(-0.1, 0.1), 0.0, 1.0),
            "beta": np.clip(base_beta + random.uniform(-0.1, 0.1), 0.0, 1.0),
            "theta": np.clip(base_theta + random.uniform(-0.1, 0.1), 0.0, 1.0)
        }