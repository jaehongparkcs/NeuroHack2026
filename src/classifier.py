import numpy as np

class BiometricClassifier:
    @staticmethod
    def predict_state(vector):
        center, wander, blink, missing, hr = vector
        
        # Priority 1: Physical Absence
        if missing > 20: return "Away"
        
        # Priority 2: Biological Sleep (HR drops + Blinks)
        if blink > 10 or (hr < 65 and blink > 7): return "Drowsy"
        
        # Priority 3: Attention Drift (Wandering eyes)
        if wander > 9: return "Distracted"
        
        # Default: Active Working
        return "Focused"

    @staticmethod
    def calculate_attention_score(center_s, wander_s, hrv):
        """
        Fuses Eye Tracking and HRV into a continuous Attention Score (0.0 to 1.0).
        """
        total_gaze = center_s + wander_s + 1 
        gaze_score = center_s / total_gaze
        
        hrv_val = float(hrv) if isinstance(hrv, (int, float)) else 50.0
        hrv_score = np.clip((hrv_val - 20) / 60.0, 0.0, 1.0)
        
        attention = (gaze_score * 0.6) + (hrv_score * 0.4)
        return np.round(attention, 2)