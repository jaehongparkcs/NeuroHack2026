class BiometricClassifier:
    @staticmethod
    def predict_state(vector):
        center, wander, blink, missing, hr = vector
        
        # Priority 1: Physical Absence
        
        # Priority 2: Biological Sleep (HR drops + Blinks)
        if blink > 15 or (hr < 65 and blink > 10): return "Drowsy"

        elif missing > 20: return "Away"
        
        # Priority 4: Attention Drift (Wandering eyes)
        if wander > 9: return "Distracted"
        
        # Default: Active Working
        return "Focused"

    @staticmethod
    def calculate_attention_score(center_s, wander_s, hrv):
        """
        Fuses Eye Tracking and HRV into a continuous Attention Score (0.0 to 1.0).
        Lightweight NumPy implementation.
        """
        # 1. Gaze Stability (0 to 1 scale)
        # We add 1 to prevent division by zero. If center=50 and wander=0, score is ~1.0
        total_gaze = center_s + wander_s + 1 
        gaze_score = center_s / total_gaze
        
        # 2. HRV Stability (0 to 1 scale)
        # Normal working HRV is typically between 20ms (high stress) and 80ms (relaxed focus).
        hrv_val = float(hrv) if isinstance(hrv, (int, float)) else 50.0
        hrv_score = np.clip((hrv_val - 20) / 60.0, 0.0, 1.0)
        
        # 3. Multimodal Fusion (The Magic)
        # We weigh Gaze 60% (since you must look at the screen to focus) 
        # and HRV 40% (to measure the cognitive quality of that focus).
        attention = (gaze_score * 0.6) + (hrv_score * 0.4)
        
        return attention