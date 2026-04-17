class BiometricClassifier:
    @staticmethod
    def predict_state(vector):
        center, wander, blink, missing, hr = vector
        
        # Priority 1: Physical Absence
        if missing > 12: return "Away"
        
        # Priority 2: Biological Sleep (HR drops + Blinks)
        if blink > 8 or (hr < 65 and blink > 4): return "Drowsy"
        
        # Priority 3: Mental Focus (High center streak + Calm HR)
        if center > 40 and hr < 78: return "Flow"
        
        # Priority 4: Attention Drift (Wandering eyes)
        if wander > 10: return "Distracted"
        
        # Default: Active Working
        return "Focused"