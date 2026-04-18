import cv2
from gaze_tracking import GazeTracking
from audio_controller import AudioController
from biometrics import BiometricSystem
from classifier import BiometricClassifier

class FocusTrackStress:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.audio = AudioController()
        self.bio = BiometricSystem()

        from ai_engine import AIEngine 
        self.ai_engine = AIEngine()
        
        self.mode = "Focused"
        self.fatigue_level = 0.0
        self.flow_warmup = 0 
        self.WARMUP_THRESHOLD = 150 # ~5 seconds to "earn" Flow state
        self.was_distracted = True # Start true so the first focus is "clean"
        
        # Streaks
        self.center_s, self.wander_s, self.blink_s, self.miss_s = 0, 0, 0, 0

        import time
        self.session_start = time.time()
        self.history_stress = []
        self.history_states = []

    def print_session_summary(self):
        import time
        import collections
        import numpy as np

        total_time = int(time.time() - self.session_start)
        minutes, seconds = divmod(total_time, 60)
        
        print("\n" + "="*50)
        print("🧠 FOCUSTrackStress: SESSION ANALYTICS REPORT 🧠")
        print("="*50)
        print(f"⏱️  Total Session Time: {minutes}m {seconds}s")
        
        if not self.history_stress:
            print("⚠️ Not enough calibrated data collected during session.")
            print("="*50 + "\n")
            return

        # Calculate Averages
        avg_stress = np.mean(self.history_stress) * 100
        max_stress = np.max(self.history_stress) * 100
        
        # Calculate State Percentages
        state_counts = collections.Counter(self.history_states)
        total_frames = len(self.history_states)
        
        flow_pct = (state_counts.get("Flow", 0) / total_frames) * 100
        focus_pct = (state_counts.get("Focused", 0) / total_frames) * 100
        dist_pct = (state_counts.get("Distracted", 0) / total_frames) * 100

        print(f"\n📊 STRESS METRICS:")
        print(f"   Average Stress Level: {avg_stress:.1f}%")
        print(f"   Peak Stress Level:    {max_stress:.1f}%")
        
        print(f"\n🎯 ATTENTION DISTRIBUTION:")
        print(f"   🌊 Deep Flow:   {flow_pct:.1f}%")
        print(f"   👁️  Focused:    {focus_pct:.1f}%")
        print(f"   ⚠️  Distracted: {dist_pct:.1f}%")
        
        # Give a smart recommendation
        print("\n💡 INSIGHT:")
        if avg_stress > 60 and dist_pct > 30:
            print("   High stress and high distraction detected. You are experiencing cognitive overload. Take a 10-minute break.")
        elif flow_pct > 50:
            print("   Excellent session! You achieved highly sustained Flow states.")
        else:
            print("   Good effort, but your focus was fragmented. Try removing environmental triggers next session.")
        print("="*50 + "\n")

    def run(self):
        print("🚀 FocusTrackStress: Optimized Logic & AI active.")
        while True:
            ret, frame = self.webcam.read()
            if not ret: break
            
            self.gaze.refresh(frame)

            h, v = self.gaze.horizontal_ratio(), self.gaze.vertical_ratio()
            
            left_pupil = self.gaze.pupil_left_coords()
            right_pupil = self.gaze.pupil_right_coords()
            one_eye_missing = (left_pupil is None and right_pupil is not None) or (right_pupil is None and left_pupil is not None)

            is_center = (0.43 < h < 0.67) and (0.42 < v < 0.70) if (h and v) else False
            
            # --- STREAK LOGIC ---
            if not (h or v or self.gaze.is_blinking()):
                self.miss_s = min(self.miss_s + 1, 30) 
                self.wander_s = min(self.wander_s + 2, 25) 
                self.center_s = 0
                self.blink_s = 0
            elif self.gaze.is_blinking():
                self.blink_s = min(self.blink_s + 1, 30)
                self.miss_s = 0
                if self.blink_s > 5: 
                    self.center_s = 0
            else:
                self.blink_s = 0
                self.miss_s = 0
                
                if one_eye_missing:
                    self.wander_s = min(self.wander_s + 2, 25) 
                    self.center_s = 0
                    self.was_distracted = True
                elif not is_center:
                    self.wander_s = min(self.wander_s + 1, 25)
                    self.center_s = 0
                    self.was_distracted = True
                else:
                    if self.was_distracted:
                        self.wander_s = 0   
                        self.was_distracted = False
                    self.center_s += 1
                    self.wander_s = max(0, self.wander_s - 3)

           # --- SENSOR FUSION & AI ---
            # 1. Calculate Attention FIRST using the last known HRV
            attention_score = BiometricClassifier.calculate_attention_score(
                self.center_s, self.wander_s, self.bio.current_hrv
            )
            
            # 2. Pass the Attention Score to generate the continuous EEG
            hr, hrv, eeg, calibrated = self.bio.process_frame(frame, self.mode, attention_score)
            
            # 3. Predict the Base Mode
            vector = [self.center_s, self.wander_s, self.blink_s, self.miss_s, hr]
            base_mode = BiometricClassifier.predict_state(vector)
            
            # 4. Neural Network predicts pure stress
            stress_level = self.ai_engine.predict_stress(hr, hrv, eeg, self.center_s, self.wander_s, base_mode)

            if calibrated:
                self.history_stress.append(stress_level)
                self.history_states.append(base_mode)
            
            # --- THE FLOW GATEKEEPER ---
            if base_mode == "Focused":
                if self.flow_warmup < self.WARMUP_THRESHOLD:
                    new_mode = "Focused" 
                    self.flow_warmup += 1
                else:
                    new_mode = "Flow"
            else:
                new_mode = base_mode 
                self.flow_warmup = max(0, self.flow_warmup - 50) 

            # --- AUDIO UPDATES ---
            if new_mode != self.mode:
                self.mode = new_mode
                if self.mode == "Away": 
                    self.audio.pause_music()
                elif self.mode == "Drowsy": 
                    self.audio.play_music()
                    self.audio.set_music_volume(100)
                elif self.mode == "Flow": 
                    self.audio.play_music()
                    self.audio.set_music_volume(50)
                elif self.mode == "Distracted": 
                    self.audio.play_music()
                    self.audio.set_music_volume(5)   
                elif self.mode == "Focused": 
                    self.audio.play_music()
                    self.audio.set_music_volume(40)

            # ==========================================
            # --- HUD RENDERING (CLEAN AND ORGANIZED) ---
            # ==========================================
            res = self.gaze.annotated_frame()
            
            # Semi-transparent dark background for readability
            overlay = res.copy()
            cv2.rectangle(overlay, (10, 10), (520, 220), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.7, res, 0.3, 0, res)

            # Colors
            color_map = {
                "Flow": (255, 100, 255), "Focused": (100, 255, 100), 
                "Distracted": (50, 50, 255), "Drowsy": (0, 165, 255), "Away": (150, 150, 150)
            }
            white = (240, 240, 240)
            gray = (170, 170, 170)
            
            # Row 1: Primary State
            cv2.putText(res, f"STATE: {self.mode.upper()}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[self.mode], 2)
            
            if self.mode == "Focused":
                prog = int((self.flow_warmup / self.WARMUP_THRESHOLD) * 100)
                cv2.putText(res, f"Flow Engine: {prog}%", (260, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

            cv2.line(res, (25, 60), (500, 60), (75, 75, 75), 1)

            # Row 2: Vitals / Biometrics
            hr_text = f"HR:  {hr:.1f} bpm" if calibrated else "HR:  Calibrating..."
            hrv_text = f"HRV: {hrv:.1f} ms" if calibrated else "HRV: Calibrating..."
            
            cv2.putText(res, hr_text, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
            cv2.putText(res, hrv_text, (260, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

            # Row 3: Gaze & Attention
            h_str = f"{h:.2f}" if h else "---"
            v_str = f"{v:.2f}" if v else "---"
            cv2.putText(res, f"Gaze: [H: {h_str} | V: {v_str}]", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1)
            
            attn_pct = int(attention_score * 100)
            attn_color = (100, 255, 100) if attention_score > 0.6 else (50, 50, 255)
            cv2.putText(res, f"Attention: {attn_pct}%", (260, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, attn_color, 1)

            # Row 4: AI Stress Analytics
            cv2.line(res, (25, 160), (500, 160), (75, 75, 75), 1)
            
            stress_pct = int(stress_level * 100)
            stress_color = (50, 50, 255) if stress_level > 0.6 else (100, 255, 100)
            
            # Simple bar chart for stress
            cv2.putText(res, f"Neural Stress:", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
            cv2.rectangle(res, (160, 182), (160 + 100, 198), (50, 50, 50), -1) # Bar background
            cv2.rectangle(res, (160, 182), (160 + stress_pct, 198), stress_color, -1) # Bar fill
            cv2.putText(res, f"{stress_pct}%", (270, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_color, 1)

            cv2.imshow("FocusTrackStress", res)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.webcam.release()
        cv2.destroyAllWindows()
        self.print_session_summary()

if __name__ == "__main__":
    FocusTrackStress().run()