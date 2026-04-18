import cv2
from gaze_tracking import GazeTracking
from audio_controller import AudioController
from biometrics import BiometricSystem
from classifier import BiometricClassifier

class FocusScape:
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
        print("🧠 FOCUSSCAPE: SESSION ANALYTICS REPORT 🧠")
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
        print("\n💡 AI INSIGHT:")
        if avg_stress > 60 and dist_pct > 30:
            print("   High stress and high distraction detected. You are experiencing cognitive overload. Take a 10-minute break.")
        elif flow_pct > 50:
            print("   Excellent session! You achieved highly sustained Flow states.")
        else:
            print("   Good effort, but your focus was fragmented. Try removing environmental triggers next session.")
        print("="*50 + "\n")

    def run(self):
        print("🚀 FocusScape 2.0: Optimized Logic & AI active.")
        while True:
            ret, frame = self.webcam.read()
            if not ret: break
            
            self.gaze.refresh(frame)

            h, v = self.gaze.horizontal_ratio(), self.gaze.vertical_ratio()
            
            # --- NEW: DETECT HEAD TURNS (One eye missing) ---
            left_pupil = self.gaze.pupil_left_coords()
            right_pupil = self.gaze.pupil_right_coords()
            # Uses an XOR-style check: if one is None but the other isn't, the head is turned
            one_eye_missing = (left_pupil is None and right_pupil is not None) or (right_pupil is None and left_pupil is not None)

            # --- NEW: TIGHTER VERTICAL THRESHOLDS ---
            # Dropped the 'v' max from 0.85 to 0.65. If they look down at a phone, it breaks.
            is_center = (0.43 < h < 0.67) and (0.42 < v < 0.70) if (h and v) else False
            
            # --- HIGH-SENSITIVITY STREAK LOGIC ---
            if not (h or v or self.gaze.is_blinking()):
                self.miss_s = min(self.miss_s + 1, 30) # Cap at 1 sec
                self.wander_s = min(self.wander_s + 2, 25) # CAP: Max 25 (easy to recover)
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
                    self.was_distracted = True # Mark that we left focus
                elif not is_center:
                    self.wander_s = min(self.wander_s + 1, 25)
                    self.center_s = 0
                    self.was_distracted = True # Mark that we left focus
                else:
                    # --- NEW RESET LOGIC ---
                    if self.was_distracted:
                        self.wander_s = 0   # Hard reset wander on first sight
                        self.was_distracted = False
                    
                    self.center_s += 1
                    self.wander_s = max(0, self.wander_s - 3)

            # --- SENSOR FUSION & AI ---
            # --- SENSOR FUSION & CLASSIFIER ---
            hr, hrv, eeg, calibrated = self.bio.process_frame(frame, self.mode)
            
            vector = [self.center_s, self.wander_s, self.blink_s, self.miss_s, hr]
            base_mode = BiometricClassifier.predict_state(vector)
            attention_score = BiometricClassifier.calculate_attention_score(self.center_s, self.wander_s, hrv)
            stress_level, ai_state = self.ai_engine.predict(hr, hrv, eeg, self.center_s, self.wander_s, self.mode)

            if calibrated:
                self.history_stress.append(stress_level)
                self.history_states.append(ai_state)
            
            # --- NEW MULTI-HEAD AI PREDICTION ---
            # Pass HRV and base_mode to the AI, and unpack the 4 returned values
            _, self.fatigue_level, stress_level, time_to_crash = self.ai_engine.predict(
                hr, hrv, eeg, self.center_s, self.wander_s, current_state=base_mode
            )
            
            # --- THE FLOW GATEKEEPER ---
            if base_mode == "Focused":
                if self.flow_warmup < self.WARMUP_THRESHOLD:
                    new_mode = "Focused" # Stay in focused until warmup is done
                    self.flow_warmup += 1
                else:
                    new_mode = "Flow"
            else:
                new_mode = base_mode # Inherit Distracted, Drowsy, or Away
                self.flow_warmup = max(0, self.flow_warmup - 50) # Lose progress if distracted

            # --- AUDIO & STATE UPDATES ---
            if new_mode != self.mode:
                self.mode = new_mode
                if self.mode == "Away": 
                    self.audio.pause_music()
                elif self.mode == "Drowsy": 
                    self.audio.play_music()
                    self.audio.set_music_volume(100) # Blast volume to wake up!
                elif self.mode == "Flow": 
                    self.audio.play_music()
                    self.audio.set_music_volume(50)
                elif self.mode == "Distracted": 
                    self.audio.play_music()
                    self.audio.set_music_volume(5)   # Drop volume to break distraction
                elif self.mode == "Focused": 
                    self.audio.play_music()
                    self.audio.set_music_volume(40)

            # --- HUD RENDERING ---
            color_map = {
                "Flow": (255, 0, 255), "Focused": (0, 255, 0), 
                "Distracted": (0, 0, 255), "Drowsy": (0, 165, 255), "Away": (150, 150, 150)
            }
            res = self.gaze.annotated_frame()
            cv2.rectangle(res, (10, 10), (480, 220), (0, 0, 0), -1)
            
            # State & Warmup
            cv2.putText(res, f"STATE: {self.mode}", (20, 45), 2, 0.8, color_map[self.mode], 2)
            if self.mode == "Focused":
                prog = int((self.flow_warmup / self.WARMUP_THRESHOLD) * 100)
                cv2.putText(res, f"Flow Warmup: {prog}%", (250, 45), 2, 0.5, (255, 255, 255), 1)

            # Debugging Gaze (So you can see why it's not focused)
            h_str = f"{h:.2f}" if h else "N/A"
            v_str = f"{v:.2f}" if v else "N/A"
            cv2.putText(res, f"Gaze H:{h_str} V:{v_str} | Blink:{self.blink_s}", (20, 90), 2, 0.5, (200, 200, 200), 1)
            
            # Display HR
            cv2.putText(res, f"HR: {hr if calibrated else 'Calibrating...'}", (20, 130), 2, 0.6, (255, 255, 255), 1)
            
            # Display HRV
            cv2.putText(res, f"HRV: {hrv if calibrated else '...'} ms", (250, 130), 2, 0.6, (255, 255, 255), 1)
            cv2.putText(res, f"Fatigue Projection: {int(self.fatigue_level * 100)}%", (20, 170), 2, 0.6, (0, 255, 255), 1)
            cv2.putText(res, f"Stress Level: {int(stress_level * 100)}%", (250, 170), 2, 0.6, (0, 0, 255) if stress_level > 0.6 else (0, 255, 0), 1)
            cv2.putText(res, f"Next Focus Crash in: {int(time_to_crash)} mins", (20, 200), 2, 0.6, (255, 100, 100), 1)

            attn_color = (0, 255, 0) if attention_score > 0.6 else (0, 0, 255)
            cv2.putText(res, f"Attention Score: {int(attention_score * 100)}%", (250, 200), 2, 0.6, attn_color, 1)

            stress_color = (0, 0, 255) if stress_level > 0.6 else (0, 255, 0)
            cv2.putText(res, f"AI Stress: {int(stress_level * 100)}%", (250, 170), 2, 0.6, stress_color, 1)
            
            cv2.imshow("FocusScape 2.0", res)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.webcam.release()
        cv2.destroyAllWindows()
        self.print_session_summary()

if __name__ == "__main__":
    FocusScape().run()