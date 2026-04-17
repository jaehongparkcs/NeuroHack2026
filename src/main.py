import cv2
import time
from gaze_tracking import GazeTracking
from audio_controller import AudioController
from biometrics import BiometricSystem
from classifier import BiometricClassifier
from ai_engine import AIEngine 

class FocusScape:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.audio = AudioController()
        self.bio = BiometricSystem()
        self.ai_engine = AIEngine()
        
        self.mode = "Focused"
        self.fatigue_level = 0.0
        self.flow_warmup = 0 
        self.WARMUP_THRESHOLD = 150 # ~5 seconds to "earn" Flow state
        
        # Streaks
        self.center_s, self.wander_s, self.blink_s, self.miss_s = 0, 0, 0, 0

    def run(self):
        print("🚀 FocusScape 2.0: Optimized Logic & AI active.")
        while True:
            ret, frame = self.webcam.read()
            if not ret: break
            
            self.gaze.refresh(frame)
            h, v = self.gaze.horizontal_ratio(), self.gaze.vertical_ratio()
            
            # --- FIX 1: WIDER CENTER THRESHOLDS ---
            # Expanded from .4-.6 to .35-.65 to be more forgiving of head tilt
            is_center = (0.35 < h < 0.65) and (0.35 < v < 0.85) if (h and v) else False
            
            # --- FIX 2: IMPROVED MUTUALLY EXCLUSIVE LOGIC ---
            if not (h or v or self.gaze.is_blinking()):
                self.miss_s += 1
                self.center_s = self.wander_s = self.blink_s = 0
            elif self.gaze.is_blinking():
                self.blink_s += 1
                # We DON'T reset center/wander here so a quick blink doesn't break a streak
                self.miss_s = 0
            else:
                # EYES ARE OPEN: Reset blink counter immediately
                self.blink_s = 0 
                self.miss_s = 0
                if is_center:
                    self.center_s += 1
                    self.wander_s = 0
                else:
                    self.wander_s += 1
                    self.center_s = 0

            # --- SENSOR FUSION & AI ---
            hr, eeg, calibrated = self.bio.process_frame(frame, self.mode)
            ai_mode, self.fatigue_level = self.ai_engine.predict(hr, eeg, self.center_s, self.wander_s)
            
            # --- FIX 3: THE FLOW GATEKEEPER ---
            if ai_mode == "Flow":
                if self.flow_warmup < self.WARMUP_THRESHOLD:
                    ai_mode = "Focused" # Stay in focused until warmup is done
                    self.flow_warmup += 1
            elif ai_mode == "Focused":
                self.flow_warmup += 1
            else:
                self.flow_warmup = max(0, self.flow_warmup - 5) # Lose progress if distracted

            # Final Override Logic
            if self.miss_s > 15:
                new_mode = "Away"
            elif self.blink_s > 40: # Only go Drowsy if eyes are shut for ~1.5 seconds
                new_mode = "Drowsy"
            else:
                new_mode = ai_mode

            # --- AUDIO & STATE UPDATES ---
            if new_mode != self.mode:
                self.mode = new_mode
                if self.mode == "Away": self.audio.pause_music()
                elif self.mode == "Drowsy": self.audio.set_music_volume(100)
                elif self.mode == "Flow": self.audio.set_music_volume(65)
                elif self.mode == "Distracted": self.audio.set_music_volume(20)
                elif self.mode == "Focused": self.audio.set_music_volume(45)

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
            
            cv2.putText(res, f"HR: {hr if calibrated else 'Calibrating...'}", (20, 130), 2, 0.6, (255, 255, 255), 1)
            cv2.putText(res, f"Fatigue Projection: {int(self.fatigue_level * 100)}%", (20, 170), 2, 0.6, (0, 255, 255), 1)
            
            cv2.imshow("FocusScape 2.0", res)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    FocusScape().run()