import cv2
import time
import json
import random
import subprocess
from datetime import datetime
from gaze_tracking import GazeTracking

class EmotivCortexMock:
    """Mock implementation of the Emotiv Cortex Websocket API."""
    def __init__(self):
        print("🧠 [Emotiv Cortex API] Connecting to headset (MOCK MODE)...")
        self.is_connected = True
        
    def get_band_power(self, current_mode):
        """Simulates live EEG frequency bands based on the user's physical state."""
        eeg_alpha = random.uniform(0.4, 0.6) # Relaxation
        eeg_beta = random.uniform(0.4, 0.6)  # Stress/Focus
        eeg_theta = random.uniform(0.2, 0.4) # Drowsiness
        
        if current_mode == "Flow":
            eeg_alpha = random.uniform(0.7, 0.9) # High Alpha (Flow)
            eeg_beta = random.uniform(0.2, 0.3)
        elif current_mode == "Focused": # NEW: Standard working state
            eeg_alpha = random.uniform(0.5, 0.6) # Balanced
            eeg_beta = random.uniform(0.6, 0.8)  # Active thinking
        elif current_mode == "Distracted":
            eeg_alpha = random.uniform(0.2, 0.4)
            eeg_beta = random.uniform(0.7, 0.9)  # High Beta (Agitation)
        elif current_mode == "Drowsy":
            eeg_theta = random.uniform(0.7, 0.9) # High Theta (Sleepy)
            eeg_alpha = random.uniform(0.2, 0.4)
            eeg_beta = random.uniform(0.1, 0.2)
            
        return {
            "alpha": round(eeg_alpha, 2),
            "beta": round(eeg_beta, 2),
            "theta": round(eeg_theta, 2)
        }

class FocusScapeTracker:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.emotiv_api = EmotivCortexMock() # Connect to Mock Headset
        
        self.session_data = {
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "start_time": datetime.now().strftime("%H:%M:%S"),
            "end_time": None,
            "total_duration_seconds": 0,
            "data_log": []
        }
        
        self.start_time = time.time()
        self.frames_processed = 0
        self.current_hr = 72
        
        # --- ROLLING BUFFERS ---
        self.center_streak = 0
        self.wander_streak = 0
        self.blink_streak = 0
        self.missing_streak = 0 
        self.current_mode = "None"

    def extract_rppg_heart_rate(self):
        """Simulated rPPG that reacts to the user's current focus state."""
        target_hr = 72 
        
        if self.current_mode == "Flow":
            target_hr = random.randint(60, 65) # Calm
        elif self.current_mode == "Distracted":
            target_hr = random.randint(80, 88) # Elevated
        elif self.current_mode == "Drowsy":
            target_hr = random.randint(55, 60) # Resting

        # Smooth drift towards the target
        if self.current_hr < target_hr:
            self.current_hr += 1
        elif self.current_hr > target_hr:
            self.current_hr -= 1
            
        return self.current_hr

    # --- APPLE MUSIC CONTROLS ---
    def get_current_song(self):
        """Silently asks Apple Music what is playing via AppleScript."""
        try:
            is_running = subprocess.run(['osascript', '-e', 'application "Music" is running'], capture_output=True, text=True).stdout.strip()
            if is_running == 'true':
                track = subprocess.run(['osascript', '-e', 'tell application "Music" to name of current track'], capture_output=True, text=True).stdout.strip()
                artist = subprocess.run(['osascript', '-e', 'tell application "Music" to artist of current track'], capture_output=True, text=True).stdout.strip()
                if track and artist:
                    return f"{track} - {artist}"
        except Exception:
            pass
        return "None / Silence"

    def pause_music(self):
        try:
            subprocess.run(['osascript', '-e', 'tell application "Music" to pause'])
        except Exception:
            pass

    def play_music(self):
        try:
            subprocess.run(['osascript', '-e', 'tell application "Music" to play'])
        except Exception:
            pass

    def set_music_volume(self, target_level):
        """Fades Apple Music volume very smoothly in the background."""
        applescript = f"""
        tell application "Music"
            set targetVol to {target_level}
            set currentVol to sound volume
            
            if currentVol < targetVol then
                repeat while currentVol < targetVol
                    set currentVol to currentVol + 1
                    if currentVol > targetVol then set currentVol to targetVol
                    set sound volume to currentVol
                    delay 0.1
                end repeat
            else if currentVol > targetVol then
                repeat while currentVol > targetVol
                    set currentVol to currentVol - 1
                    if currentVol < targetVol then set currentVol to targetVol
                    set sound volume to currentVol
                    delay 0.1
                end repeat
            end if
        end tell
        """
        try:
            subprocess.Popen(['osascript', '-e', applescript])
        except Exception:
            pass

    # --- MAIN LOOP ---
    def run(self):
        print("🚀 FocusScape 2.0 Tracker Initialized. (Apple Music DJ Active!)")
        print("👁️  Looking for webcam feed... Click the video window and press 'ESC' to end.")
        
        try:
            while True:
                ret, frame = self.webcam.read()
                if not ret:
                    break

                h_ratio = None
                v_ratio = None

                try:
                    self.gaze.refresh(frame)
                    annotated_frame = self.gaze.annotated_frame()
                    is_blinking = self.gaze.is_blinking()
                    
                    h_ratio = self.gaze.horizontal_ratio()
                    v_ratio = self.gaze.vertical_ratio()
                    
                    if h_ratio is None or v_ratio is None:
                        is_center = False
                    else:
                        is_h_center = (0.45 < h_ratio < 0.65)
                        is_v_center = (0.40 < v_ratio < 0.70) 
                        
                        is_center = is_h_center and is_v_center

                except Exception:
                    annotated_frame = frame.copy()
                    is_blinking = False
                    is_center = False
                
                # --- CHECK IF FACE IS VISIBLE ---
                face_visible = is_blinking or is_center or (h_ratio is not None)
                
                # --- TEMPORAL LOGIC (Capped Streaks & Fast Recovery!) ---
                if not face_visible:
                    # Cap at 20 so it never stacks to infinity
                    self.wander_streak = min(20, self.wander_streak + 2) 
                    self.missing_streak = min(20, self.missing_streak + 1)
                    self.center_streak = max(0, self.center_streak - 5)
                    self.blink_streak = 0
                else:
                    self.missing_streak = max(0, self.missing_streak - 2)
                    
                    if is_blinking:
                        self.blink_streak = min(15, self.blink_streak + 1) # Cap blink stacking
                    else:
                        self.blink_streak = max(0, self.blink_streak - 1) 
                        
                    # FOCUS VS WANDERING
                    if is_center and not is_blinking:
                        self.center_streak += 1
                        # Drains massively fast so you immediately escape "Distracted"
                        self.wander_streak = max(0, self.wander_streak - 5) 
                    elif not is_center and not is_blinking:
                        # Cap wandering at 20 so you aren't trapped in it
                        self.wander_streak = min(20, self.wander_streak + 1)
                        self.center_streak = max(0, self.center_streak - 1)

                # --- STATE SELECTION & DYNAMIC AUDIO ---
                status_text = "Analyzing..."
                status_color = (255, 255, 255)
                
                if self.missing_streak > 50: 
                    status_text = "User Away / Off-Screen 👻"
                    status_color = (150, 150, 150) 
                    if self.current_mode != "Away": # Only fire when state changes!
                        self.current_mode = "Away"
                        self.pause_music() 
                    
                elif self.blink_streak > 7:
                    status_text = "Eyes Closed / Drowsy 💤"
                    status_color = (0, 165, 255) 
                    if self.current_mode != "Drowsy":
                        self.current_mode = "Drowsy"
                        self.play_music()
                        self.set_music_volume(90) 
                    
                elif self.wander_streak > 10:
                    status_text = "Distracted (Wandering) ⚠️"
                    status_color = (0, 0, 255) 
                    if self.current_mode != "Distracted":
                        self.current_mode = "Distracted"
                        self.set_music_volume(15) 
                    
                elif self.center_streak > 45:
                    status_text = "Deep Flow State 🧠⚡"
                    status_color = (255, 0, 255) 
                    if self.current_mode != "Flow":
                        self.current_mode = "Flow"
                        self.set_music_volume(75) 
                    
                else:
                    status_text = "Focused (Active) 👀"
                    status_color = (0, 255, 0) 
                    if self.current_mode != "Focused":
                        self.current_mode = "Focused"
                        self.play_music()
                        self.set_music_volume(50)

                # --- LOGGING & HUD ---
                hr_bpm = self.extract_rppg_heart_rate()
                eeg_data = self.emotiv_api.get_band_power(self.current_mode)
                
                self.frames_processed += 1
                if self.frames_processed % 30 == 0:
                    elapsed_time = round(time.time() - self.start_time, 2)
                    current_song = self.get_current_song()
                    
                    log_entry = {
                        "timestamp_sec": elapsed_time,
                        "focus_status": status_text.split(" ")[0], 
                        "heart_rate_bpm": hr_bpm,
                        "eeg_alpha": eeg_data["alpha"],
                        "eeg_beta": eeg_data["beta"],
                        "eeg_theta": eeg_data["theta"],
                        "background_audio": current_song
                    }
                    self.session_data["data_log"].append(log_entry)

                cv2.rectangle(annotated_frame, (10, 10), (450, 220), (0, 0, 0), -1) 
                
                cv2.putText(annotated_frame, f"State: {status_text}", (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.putText(annotated_frame, f"rPPG HR: {hr_bpm} BPM", (20, 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                            
                cv2.putText(annotated_frame, f"EEG | Alpha: {eeg_data['alpha']} | Beta: {eeg_data['beta']}", 
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
                            
                cv2.putText(annotated_frame, f"DEV | C: {self.center_streak} | W: {self.wander_streak} | B: {self.blink_streak}", 
                            (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            
                h_text = f"{h_ratio:.2f}" if h_ratio else "None"
                v_text = f"{v_ratio:.2f}" if v_ratio else "None"
                cv2.putText(annotated_frame, f"RAW | H: {h_text} | V: {v_text}", 
                            (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow("FocusScape 2.0 - Biometric Tracker", annotated_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Session interrupted by user (Ctrl+C).")
        finally:
            self.end_session()

    def end_session(self):
        self.webcam.release()
        cv2.destroyAllWindows()
        
        self.session_data["end_time"] = datetime.now().strftime("%H:%M:%S")
        self.session_data["total_duration_seconds"] = round(time.time() - self.start_time, 2)
        
        output_file = "docs/session_log.json"
        
        import os
        os.makedirs("docs", exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(self.session_data, f, indent=4)
            
        print(f"\n✅ Session Complete! Data saved to {output_file}.")
        print("🧠 Ready to pass this to Gemini for the End-of-Session AI Report!")

if __name__ == "__main__":
    tracker = FocusScapeTracker()
    tracker.run()
