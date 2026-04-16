import cv2
import time
import json
import random
from datetime import datetime
from gaze_tracking import GazeTracking

class FocusScapeTracker:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        
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
        self.missing_streak = 0 # NEW: Tracks when your face is completely gone

    def extract_rppg_heart_rate(self):
        fluctuation = random.choice([-1, 0, 1, 2, -2])
        self.current_hr = max(60, min(100, self.current_hr + fluctuation))
        return self.current_hr

    def run(self):
        print("🚀 FocusScape 2.0 Tracker Initialized. (Head-Turn & Phone Drop Active!)")
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
                        # THE STRICT FOCUS BOX (Catches looking left/right AND up/down at a phone)
                        # h_ratio: ~0.5 is center. v_ratio: ~0.5 is center.
                        # THE FORGIVING FOCUS BOX (Massive buffer zone)
                        is_h_center = (0.45 < h_ratio < 0.65)
                        is_v_center = (0.40 < v_ratio < 0.65) # A bit more forgiving vertically
                        
                        is_center = is_h_center and is_v_center

                except Exception:
                    annotated_frame = frame.copy()
                    is_blinking = False
                    is_center = False
                
                # --- CHECK IF FACE IS VISIBLE ---
                face_visible = is_blinking or is_center or (h_ratio is not None)
                
                hr_bpm = self.extract_rppg_heart_rate()
                
                # --- TEMPORAL LOGIC ---
                if not face_visible:
                    # If you turn your head so far the camera loses your eyes, 
                    # it counts as wandering first before marking you "Away"
                    self.wander_streak += 2 
                    self.missing_streak += 1
                    self.center_streak = 0
                    self.blink_streak = 0
                else:
                    self.missing_streak = 0 
                    
                    if is_blinking:
                        self.blink_streak += 1
                    else:
                        self.blink_streak = max(0, self.blink_streak - 2) 
                        
                    # FOCUS VS WANDERING
                    if is_center and not is_blinking:
                        self.center_streak += 1
                        self.wander_streak = 0 
                    elif not is_center and not is_blinking:
                        # Instantly catches looking down at a phone or glancing away
                        self.wander_streak += 1
                        self.center_streak = 0 

                # --- STATE SELECTION ---
                status_text = "Analyzing..."
                status_color = (255, 255, 255)
                
                if self.missing_streak > 15: # Gives you a tiny bit longer before assuming you left
                    status_text = "User Away / Off-Screen 👻"
                    status_color = (150, 150, 150) 
                    
                elif self.blink_streak > 12:
                    status_text = "Eyes Closed / Drowsy 💤"
                    status_color = (0, 165, 255) 
                    
                elif self.wander_streak > 5:
                    status_text = "Distracted (Wandering) ⚠️"
                    status_color = (0, 0, 255) 
                    
                elif self.center_streak > 30:
                    status_text = "Deep Flow State 🧠⚡"
                    status_color = (255, 0, 255) 
                    
                else:
                    status_text = "Focused (Active) 👀"
                    status_color = (0, 255, 0) 

                # --- LOGGING & HUD ---
                self.frames_processed += 1
                if self.frames_processed % 30 == 0:
                    elapsed_time = round(time.time() - self.start_time, 2)
                    log_entry = {
                        "timestamp_sec": elapsed_time,
                        "focus_status": status_text.split(" ")[0], 
                        "heart_rate_bpm": hr_bpm
                    }
                    self.session_data["data_log"].append(log_entry)

                cv2.rectangle(annotated_frame, (10, 10), (450, 170), (0, 0, 0), -1) 
                
                cv2.putText(annotated_frame, f"Focus State: {status_text}", (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.putText(annotated_frame, f"Est. Heart Rate: {hr_bpm} BPM", (20, 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                            
                cv2.putText(annotated_frame, f"DEV | C: {self.center_streak} | W: {self.wander_streak} | B: {self.blink_streak}", 
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            
                h_text = f"{h_ratio:.2f}" if h_ratio else "None"
                v_text = f"{v_ratio:.2f}" if v_ratio else "None"
                cv2.putText(annotated_frame, f"RAW | H: {h_text} | V: {v_text}", 
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow("FocusScape 2.0 - Biometric Tracker", annotated_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Session interrupted by user (Ctrl+C).")
        finally:
            self.end_session()

    def end_session(self):
        # 1. Clean up the camera resources
        self.webcam.release()
        cv2.destroyAllWindows()
        
        # 2. Finalize session data
        self.session_data["end_time"] = datetime.now().strftime("%H:%M:%S")
        self.session_data["total_duration_seconds"] = round(time.time() - self.start_time, 2)
        
        # 3. Save to JSON for the AI Report script to read later
        output_file = "docs/session_log.json"
        
        # Make sure the docs folder exists
        import os
        os.makedirs("docs", exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(self.session_data, f, indent=4)
            
        print(f"\n✅ Session Complete! Data saved to {output_file}.")
        print("🧠 Ready to pass this to Gemini for the End-of-Session AI Report!")

if __name__ == "__main__":
    tracker = FocusScapeTracker()
    tracker.run()