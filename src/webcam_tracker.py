import cv2
import time
import json
import random
from datetime import datetime
from gaze_tracking import GazeTracking

class FocusScapeTracker:
    def __init__(self):
        # Initialize the GazeTracking module
        self.gaze = GazeTracking()
        
        # Capture video from standard laptop webcam (Index 0)
        self.webcam = cv2.VideoCapture(0)
        
        # --- Data Storage for the AI Summary ---
        self.session_data = {
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "start_time": datetime.now().strftime("%H:%M:%S"),
            "end_time": None,
            "total_duration_seconds": 0,
            "data_log": []
        }
        
        # Tracking metrics
        self.start_time = time.time()
        self.frames_processed = 0
        
        # Simulated HR buffer (pyVHR placeholder for live demo)
        self.current_hr = 72

    def extract_rppg_heart_rate(self):
        """
        Placeholder for pyVHR live processing.
        In a full build, pyVHR extracts BPM from skin color shifts.
        For the hackathon demo, we simulate a realistic fluctuating HR.
        """
        # Add a slight random fluctuation to make the demo look alive
        fluctuation = random.choice([-1, 0, 1, 2, -2])
        self.current_hr = max(60, min(100, self.current_hr + fluctuation))
        return self.current_hr

    def run(self):
        print("🚀 FocusScape 2.0 Tracker Initialized.")
        print("👁️  Looking for webcam feed... Press 'ESC' to end the study session.")
        
        while True:
            # 1. Grab a new frame from the webcam
            ret, frame = self.webcam.read()
            if not ret:
                print("❌ Error: Could not read from webcam.")
                break

            # 2. Send the frame to GazeTracking for analysis
            self.gaze.refresh(frame)

            # 3. Get the annotated frame (draws red target circles on the pupils)
            annotated_frame = self.gaze.annotated_frame()
            
            # --- EXTRACT BIOMETRIC METRICS ---
            is_blinking = self.gaze.is_blinking()
            is_center = self.gaze.is_center()
            is_right = self.gaze.is_right()
            is_left = self.gaze.is_left()
            
            hr_bpm = self.extract_rppg_heart_rate()
            
            # Determine Focus State
            status_text = "Analyzing..."
            status_color = (255, 255, 255) # White
            
            if is_blinking:
                status_text = "Blinking / Eyes Closed 💤"
                status_color = (0, 165, 255) # Orange
            elif is_right or is_left:
                status_text = "Distracted (Wandering) ⚠️"
                status_color = (0, 0, 255) # Red
            elif is_center:
                status_text = "Focused (Flow State) 🎯"
                status_color = (0, 255, 0) # Green

            # --- LOG DATA (Every 30 frames to keep the JSON lightweight) ---
            self.frames_processed += 1
            if self.frames_processed % 30 == 0:
                elapsed_time = round(time.time() - self.start_time, 2)
                
                log_entry = {
                    "timestamp_sec": elapsed_time,
                    "focus_status": status_text.split(" ")[0], # Just log the core word
                    "heart_rate_bpm": hr_bpm
                }
                self.session_data["data_log"].append(log_entry)

            # --- BUILD THE HUD (Heads-Up Display) ---
            # Create a slick semi-transparent background box for text
            cv2.rectangle(annotated_frame, (10, 10), (450, 120), (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, f"Focus State: {status_text}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.putText(annotated_frame, f"Est. Heart Rate (rPPG): {hr_bpm} BPM", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            
            # Show the video feed
            cv2.imshow("FocusScape 2.0 - Biometric Tracker", annotated_frame)

            # Press 'ESC' to exit the loop
            if cv2.waitKey(1) == 27:
                break

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