# FocusTrackStress

An AI-powered productivity system that uses real-time rPPG (webcam-based heart rate), eye-tracking, and neural EEG wave prediction to predict current stress levels and  modulate your environment via Apple Music.

Note: Needs Apple Music and must run on MacOS

## 🚀 Getting Started

Follow these steps to set up your environment and run the application.
Note: It is highly recommended to use a virtual environment to avoid version conflicts with other Python projects.

```bash
# Navigate to the project folder
cd FocusTrackStress

# Create the environment
python3.11 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate

# First, ensure pip is up to date
pip install --upgrade pip

# Install the compatible "Golden Trio" and other requirements
pip install -r requirements.txt

# Run the program
.venv/bin/python src/main.py