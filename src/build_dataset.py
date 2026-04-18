import pandas as pd
import numpy as np
import os

print("🧬 Initializing Multi-Subject Data Fusion Protocol...")

# --- CONFIGURATION ---
# Set the name of the folder where your EEG subject files are stored
DATA_DIR = "data/eeg_subjects"

# Create the directory if it doesn't exist (just so the script doesn't crash)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"📁 Created folder '{DATA_DIR}'. Please move your EEG CSVs there!")
# ---------------------

def load_all_subjects(prefix):
    """Searches a specific directory for files starting with a prefix."""
    all_subject_data = []
    
    # List all files in the specific directory
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith(".csv")]
    
    if not files:
        print(f"⚠️ Warning: No files starting with '{prefix}' found in '{DATA_DIR}'")
        return None
        
    print(f"   -> Found {len(files)} files for {prefix} in '{DATA_DIR}'...")
    
    for file_name in files:
        try:
            # Construct the full path
            file_path = os.path.join(DATA_DIR, file_name)
            data = pd.read_csv(file_path).values[:, 1:] # Drop index column
            all_subject_data.append(data)
        except Exception as e:
            print(f"      Error reading {file_name}: {e}")
            
    if all_subject_data:
        return np.vstack(all_subject_data)
    return None

# 2. Load the Heart Data (usually kept in the main folder)
try:
    df_hr = pd.read_csv("data/train.csv")
    
    # Load subjects from the DATA_DIR
    eeg_relax = load_all_subjects("Relax_sub_")
    eeg_math = load_all_subjects("Arithmetic_sub_")
    eeg_stroop = load_all_subjects("Stroop_sub_")
    
    if eeg_relax is None or eeg_math is None or eeg_stroop is None:
        print(f"❌ Error: Ensure your CSVs are inside the '{DATA_DIR}' folder.")
        exit()
        
    print(f"✅ Data Fusion Complete. Ready to process {len(df_hr)} heart-rate samples.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# 3. Fast Fourier Transform (FFT) to extract Brainwaves
def extract_brainwaves(eeg_chunk, fs=128):
    """Converts raw voltage over time into Alpha, Beta, Theta frequency bands."""
    signal = eeg_chunk[:, 0] # Using Channel 0 
    
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0/fs)
    
    theta = np.mean(fft_vals[(freqs >= 4) & (freqs <= 8)])
    alpha = np.mean(fft_vals[(freqs >= 8) & (freqs <= 12)])
    beta = np.mean(fft_vals[(freqs >= 12) & (freqs <= 30)])
    
    total = theta + alpha + beta + 1e-5
    return alpha/total, beta/total, theta/total

unified_data = []

print("⚙️ Fusing datasets and calculating FFTs... This might take a minute.")

# 4. Stitch them together to create a "Virtual Population"
for index, row in df_hr.iterrows():
    hr = row['HR']
    hrv = row['RMSSD']
    condition = row['condition']
    
    chunk_size = 256 # 2 seconds of EEG data at 128Hz
    
    if condition == "no stress":
        state_idx, state_encoded = 0, 1.0 # Flow
        c_streak = np.random.uniform(30, 60)
        w_streak = np.random.uniform(0, 5)
        
        start = np.random.randint(0, len(eeg_relax) - chunk_size)
        alpha, beta, theta = extract_brainwaves(eeg_relax[start:start+chunk_size])
        stress_target, crash_target = 0.1, 0.9

    elif condition == "time pressure":
        state_idx, state_encoded = 1, 0.8 # Focused
        c_streak = np.random.uniform(10, 40)
        w_streak = np.random.uniform(0, 10)
        
        start = np.random.randint(0, len(eeg_math) - chunk_size)
        alpha, beta, theta = extract_brainwaves(eeg_math[start:start+chunk_size])
        stress_target, crash_target = 0.4, 0.6

    else: # "interruption"
        state_idx, state_encoded = 2, 0.2 # Distracted
        c_streak = np.random.uniform(0, 5)
        w_streak = np.random.uniform(15, 30)
        
        start = np.random.randint(0, len(eeg_stroop) - chunk_size)
        alpha, beta, theta = extract_brainwaves(eeg_stroop[start:start+chunk_size])
        stress_target, crash_target = 0.85, 0.1
        
    fatigue_target = np.clip(1.0 - (hrv / 100.0) + (beta * 0.5), 0, 1)

    # Normalize inputs exactly how the PyTorch engine expects them
    norm_hr = (hr - 70) / 20.0
    norm_hrv = (hrv - 50) / 20.0
    norm_c = c_streak / 50.0
    norm_w = w_streak / 20.0

    unified_data.append([
        norm_hr, norm_hrv, alpha, beta, theta, norm_c, norm_w, state_encoded, 
        state_idx, fatigue_target, stress_target, crash_target
    ])

# Save the final cross-subject dataset
df_final = pd.DataFrame(unified_data, columns=['n_hr', 'n_hrv', 'alpha', 'beta', 'theta', 'n_c', 'n_w', 's_enc', 
                                 't_state', 't_fatigue', 't_stress', 't_crash'])
df_final.to_csv("unified_training_data.csv", index=False)
print(f"🎉 Success! Generated 'unified_training_data.csv' across all subjects with {len(df_final)} rows.")