import os
# Force PyTorch to avoid searching for GPUs/Metal and keep it on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()
        # Inputs: [HR, EEG_Alpha, EEG_Beta, EEG_Theta, Center_Streak, Wander_Streak]
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 8)
        
        # Head 1: State Classifier (Flow, Focused, Distracted, Drowsy)
        self.classifier = nn.Linear(8, 4)
        
        # Head 2: Fatigue Projection (0.0 to 1.0)
        self.projector = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        state_logits = self.classifier(x)
        fatigue_score = torch.sigmoid(self.projector(x))
        
        return state_logits, fatigue_score

class AIEngine:
    def __init__(self):
        self.model = FocusNet()
        self.model.eval() 
        self.states = ["Flow", "Focused", "Distracted", "Drowsy"]
        self.start_time = time.time() # Track how long the session has lasted

    def predict(self, hr, eeg, c_streak, w_streak):
        # 1. Calculate Session Duration (Mental Fatigue increases over time)
        session_minutes = (time.time() - self.start_time) / 60
        time_factor = min(0.4, session_minutes / 60) # Adds up to 0.4 to fatigue over an hour

        # 2. Prepare the tensor
        val_hr = float(hr) if isinstance(hr, (int, float)) else 70.0
        input_tensor = torch.tensor([[
            (val_hr - 70) / 20.0, 
            eeg['alpha'], eeg['beta'], eeg['theta'],
            c_streak / 50.0, 
            w_streak / 20.0
        ]], dtype=torch.float32)

        with torch.no_grad():
            logits, fatigue = self.model(input_tensor)
            
        # 3. COMBINE AI + REAL-TIME DRIFT
        # We take the AI's base guess and add the time_factor + HR volatility
        hr_volatility = abs(val_hr - 75) / 50
        combined_fatigue = torch.clamp(fatigue + time_factor + hr_volatility, 0, 1).item()

        state_idx = torch.argmax(logits, dim=1).item()
        
        # If HR is very high or streaks are low, force the state toward 'Distracted'
        if hr_volatility > 0.3 and state_idx == 0: # If AI says Flow but HR is racing
            state_idx = 1 # Downgrade to Focused

        return self.states[state_idx], combined_fatigue