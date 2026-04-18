import torch
import torch.nn as nn

class NeuroStressModel(nn.Module):
    def __init__(self):
        super(NeuroStressModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.stress_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid() 
        )

    def forward(self, x):
        features = self.shared(x)
        return self.stress_head(features) # Returns a single tensor now

class AIEngine:
    def __init__(self, model_path="neuro_stress_model.pth"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = NeuroStressModel().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"🧠 AI Engine: Weights loaded successfully on {self.device}")
        except Exception as e:
            print(f"⚠️ AI Engine: No weights found. Error: {e}")

    def predict_stress(self, hr, hrv, eeg, c_streak, w_streak, current_mode):
        val_hr = (float(hr) - 70) / 20.0
        val_hrv = (float(hrv) - 50) / 20.0
        s_enc = 1.0 if current_mode in ["Focused", "Flow"] else 0.2
        
        input_vec = torch.tensor([[
            val_hr, val_hrv, 
            eeg.get('alpha', 0.5), eeg.get('beta', 0.5), eeg.get('theta', 0.5),
            c_streak / 50.0, w_streak / 20.0, s_enc
        ]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Only one output comes from the model now
            stress_tensor = self.model(input_vec) 
            stress_val = stress_tensor.item()
            
        return stress_val