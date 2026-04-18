import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# --- 1. THE MODEL ARCHITECTURE ---
class NeuroStressModel(nn.Module):
    def __init__(self):
        super(NeuroStressModel, self).__init__()
        # Shared layers: Learns the relationship between HRV, Brainwaves, and Gaze
        self.shared = nn.Sequential(
            nn.Linear(8, 64),  # Made slightly wider for 370k rows
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Stress Prediction Head (Continuous 0.0 to 1.0)
        self.stress_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

        # State Prediction Head (Classification: Flow, Focused, Distracted)
        self.state_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3) 
        )

    def forward(self, x):
        features = self.shared(x)
        stress = self.stress_head(features)
        state = self.state_head(features)
        return stress, state

# --- 2. PREPARE DATA (BATCHING) ---
print("📊 Loading massive 370k Unified Dataset...")
df = pd.read_csv("unified_training_data.csv")

# Inputs (8 columns)
X_np = df[['n_hr', 'n_hrv', 'alpha', 'beta', 'theta', 'n_c', 'n_w', 's_enc']].values
X = torch.tensor(X_np, dtype=torch.float32)

# Targets
y_stress_np = df['t_stress'].values
y_stress = torch.tensor(y_stress_np, dtype=torch.float32).view(-1, 1)

y_state_np = df['t_state'].values
y_state = torch.tensor(y_state_np, dtype=torch.long)

# Create a DataLoader to handle the 370,000 rows in batches of 1024
dataset = TensorDataset(X, y_stress, y_state)
# shuffle=True is critical so the AI doesn't memorize the order of the rows!
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True) 

# --- 3. TRAINING ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = NeuroStressModel().to(device)

criterion_reg = nn.MSELoss()      
criterion_cls = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Because we have 370k rows, we don't need 500 epochs. 20 epochs will be plenty!
epochs = 20 

print(f"🚀 Training on {device} using Mini-Batches...")

for epoch in range(epochs):
    model.train()
    total_stress_loss = 0
    total_state_loss = 0
    
    # Process the data 1024 rows at a time
    for batch_X, batch_y_stress, batch_y_state in dataloader:
        batch_X = batch_X.to(device)
        batch_y_stress = batch_y_stress.to(device)
        batch_y_state = batch_y_state.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_stress, pred_state = model(batch_X)
        
        # Calculate loss
        loss_stress = criterion_reg(pred_stress, batch_y_stress)
        loss_state = criterion_cls(pred_state, batch_y_state)
        
        # Total loss (we care slightly more about stress prediction here)
        loss = (loss_stress * 2.0) + loss_state
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_stress_loss += loss_stress.item()
        total_state_loss += loss_state.item()
        
    # Calculate average loss for the epoch
    avg_stress_loss = total_stress_loss / len(dataloader)
    avg_state_loss = total_state_loss / len(dataloader)
    
    print(f"Epoch {epoch+1}/{epochs} | Avg Stress Loss: {avg_stress_loss:.4f} | Avg State Loss: {avg_state_loss:.4f}")

# --- 4. SAVE THE WEIGHTS ---
torch.save(model.state_dict(), "neuro_stress_model.pth")
print("✅ Done! Weights saved to 'neuro_stress_model.pth'")