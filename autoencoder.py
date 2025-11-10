import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import numpy as np
from numpy import array
import pickle
import torch.optim as optim
from tqdm import tqdm  # For progress bar
import os
import matplotlib.pyplot as plt
import tracemalloc

#tracemalloc.start()
loaded_data = []
with open('RDM2_dataset.pkl', 'rb') as f:
    while True:
        try:
            t = pickle.load(f)  # Load one tuple at a time
            loaded_data.append(t)
        except EOFError:
            break  # End of file reached


# Convert list of tuples containing NumPy arrays to PyTorch tensors
tensor_data = [(torch.from_numpy(x).float(), torch.from_numpy(y).float()) for x, z, y in loaded_data]

inputs = torch.stack([_ for x, _ in tensor_data])  # Stack all inputs (3x3 arrays)
labels = torch.stack([_ for _, y in tensor_data])  # Stack all labels (2x2 arrays)


inputs = inputs.view(2268, 144, 144).unsqueeze(1)  # Shape becomes (2268, 1, 144, 144)

labels = labels.unsqueeze(1)  # Shape becomes (2268, 1, 143, 143)
dataset = TensorDataset(inputs, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16  # Define the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training setup

# Training loop
num_epochs = 80
train_recon_losses = []
train_label_losses = []
val_recon_losses = []
val_label_losses = []



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = UNetBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(128, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        
        # Output for reconstruction
        self.out_recon = nn.Conv2d(64, 1, kernel_size=1)
        
        # Output for 12x12 label
        self.label_out = nn.Sequential(
            nn.AdaptiveAvgPool2d((12, 12)),
            nn.Conv2d(256, 1, kernel_size=1)  # Assuming 1 label channel
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        # Label output
        label = self.label_out(bottleneck)
        
        # Decoder
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        
        recon = self.out_recon(dec2)
        
        return recon, label

# Example usage
model = UNetAutoencoder()

# Load the saved parameters
#model.load_state_dict(torch.load('PINN_autoencoder_model.pth'))
def count_all_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

#total, trainable = count_all_parameters(model)
#print(f"Total parameters: {total:,}")
#print(f"Trainable parameters: {trainable:,}")
#x = torch.randn(1, 1, 144, 144)  # Example input with 1 channel
#recon, label = model(x)
#print(recon.shape, label.shape)  # (1, 1, 144, 144), (1, 1, 12, 12)
criterion_recon = nn.MSELoss()
criterion_label = nn.MSELoss()
recon_weight = 0.7  # Weight for reconstruction loss
label_weight = 0.3  # Weight for label loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    model.train()
    epoch_train_recon_loss = 0
    epoch_train_label_loss = 0
    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        recon, label = model(batch_inputs)
        
        loss_recon = criterion_recon(recon, batch_inputs)
        loss_label = criterion_label(label, batch_labels)
        
        loss = recon_weight * loss_recon + label_weight * loss_label
        loss.backward()
        optimizer.step()
        
        epoch_train_recon_loss += loss_recon.item()
        epoch_train_label_loss += loss_label.item()
    
    train_recon_losses.append(epoch_train_recon_loss / len(train_loader))
    train_label_losses.append(epoch_train_label_loss / len(train_loader))
    
    model.eval()
    epoch_val_recon_loss = 0
    epoch_val_label_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            recon, label = model(batch_inputs)
            loss_recon = criterion_recon(recon, batch_inputs)
            loss_label = criterion_label(label, batch_labels)
            
            epoch_val_recon_loss += loss_recon.item()
            epoch_val_label_loss += loss_label.item()
    
    val_recon_losses.append(epoch_val_recon_loss / len(val_loader))
    val_label_losses.append(epoch_val_label_loss / len(val_loader))
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Recon Loss: {train_recon_losses[-1]:.4f}, Train Label Loss: {train_label_losses[-1]:.4f}, Val Recon Loss: {val_recon_losses[-1]:.4f}, Val Label Loss: {val_label_losses[-1]:.4f}")

torch.save(model.state_dict(), 'PINN_autoencoder_model.pth')
# Plotting loss curves
plt.figure()
plt.plot(range(1, num_epochs+1), train_recon_losses, label='Train Recon Loss')
plt.plot(range(1, num_epochs+1), val_recon_losses, label='Val Recon Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Reconstruction Loss Curves')
plt.legend()
plt.savefig('reconstruction_loss_curve.png')

plt.figure()
plt.plot(range(1, num_epochs+1), train_label_losses, label='Train Label Loss')
plt.plot(range(1, num_epochs+1), val_label_losses, label='Val Label Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Label Loss Curves')
plt.legend()
plt.savefig('label_loss_curve.png')
