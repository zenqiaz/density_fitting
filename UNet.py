import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
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

#print(f"Loaded {len(loaded_data)} tuples.")
#print(f"Shape of first tuple's third element: {loaded_data[0][2].shape}")

#current, peak = tracemalloc.get_traced_memory()
#print(f"Current memory usage: {current / 1024**2:.2f} MB")
#print(f"Peak memory usage: {peak / 1024**2:.2f} MB")

#print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Cached memory by the memory allocator in MB
#print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
# Convert list of tuples containing NumPy arrays to PyTorch tensors
tensor_data = [(torch.from_numpy(x).float(), torch.from_numpy(y).float()) for x, z, y in loaded_data]

# Check the result
#print(tensor_data[0])  # Example: First tuple
#print(type(tensor_data[0][0]), type(tensor_data[0][1]))  # Should print torch.Tensor for both

# Stack inputs and labels into batched tensors
inputs = torch.stack([_ for x, _ in tensor_data])  # Stack all inputs (3x3 arrays)
labels = torch.stack([_ for _, y in tensor_data])  # Stack all labels (2x2 arrays)

#print(inputs.shape)  # Example: torch.Size([3, 3, 3]) if 3 samples
#print(labels.shape)  # Example: torch.Size([3, 2, 2]) if 3 samples


trimmed_labels = labels[:, :-1, :-1]  # Remove the last row and column

# Add a channel dimension by unsqueezing at dimension 1
inputs = inputs.view(2268, 144, 144).unsqueeze(1)  # Shape becomes (2268, 1, 144, 144)
trimmed_labels = trimmed_labels.unsqueeze(1)  # Shape becomes (2268, 1, 143, 143)

labels = labels.unsqueeze(1)  # Shape becomes (2268, 1, 143, 143)

# Create dataset with correct shape
dataset = TensorDataset(inputs, trimmed_labels)

# Create a DataLoader
batch_size = 16  # Define the batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through batches
#for batch_idx, (batch_inputs, batch_labels) in enumerate(dataloader):
#    print(f"Batch {batch_idx + 1}:")
#    print(f"Input batch shape: {batch_inputs.shape}")  # Should be (batch_size, 144, 144)
#    print(f"Label batch shape: {batch_labels.shape}")  # Should be (batch_size, 11, 11)
#    break

#print("Merged tensor shape:", merged_tensor.shape)      # Output: torch.Size([100, 12, 12, 12, 12])
#print("Reshaped tensor shape:", reshaped_tensor.shape)  # Output: torch.Size([100, 144, 144])


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define U-Net model (same as before, no changes to architecture)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        self.dec3 = self.upconv_block(128, 64)
        self.dec2 = self.upconv_block(64, 32)
        self.dec1 = self.upconv_block(32, 16)
        self.output_layer = nn.Conv2d(16, 1, kernel_size=5, stride=12, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        dec3 = self.dec3(enc4) + enc3
        dec2 = self.dec2(dec3) + enc2
        dec1 = self.dec1(dec2) + enc1
        output = self.output_layer(dec1)
        return output

# Initialize the model and move it to the GPU
model = UNet().to(device)

# Create input and target tensors, and move them to the GPU
input_image = torch.randn(1, 1, 144, 144).to(device)  # Batch size 1, single-channel image
target_image = torch.randn(1, 1, 12, 12).to(device)  # Target output

# Forward pass
output_image = model(input_image)
print(output_image.shape)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 16
learning_rate = 0.001
# Training loop
train_losses = []
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0  # Track loss for this epoch

    for batch_idx, (input_image, target_image) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        input_image, target_image = input_image.to(device), target_image.to(device)

        # Forward pass
        output_image = model(input_image)
        loss = criterion(output_image, target_image)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

# Plot the loss curve and save it to a file
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid()

# Save the plot to a file (e.g., PNG format)
plot_filename = "training_loss_curve.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Loss curve saved as {plot_filename}")

# Optional: Clear the plot to avoid overlap in subsequent calls
plt.close()