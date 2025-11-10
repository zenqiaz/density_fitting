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
from sklearn.metrics import mean_squared_error, mean_absolute_error

#tracemalloc.start()
loaded_data = []
with open('RDM2_dataset.pkl', 'rb') as f:
    while True:
        try:
            t = pickle.load(f)  # Load one tuple at a time
            loaded_data.append(t)
        except EOFError:
            break  # End of file reached


num_epochs = 150
batch_size = 16
learning_rate = 0.0005
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
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
labels = labels.unsqueeze(1)  # Shape becomes (2268, 1, 143, 143)

# Create dataset with correct shape
dataset = TensorDataset(inputs, labels)


dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size  # Ensure the sum equals dataset size

# Split dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
#input_image = torch.randn(1, 1, 144, 144).to(device)  # Batch size 1, single-channel image
#target_image = torch.randn(1, 1, 11, 11).to(device)  # Target output

# Forward pass
#output_image = model(input_image)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop

# Track losses and validation scores
train_losses, val_losses, val_scores = [], [], []

# Training loop with validation
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0

    # Training phase
    for input_image, target_image in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
        input_image, target_image = input_image.to(device), target_image.to(device)

        # Forward pass
        output_image = model(input_image)
        loss = criterion(output_image, target_image)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_targets, all_predictions = [], []

    with torch.no_grad():  # Disable gradient computation for validation
        for input_image, target_image in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            input_image, target_image = input_image.to(device), target_image.to(device)

            output_image = model(input_image)
            loss = criterion(output_image, target_image)
            val_loss += loss.item()

            # Store predictions and targets for validation score calculation
            all_targets.extend(target_image.cpu().numpy().flatten())
            all_predictions.extend(output_image.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Calculate validation score (Mean Absolute Error in this case)
    val_score = mean_absolute_error(all_targets, all_predictions)
    val_scores.append(val_score)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_score:.4f}")

# Plot loss and validation score curves and save
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
print("Loss curve saved as loss_curve.png")
plt.close()

# Plot validation score curve and save
plt.figure()
plt.plot(range(1, num_epochs + 1), val_scores, label="Validation MAE", marker='s', color='red')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Validation Score (MAE)')
plt.legend()
plt.grid()
plt.savefig("validation_score_curve.png", dpi=300, bbox_inches='tight')
print("Validation score curve saved as validation_score_curve.png")
plt.close()

# Final test phase
model.eval()  # Set to evaluation mode for testing
test_loss = 0.0
all_test_targets, all_test_predictions = [], []

with torch.no_grad():
    for input_image, target_image in tqdm(test_loader, desc="Testing"):
        input_image, target_image = input_image.to(device), target_image.to(device)

        output_image = model(input_image)
        loss = criterion(output_image, target_image)
        test_loss += loss.item()

        all_test_targets.extend(target_image.cpu().numpy().flatten())
        all_test_predictions.extend(output_image.cpu().numpy().flatten())

avg_test_loss = test_loss / len(test_loader)
test_mae = mean_absolute_error(all_test_targets, all_test_predictions)
print(f"Test Loss: {avg_test_loss:.4f}, Test MAE: {test_mae:.4f}")
torch.save(model.state_dict(), "RDM2_coeff_Unet.pth")