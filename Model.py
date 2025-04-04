import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MFCCDataset(Dataset): # Loads preprocessed MFCC feature arrays and labels from .npy files.
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        
        self.features = torch.tensor(self.features).float() # Converts them into PyTorch tensors.
        self.labels = torch.tensor(self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].unsqueeze(0)  
        y = self.labels[idx]
        return x, y

class AASIST(nn.Module):
    def __init__(self, num_classes=2):
        super(AASIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Extract local audio patterns using CNN.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) # Downsample spatial dimensions
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True) # Multi-head attention to focus on important temporal regions.

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # Condense the features into one vector per sample.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 128)  # Fully connected layers for classification
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Attention block: reshape [B, C, H, W] -> [B, L, E]
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)  # [B, L, E]
        x, _ = self.attn(x, x, x)

        x = self.global_pool(x.permute(0, 2, 1).unsqueeze(-1))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EarlyStopping: # Stops training if validation loss doesn't improve after patience epochs.
    def __init__(self, patience=5, delta=0.001, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # Save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Paths to .npy files
train_dataset = MFCCDataset("Training_features.npy", "Training_labels.npy")
test_dataset = MFCCDataset("Testing_features.npy", "Testing_labels.npy")
val_dataset = MFCCDataset("Validation_features.npy", "Validation_labels.npy")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AASIST(num_classes=2).to(device)
for param in model.conv1.parameters():
    param.requires_grad = False

for param in model.conv2.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

early_stopping = EarlyStopping(patience=5)

# Training loop
epochs = 10 
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Training Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    acc = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {acc:.2f}%")

    
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Load the best model after training
model.load_state_dict(torch.load("Model.pt"))

# Evaluation
model.eval()

# Store predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        outputs = model(x_batch)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Calculate metrics
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

# Print results
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()