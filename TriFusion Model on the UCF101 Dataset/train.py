import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import VGG16Features, TriFusion, TransferModel, BiGRU
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set device - GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Define the directory paths
save_dir = './saved_models'
results_dir = './evaluation_results'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Set up data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 64
num_classes = 101

# Set up data loaders for train, val, and test with data augmentation
train_dataset = datasets.ImageFolder(root='./train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.ImageFolder(root='./val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Move model to GPU
vgg_features = VGG16Features().to(device)
bigru = BiGRU(input_size=512, hidden_size=256, num_classes=num_classes).to(device)
transfer_model = TransferModel(num_classes=num_classes).to(device)

hybrid_model = TriFusion(vgg_features, bigru, transfer_model).to(device)


# Read labels from labels.txt
labels_path = './labels.txt'
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]  # Convert to Python list

# Define labels as a Python list
labels_list = list(labels)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training parameters
num_epochs = 100
train_losses = []
val_losses = []
precision_list = []
recall_list = []
f1_list = []
accuracy_list = []

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    hybrid_model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = hybrid_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    hybrid_model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = hybrid_model(inputs)
            val_loss += criterion(outputs, labels).item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=1)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    scheduler.step(average_val_loss)

    # Calculate accuracy on validation set
    accuracy = accuracy_score(all_labels, all_preds)
    accuracy_list.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {average_val_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Compute confusion matrix
    confusion = confusion_matrix(all_labels, all_preds)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
    plt.close()

    # Confusion Matrix
    confusion_df = pd.DataFrame(confusion, index=labels_list, columns=labels_list)
    confusion_df.to_csv(os.path.join(results_dir, f'confusion_matrix_epoch_{epoch+1}.csv'), index=False)

end_time = time.time()

total_training_time = end_time - start_time

# Save all metrics to a CSV file
metrics_dict = {
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1 Score': f1_list,
    'Accuracy': accuracy_list
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

# Save the entire model state
torch.save({
    'model_state_dict': hybrid_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'precision_list': precision_list,
    'recall_list': recall_list,
    'f1_list': f1_list,
    'accuracy_list': accuracy_list,
    'labels_list': labels_list
}, os.path.join(save_dir, 'TriFusion_ucf101_100_epoch.pth'))


# Save training time
time_file_path = os.path.join(results_dir, 'training_time.txt')
with open(time_file_path, 'w') as time_file:
    time_file.write(f'Total Training Time: {total_training_time} seconds')

# Plot training and validation metrics
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
plt.show()

plt.plot(range(1, num_epochs+1), precision_list, label='Precision')
plt.plot(range(1, num_epochs+1), recall_list, label='Recall')
plt.plot(range(1, num_epochs+1), f1_list, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Precision, Recall, and F1 Score')
plt.legend()
plt.savefig(os.path.join(results_dir, 'metrics_plot.png'))
plt.show()

plt.plot(range(1, num_epochs+1), accuracy_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
plt.show()
