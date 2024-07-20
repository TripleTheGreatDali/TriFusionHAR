import os
import torch
import time
from torchvision import transforms, datasets
from model import VGG16Features,TransferModel, BiGRU, TriFusion
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Set device - GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the directory paths
save_dir = './saved_models'
results_dir = './evaluation_results'
test_dir = './test'

# Load the saved model
model_filename = f'hybrid_model_epoch_100.pth'
model_path = os.path.join(save_dir, model_filename)

checkpoint = torch.load(model_path)
vgg_features = VGG16Features().to(device)
bigru = BiGRU(input_size=512, hidden_size=256, num_classes=101).to(device)
transfer_model = TransferModel(num_classes=101).to(device)

hybrid_model = TriFusion(vgg_features, bigru, transfer_model).to(device)
hybrid_model.load_state_dict(checkpoint['model_state_dict'])
hybrid_model.eval()

# Set up data transformations for testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Read labels from labels.txt
labels_path = './labels.txt'
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize variables for tracking correct and wrong predictions and timing
correct_predictions = 0
wrong_predictions = 0
all_preds = []
all_labels = []
batch_times = []
frame_times = []
total_frames = 0

# Test the model
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        start_time = time.time()  # Start timing
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = hybrid_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct_predictions += torch.sum(preds == labels).item()
        wrong_predictions += torch.sum(preds != labels).item()
        end_time = time.time()  # End timing
        
        batch_time = (end_time - start_time) * 1000  # Convert to milliseconds
        batch_times.append(batch_time)
        total_frames += inputs.size(0)  # Count the number of frames

        # Record time per frame
        frame_times.extend([batch_time / inputs.size(0)] * inputs.size(0))

# Prepare the results as a DataFrame
results_df = pd.DataFrame({'Original Labels': all_labels, 'Predicted Labels': all_preds})

# Save the results as a CSV file
results_csv_path = os.path.join(results_dir, 'test_results.csv')
results_df.to_csv(results_csv_path, index=False)

# Calculate overall accuracy
total_predictions = correct_predictions + wrong_predictions
accuracy = correct_predictions / total_predictions

# Display overall accuracy and save it to a text file
accuracy_text_path = os.path.join(results_dir, 'test_accuracy.txt')
with open(accuracy_text_path, 'w') as accuracy_file:
    accuracy_file.write(f'Overall Accuracy: {accuracy:.4f}')

# Calculate and display the average inference speed per frame in milliseconds
average_frame_time = np.mean(frame_times)
std_frame_time = np.std(frame_times)
print(f'Average inference speed per frame: {average_frame_time:.4f} ± {std_frame_time:.4f} ms')

# Save the average inference speed per frame to a text file
inference_speed_text_path = os.path.join(results_dir, 'average_inference_speed_per_frame.txt')
with open(inference_speed_text_path, 'w') as speed_file:
    speed_file.write(f'Average inference speed per frame: {average_frame_time:.4f} ± {std_frame_time:.4f} ms')
