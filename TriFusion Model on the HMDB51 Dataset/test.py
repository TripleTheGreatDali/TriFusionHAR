import os
import torch
from torchvision import transforms, datasets
from model import VGG16Features, TriFusion, TransferModel, BiGRU
from tqdm import tqdm
import pandas as pd

# Set device - GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the directory paths
save_dir = './saved_models'
results_dir = './evaluation_results'
test_dir = './Dataset Fv01/test'

# Load the saved model
model_filename = f'hybrid_model.pth'
model_path = os.path.join(save_dir, model_filename)

checkpoint = torch.load(model_path)
vgg_features = VGG16Features().to(device)
bigru = BiGRU(input_size=512, hidden_size=256, num_classes=51).to(device)
transfer_model = TransferModel(num_classes=51).to(device)

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

# Initialize variables for tracking correct and wrong predictions
correct_predictions = 0
wrong_predictions = 0
all_preds = []
all_labels = []

# Test the model
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = hybrid_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct_predictions += torch.sum(preds == labels).item()
        wrong_predictions += torch.sum(preds != labels).item()

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
