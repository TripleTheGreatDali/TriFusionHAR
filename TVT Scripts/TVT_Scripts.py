import os
import shutil
import random
from tqdm import tqdm

# Define paths
input_folder = './UCF101'  
output_folder = './UCF101_Frames' 

# Define train, test, and val ratios
train_ratio = 0.7
test_ratio = 0.1
val_ratio = 0.2

# Function to create train, test, val folders
def create_folders(output_folder):
    for folder in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

# Function to split dataset
def split_dataset(input_folder, output_folder):
    # Create folders
    create_folders(output_folder)
    
    # Iterate through each class folder
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        videos = os.listdir(class_path)
        num_videos = len(videos)
        random.shuffle(videos)
        
        # Calculate split indices
        train_end = int(train_ratio * num_videos)
        test_end = train_end + int(test_ratio * num_videos)
        
        # Move videos to respective folders
        for i, video in enumerate(tqdm(videos, desc=f'Processing {class_folder}', unit='videos')):
            src = os.path.join(class_path, video)
            if i < train_end:
                dest = os.path.join(output_folder, 'train', class_folder, video)
            elif i < test_end:
                dest = os.path.join(output_folder, 'test', class_folder, video)
            else:
                dest = os.path.join(output_folder, 'val', class_folder, video)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)

# Perform splitting
split_dataset(input_folder, output_folder)
