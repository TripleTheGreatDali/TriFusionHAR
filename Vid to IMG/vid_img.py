import os
import cv2
import imghdr
from tqdm import tqdm

def extract_frames(video_path, output_folder, num_frames=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract frames
    extracted_frames = 0
    pbar = tqdm(total=num_frames, desc="Extracting Frames", unit="frame")
    while extracted_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{extracted_frames:06d}.jpg"  # Unique numbering for frames
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        extracted_frames += 1
        pbar.update(1)

    # Release the video capture object and close the progress bar
    cap.release()
    pbar.close()

# Paths to your dataset
dataset_folder = "./UCF101"
output_folder = "./UCF101_Frames"

# Specify the number of frames to extract from each video
num_frames_per_video = 10

# Loop through each video in the dataset folder
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        video_path = os.path.join(root, file)
        # Check if the file is a video file
        if imghdr.what(video_path) is None:
            class_name = os.path.basename(os.path.dirname(video_path))  # Extract class name
            class_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)
            extract_frames(video_path, class_folder, num_frames=num_frames_per_video)

print("Frame extraction complete!")
