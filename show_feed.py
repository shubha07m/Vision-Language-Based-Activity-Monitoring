import cv2
import os
import signal
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from generate_caption import generate_caption
from PIL import Image
import csv
from sentence_transformers import SentenceTransformer
import psutil
import numpy as np

# Local Stream URL
stream_url = "http://192.168.0.20:5000/video_feed"

# Directory to save captured images
save_path = "/Users/nemo/Desktop/vlm_for_IoT/images"
os.makedirs(save_path, exist_ok=True)

# Path for CSV file
csv_file_path = os.path.join(save_path, "captions.csv")

# Load ResNet model for feature extraction
resnet = models.resnet50(pretrained=True)
resnet.eval()
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer

# Image transformation for ResNet input
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Sentence-BERT model for semantic similarity
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def get_image_feature_vector(image):
    """Extract feature vector from an image."""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet(image)
    return features.flatten().numpy()


def get_caption_embedding(caption):
    """Get the sentence embedding for a caption."""
    return sbert_model.encode(caption)


def get_last_saved_info():
    """Retrieve the last saved image number, caption, and image features from CSV."""
    last_image_num = 0
    last_caption = None
    last_features = None

    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            rows = list(reader)
            if rows:
                last_image_num = int(rows[-1][0])
                last_caption = rows[-1][2]
                last_image_path = os.path.join(save_path, f"{last_image_num}.jpg")
                if os.path.exists(last_image_path):
                    last_image = Image.open(last_image_path)
                    last_features = get_image_feature_vector(last_image)

    return last_image_num, last_caption, last_features


def terminate_orphan_processes():
    """Terminate orphan Python processes that might hang."""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'python' and proc.pid != os.getpid():
            proc.terminate()
            proc.wait()  # Ensure the process is fully terminated


def restart_process():
    """Terminate scripts, ensure clean shutdown, and restart."""
    terminate_orphan_processes()
    time.sleep(3)  # Short delay to ensure clean shutdown
    os.system("pkill -f run_on_pi.sh && pkill -f run_local.sh")  # Terminate scripts
    os.system("bash /Users/nemo/run_on_pi.sh && bash /Users/nemo/run_local.sh")  # Restart scripts
    exit()


# Initialize CSV file if it doesn't exist
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence No", "Timestamp", "Caption"])

# Retrieve the last saved image number, caption, and features
last_image_num, last_caption, last_features = get_last_saved_info()

# Open the video stream
cap = cv2.VideoCapture(stream_url)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Couldn't read video stream from file. Restarting the process...")
    restart_process()

# Capture the first frame and generate its caption
ret, initial_frame = cap.read()
retry_count = 0  # Initialize retry counter
restart_attempts = 0  # Initialize restart attempts counter

while not ret:
    retry_count += 1
    if retry_count >= 3:
        restart_attempts += 1
        print(f"Failed to capture the initial frame after {retry_count} retries. Restarting the process (Attempt {restart_attempts}/3)...")
        restart_process()

    if restart_attempts >= 3:
        print("Maximum restart attempts reached. Exiting the process.")
        exit()

    print("Retrying to capture the initial frame...")
    ret, initial_frame = cap.read()

# Convert initial frame to PIL format and extract its feature vector
initial_image = Image.fromarray(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
initial_features = get_image_feature_vector(initial_image)

if last_image_num == 0:
    # First-time execution: generate and save the first image and caption directly
    initial_filename = os.path.join(save_path, f"{last_image_num + 1}.jpg")
    cv2.imwrite(initial_filename, initial_frame)
    initial_caption = generate_caption(initial_filename)
    initial_caption_embedding = get_caption_embedding(initial_caption)

    print(f"Initial Caption: {initial_caption}")

    # Write the initial caption to CSV
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([last_image_num + 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), initial_caption])

    # Initialize for future comparisons
    previous_features = initial_features
    previous_caption = initial_caption
    previous_caption_embedding = initial_caption_embedding

else:
    # After restart: use last saved image and caption for comparison
    previous_features = last_features
    previous_caption = last_caption
    previous_caption_embedding = get_caption_embedding(last_caption)

# Start the timer
last_caption_time = time.time()

print("Video feed is active.")

# Similarity thresholds (adjust these values)
image_similarity_threshold = 0.9
caption_similarity_threshold = 0.9
image_counter = last_image_num + 2  # Increment counter for next image

while True:
    # Check if 5 minutes have passed since the last caption was generated
    current_time = time.time()
    if current_time - last_caption_time >= 300:  # 300 seconds = 5 minutes
        print("No new caption for 5 minutes. Restarting the process...")
        restart_process()

    # Capture a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        retry_count += 1
        print(f"Failed to capture a frame. Retry {retry_count}/3...")
        if retry_count >= 3:
            print("Failed to capture the frame after 3 retries. Restarting the process...")
            restart_process()
        continue  # Skip the rest of the loop and retry
    retry_count = 0  # Reset retry counter after successful frame capture

    # Add a small delay to control the frame sampling rate
    time.sleep(0.1)  # 100ms delay between each frame capture

    # Convert current frame to PIL format and extract its feature vector
    current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_features = get_image_feature_vector(current_image)

    # Calculate cosine similarity between current and previous frames
    similarity = cosine_similarity([previous_features], [current_features])[0][0]

    # If the frames are sufficiently different, generate a new caption
    if similarity < image_similarity_threshold:
        # Generate a new caption
        temp_filename = os.path.join(save_path, "temp.jpg")
        cv2.imwrite(temp_filename, frame)
        current_caption = generate_caption(temp_filename)
        current_caption_embedding = get_caption_embedding(current_caption)

        # Calculate the semantic similarity between the current and previous captions
        caption_similarity = cosine_similarity([previous_caption_embedding], [current_caption_embedding])[0][0]

        # Only save if both the image and caption are sufficiently different from the last saved ones
        if caption_similarity < caption_similarity_threshold:
            # Save the frame with a sequential filename
            filename = os.path.join(save_path, f"{image_counter}.jpg")
            os.rename(temp_filename, filename)

            # Print the new caption
            print(f"New Caption: {current_caption}")

            # Write the new caption to CSV
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_counter, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_caption])

            # Update the previous features, caption, and embedding
            previous_features = current_features
            previous_caption = current_caption
            previous_caption_embedding = current_caption_embedding

            # Update the last caption time
            last_caption_time = time.time()

            # Increment image counter
            image_counter += 1
        else:
            os.remove(temp_filename)

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
