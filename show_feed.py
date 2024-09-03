import cv2
import os
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
    transforms.Resize((224, 224)),
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


# Initialize CSV file
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence No", "Timestamp", "Caption"])

# Open the video stream
cap = cv2.VideoCapture(stream_url)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Failed to open the video stream.")
    exit()

# Capture the first frame and generate its caption
ret, initial_frame = cap.read()
if not ret:
    print("Failed to capture the initial frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Convert initial frame to PIL format and extract its feature vector
initial_image = Image.fromarray(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
previous_features = get_image_feature_vector(initial_image)

# Save the initial frame and generate its caption
initial_filename = os.path.join(save_path, "0.jpg")
cv2.imwrite(initial_filename, initial_frame)
previous_caption = generate_caption(initial_filename)
previous_caption_embedding = get_caption_embedding(previous_caption)
print(f"Initial Caption: {previous_caption}")

# Write the initial caption to CSV
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), previous_caption])

print("Video feed is active. Press 'q' to quit.")

# Similarity thresholds (adjust these values)
image_similarity_threshold = 0.9
caption_similarity_threshold = 0.8
image_counter = 1

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture a frame.")
        break

    # Display the video feed
    cv2.imshow('Video Feed', frame)

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

        # Only save if the captions are semantically different
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

            # Increment image counter
            image_counter += 1
        else:
            os.remove(temp_filename)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' to quit
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
