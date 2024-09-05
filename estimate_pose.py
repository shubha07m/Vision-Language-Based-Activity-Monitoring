import cv2
from ultralytics import YOLO

# Load your model
model = YOLO("yolov8n-pose.pt")

# Open the video
# cap = cv2.VideoCapture("golf_video.mp4")
cap = cv2.VideoCapture("http://192.168.0.20:5000/video_feed")

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run model prediction
    results = model(frame)

    # Draw keypoints and annotations on the frame
    annotated_frame = results[0].plot()  # This function will plot the keypoints on the frame

    # Show the frame with keypoints
    cv2.imshow('Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()