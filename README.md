# VisionLanguageBasedActivityMonitoring

## Description
A Raspberry Pi-powered vision-language project that monitors activity by identifying significant scene changes and generating captions to create a detailed report.

## Goal
The goal of this project is to build a visual language model that monitors a video feed, captures frames with significant scene changes, generates captions, and stores them for report generation. The project is focused on identifying meaningful events, such as unauthorized activities, through scene understanding and caption generation.

## Implementation
1. **Video Capture**: The video was captured using an RGB camera attached to a Raspberry Pi.
2. **Streaming**: The video stream is forwarded from the Raspberry Pi using Flask.
3. **Scene Change Detection**: Frames from the video are processed using ResNet-50 to extract feature vectors. Cosine similarity is used to compare consecutive frames and identify significant scene changes.
4. **Caption Generation**: BLIP (Bootstrapped Language Image Pretraining) is used to generate captions for frames with significant changes.
5. **Data Storage**: Images and captions are saved, along with timestamps, in a CSV file. Only images and captions that show significant scene changes are stored.
6. **Report Generation** (Future Work): Develop a report generation system that provides detailed activity monitoring reports based on the stored images, timestamps, and captions.


![Sample special event detection](https://github.com/shubha07m/VisionLanguageBasedActivityMonitoring/blob/main/sample_event.png)

## Future Work
- Develop a report generation feature that uses the stored data to create comprehensive activity monitoring reports for specific durations.
- Explore integrating real-time notifications for detected events.
- Improve the captioning model to handle more complex scene descriptions.

## Usage
1. Set up the Raspberry Pi with the RGB camera and Flask for video streaming.
2. Run the `show_feed.py` script to start monitoring the video feed.
3. Captions and images will be stored in the specified directory along with a CSV file containing timestamps and captions.
4. Generate reports from the stored data (future work).

## Requirements
- Python 3.11
- PyTorch
- Transformers
- OpenCV
- Flask
- PIL (Pillow)
- scikit-learn
- datetime
- pandas
- fpdf
