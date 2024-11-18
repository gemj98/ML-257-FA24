# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:38:45 2024

@author: gemj9
"""
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from utils_module import get_annotations_from_json, get_bboxes_and_labels_by_image, draw_predicted_occupancy, draw_labeled_bounding_boxes

# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
mask_image_path = os.path.join(script_dir, '..', 'data', 'mask.png')     # Path to the mask image
video_path = os.path.join(script_dir, '..', 'data', 'parking_video.mp4') # Path to the video file
model_path = os.path.join(script_dir,'model_normalized_30_30.keras')                # Path to the trained model

# Function to draw bounding boxes on the frame and predict occupancy status
def get_bboxes(frame, mask):
    # Convert the frame to grayscale and apply the mask to isolate the region of interest
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_frame', gray)
    # # Wait for user input to proceed or quit
    # while True:
    #     key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
    #     if key == ord('e'):  # If 'E' is pressed, proceed to the next image
    #         break
            
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Find contours in the masked image to detect objects
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store preprocessed ROIs and bounding box coordinates
    bounding_boxes = []

    # Loop through each contour to extract bounding boxes and preprocess ROIs
    for contour in contours:
        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the region of interest (ROI) from the frame
        bounding_boxes.append((x, y, w, h))  # Store the bounding box coordinates

    return bounding_boxes

# Load the mask image (used to define areas of interest in the frame)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# Capture video from a file
video_capture = cv2.VideoCapture(video_path)

# Load the trained model
model = load_model(model_path)


ret, frame = video_capture.read()
bboxes = get_bboxes(frame, mask)

# Process the video frame by frame
while True:
    # Read the next frame from the video
    ret, frame = video_capture.read()
    # Break the loop if no frame is retrieved (end of video or error)
    if not ret:
        print("Failed to grab a frame")
        break

    # Use the updated function to process the frame and make predictions
    frame_with_predictions = draw_predicted_occupancy(frame, bboxes, model)
    frame_with_predictions = cv2.resize(frame_with_predictions, (1200, 800))  # Adjust dimensions as needed

    # Add text overlay with total and empty spots
    text = f"Real Empty Spots: ~124"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    thickness = 2
    position = (700, 50)  # Top-left corner

    # Add the text to the image
    cv2.putText(frame_with_predictions, text, position, font, font_scale, font_color, thickness)

    # Display the processed frame with predictions
    cv2.imshow('Video with Parking Slot Occupancy', frame_with_predictions)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
