# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:38:45 2024

@author: gemj9
"""
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
mask_image_path = os.path.join(script_dir, '..', 'data', 'mask.png')     # Path to the mask image
video_path = os.path.join(script_dir, '..', 'data', 'parking_video.mp4') # Path to the video file
model_path = os.path.join(script_dir,'model.keras')                # Path to the trained model

# Function to preprocess a region of interest (ROI) for model prediction
def preprocess_for_prediction(roi, target_size=(30, 30)):
    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Resize the ROI to the target size expected by the model
    roi_resized = cv2.resize(roi_gray, target_size)
    # Normalize pixel values to the range [0, 1]
    roi_normalized = roi_resized / 255.0
    # Return the preprocessed ROI
    return roi_normalized

# Function to draw bounding boxes on the frame and predict occupancy status
def draw_bounding_boxes_and_predict(frame, mask, model):
    # Convert the frame to grayscale and apply the mask to isolate the region of interest
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Find contours in the masked image to detect objects
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store preprocessed ROIs and bounding box coordinates
    rois = []
    bounding_boxes = []

    # Loop through each contour to extract bounding boxes and preprocess ROIs
    for contour in contours:
        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the region of interest (ROI) from the frame
        roi = frame[y:y+h, x:x+w]
        # Preprocess the ROI for the model
        roi_preprocessed = preprocess_for_prediction(roi)
        rois.append(roi_preprocessed)  # Append the preprocessed ROI to the list
        bounding_boxes.append((x, y, w, h))  # Store the bounding box coordinates

    # Convert the list of ROIs to a numpy array for batch prediction
    rois_array = np.array(rois)

    # Perform batch prediction using the trained model
    predictions = model.predict(rois_array)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the predicted class for each ROI

    # Debugging output: print predictions for the first ROI
    print(predictions[0])
    print(predicted_classes[0])

    # Draw bounding boxes on the frame based on predictions
    for (x, y, w, h), predicted_class in zip(bounding_boxes, predicted_classes):
        # Set the bounding box color: green for empty, red for occupied
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  # Draw the bounding box

    # Return the frame with drawn bounding boxes and predictions
    return frame

# Load the mask image (used to define areas of interest in the frame)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# Capture video from a file
video_capture = cv2.VideoCapture(video_path)

# Load the trained model
model = load_model(model_path)

# Process the video frame by frame
while True:
    # Read the next frame from the video
    ret, frame = video_capture.read()
    # Break the loop if no frame is retrieved (end of video or error)
    if not ret:
        print("Failed to grab a frame")
        break

    # Use the updated function to process the frame and make predictions
    frame_with_predictions = draw_bounding_boxes_and_predict(frame, mask, model)

    # Display the processed frame with predictions
    cv2.imshow('Video with Parking Slot Occupancy', frame_with_predictions)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
