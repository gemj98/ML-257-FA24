# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:38:45 2024

@author: gemj9
"""
import numpy as np
import cv2

mask_image_path = r"data\mask_1920_1080.png"
video_path = r"data\parking_1920_1080.mp4"
model_path = r"model.h5"

def preprocess_for_prediction(roi, target_size=(30, 30)):
    # Resize the ROI to the target size expected by the model
    roi_resized = cv2.resize(roi, target_size)
    # Normalize pixel values if your model expects normalization
    roi_normalized = roi_resized / 255.0
    # Expand dimensions to add the batch size
    # roi_expanded = np.expand_dims(roi_normalized, axis=0)
    return roi_normalized

def draw_bounding_boxes_and_predict(frame, mask, model):
    # Convert frame to grayscale and apply the mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store preprocessed ROIs and bounding box coordinates
    rois = []
    bounding_boxes = []

    # Loop through contours to extract bounding boxes and preprocess ROIs
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        
        # Preprocess ROI for the model and store in lists
        roi_preprocessed = preprocess_for_prediction(roi)
        rois.append(roi_preprocessed)
        bounding_boxes.append((x, y, w, h))

    # Convert list of ROIs to a numpy array for batch prediction
    rois_array = np.array(rois)

    # Perform batch prediction
    predictions = model.predict(rois_array)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the predicted class for each ROI

    # Loop through bounding boxes and predicted classes to draw them on the frame
    for (x, y, w, h), predicted_class in zip(bounding_boxes, predicted_classes):
        # Draw bounding box: green if occupied, red if empty
        color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame


# Load the mask image (the same mask used earlier)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# Capture video from a file or camera (replace 'your_video.mp4' with your video file or use 0 for webcam)
video_capture = cv2.VideoCapture(video_path)

from tensorflow.keras.models import load_model
# Load the model
model = load_model(model_path)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab a frame")
        break

    # Use the updated function that includes model predictions
    frame_with_predictions = draw_bounding_boxes_and_predict(frame, mask, model)

    cv2.imshow('Video with Parking Slot Occupancy', frame_with_predictions)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
video_capture.release()
cv2.destroyAllWindows()