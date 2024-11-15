# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:38:45 2024

@author: gemj9
"""
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
import json
from collections import defaultdict

# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
test_data_path = os.path.join(script_dir, '..', 'data', 'PKLot', 'test')       # Path to test data
json_file_path = os.path.join(test_data_path, "_annotations.coco.json")  # Path to the annotations JSON file
model_path = os.path.join(script_dir, 'model.keras')                     # Path to the trained model

# Function to crop a region of interest (ROI) from an image based on a bounding box
def crop_roi(img, bbox):
    x, y, width, height = bbox  # Extract coordinates and dimensions from the bounding box
    right = x + width
    bottom = y + height
    roi = img[y:bottom, x:right]  # Crop the region from the image
    return roi  # Return the cropped region (ROI)

# Function to preprocess an ROI for prediction by the model
def preprocess_for_prediction(roi, target_size=(30, 30)):
    # Resize the ROI to match the input size expected by the model
    roi_resized = cv2.resize(roi, target_size)
    
    # Normalize pixel values to the range [0, 1] if required by the model
    roi_normalized = roi_resized / 255.0
    
    # Add a new axis for the channel (e.g., grayscale channel with depth 1)
    roi_expanded = np.expand_dims(roi_normalized, axis=-1)  # Shape: (30, 30, 1)
    
    return roi_expanded  # Return the preprocessed ROI

# Function to extract bounding boxes from the annotations JSON data
def extract_bboxes_from_json(data):
    bboxes_by_image = defaultdict(list)  # Create a dictionary to store bounding boxes by image ID
    for annotation in data["annotations"]:
        bboxes_by_image[annotation["image_id"]].append(annotation['bbox'])  # Group bounding boxes by image ID
    return bboxes_by_image  # Return the dictionary of bounding boxes

# Function to draw bounding boxes and predict classes for each ROI in an image
def draw_bounding_boxes_and_predict(image, bboxes, model):
    # Convert the input image to grayscale (required for single-channel input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize a list to store preprocessed ROIs
    rois = []
    
    # Process each bounding box
    for bbox in bboxes:
        roi = crop_roi(gray, bbox)  # Crop the ROI from the image
        roi_preprocessed = preprocess_for_prediction(roi)  # Preprocess the ROI for prediction
        rois.append(roi_preprocessed)  # Add the preprocessed ROI to the list
    
    # Convert the list of ROIs to a numpy array for batch prediction
    rois_array = np.array(rois)

    # Ensure the array is not empty before making predictions
    if rois_array.size == 0:
        print("Warning: No ROIs found for prediction.")
        return image  # Return the original image if no ROIs are found
    
    # Perform batch prediction on the ROIs
    predictions = model.predict(rois_array, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)  # Determine the predicted class for each ROI

    # Draw bounding boxes and labels on the image
    for (x, y, w, h), predicted_class in zip(bboxes, predicted_classes):
        # Set bounding box color: green for occupied, red for empty
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)  # Draw the bounding box

    return image  # Return the image with bounding boxes and predictions

# Main function to load data, predict, and visualize results
def main():
    # Load bounding box annotations from the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    bboxes_by_image = extract_bboxes_from_json(data)  # Extract bounding boxes from annotations
    
    # Load the trained model
    model = load_model(model_path)

    print("Press 'E' to proceed to the next image or 'Q' to quit.")

    # Iterate over images in the JSON file
    for image_info in data["images"]:
        image_id = image_info["id"]  # Get the image ID
        image_path = os.path.join(test_data_path, image_info["file_name"])  # Get the path to the image file
        
        # Convert bounding box coordinates to integers
        bboxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in bboxes_by_image[image_id]]

        try:
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            
            # Check if the image was successfully loaded
            if image is None:
                raise FileNotFoundError
            
            # Process the image: draw bounding boxes and predict classes
            frame_with_predictions = draw_bounding_boxes_and_predict(image, bboxes, model)
            
            # Display the output with bounding boxes and predictions
            cv2.imshow('Predicted output', frame_with_predictions)
            
            # Wait for user input to proceed or quit
            while True:
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
                if key == ord('e'):  # If 'E' is pressed, proceed to the next image
                    break
                elif key == ord('q'):  # If 'Q' is pressed, exit the loop
                    cv2.destroyAllWindows()
                    exit()

        except FileNotFoundError:
            print(f"Image file not found: {image_path}")

    # Close all OpenCV windows when done
    cv2.destroyAllWindows()

# Entry point for the script
if __name__ == "__main__":
    main()
