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
from PIL import Image
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for the progress bar

# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
image_dir = os.path.join(script_dir, 'data', 'PKLot', 'test')
test_data_path = r"data\PKLot\test"
json_file_path = os.path.join(test_data_path, "_annotations.coco.json")
model_path = r"model.h5"


def crop_roi(img, bbox):
    x, y, width, height = bbox
    right = x + width
    bottom = y + height
    roi = img[y:bottom, x:right]
    return roi

def preprocess_for_prediction(roi, target_size=(30, 30)):    
    # Resize the ROI to the target size expected by the model
    roi_resized = cv2.resize(roi, target_size)
    
    # Normalize pixel values if your model expects normalization
    roi_normalized = roi_resized / 255.0
    
    # Add another new axis for the channel (grayscale channel of 1)
    roi_expanded = np.expand_dims(roi_normalized, axis=-1)   # Shape: (1, 30, 30, 1)
    
    return roi_expanded

def extract_bboxes_from_json(data):
    bboxes_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        bboxes_by_image[annotation["image_id"]].append(annotation['bbox'])
    return bboxes_by_image


def draw_bounding_boxes_and_predict(image, bboxes, model):
    # Convert frame to grayscale and apply the mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize lists to store preprocessed ROIs and bounding box coordinates
    rois = []
    # Crop for each annotation
    for bbox in bboxes:
        roi = crop_roi(gray, bbox)
        roi_preprocessed = preprocess_for_prediction(roi)
        rois.append(roi_preprocessed)
    # Convert list of ROIs to a numpy array for batch prediction
    rois_array = np.array(rois)
    print()
    print(rois_array.shape)

    # Ensure `rois_array` is not empty before predicting
    if rois_array.size == 0:
        print("Warning: No ROIs found for prediction.")
        return image  # Return the original image if no ROIs are found
    
    # Perform batch prediction
    predictions = model.predict(rois_array)
    predicted_classes = np.argmax(predictions, axis=1)  # Get the predicted class for each ROI

    # Loop through bounding boxes and predicted classes to draw them on the frame
    for (x, y, w, h), predicted_class in zip(bboxes, predicted_classes):
        # Draw bounding box: green if occupied, red if empty
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    return image


def main():
    # Load bboxes
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    bboxes_by_image = extract_bboxes_from_json(data)
    # Load the model
    model = load_model(model_path)

    with tqdm(total=len(data["images"]), desc="Processing images") as pbar:
        for image_info in data["images"]:
            image_id = image_info["id"]
            image_path = os.path.join(image_dir, image_info["file_name"])
            bboxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in bboxes_by_image[image_id]]
    
            try:
                # Open the image with OpenCV
                image = cv2.imread(image_path)
                
                # Check if the image was loaded successfully
                if image is None:
                    raise FileNotFoundError
                
                # Retrieve annotations for this image_id
                frame_with_predictions = draw_bounding_boxes_and_predict(image, bboxes, model)
                
                # Display the output with predictions
                cv2.imshow('Predicted output', frame_with_predictions)
                
                # Wait for 1 ms and check if 'q' is pressed to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Update progress bar after processing each image
                pbar.update(1)
            
            except FileNotFoundError:
                print(f"Image file not found: {image_path}")
                pbar.update(1)  # Still update the progress bar if an image is missing

    # Close all OpenCV windows once done
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()