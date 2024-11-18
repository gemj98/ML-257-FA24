    # -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:38:45 2024

@author: gemj9
"""
import os
import cv2
from tensorflow.keras.models import load_model  
import json

from utils_module import get_annotations_from_json, get_bboxes_and_labels_by_image, draw_predicted_occupancy, draw_labeled_bounding_boxes

# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
test_data_path = os.path.join(script_dir, '..', 'data', 'PKLot', 'test')       # Path to test data
json_file_path = os.path.join(test_data_path, "_annotations.coco.json")  # Path to the annotations JSON file
model_path = os.path.join(script_dir, 'model_normalized_30_30.keras')                     # Path to the trained model

# Main function to load data, predict, and visualize results
def main():
    # Load bounding box annotations from the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    annotations_by_image = get_annotations_from_json(data)  # Extract bounding boxes from annotations
    
    # Load the trained model
    model = load_model(model_path)

    print("Press 'E' to proceed to the next image or 'Q' to quit.")

    # Iterate over images in the JSON file
    for image_info in data["images"]:
        image_id = image_info["id"]  # Get the image ID
        image_path = os.path.join(test_data_path, image_info["file_name"])  # Get the path to the image file

        try:
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            
            # Check if the image was successfully loaded
            if image is None:
                raise FileNotFoundError
            
            bboxes, labels = get_bboxes_and_labels_by_image(annotations_by_image, image_id)
            
            # Process the image: draw bounding boxes and predict classes
            image_with_predictions = draw_predicted_occupancy(image.copy(), bboxes, model)
            image_with_correct_labels = draw_labeled_bounding_boxes(image.copy(), bboxes, labels)

            
            # Display the output with bounding boxes and predictions
            cv2.imshow('Predicted output', image_with_predictions)
            cv2.imshow('Real labels', image_with_correct_labels)
            
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
