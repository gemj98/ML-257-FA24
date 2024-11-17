from collections import defaultdict
import numpy as np
import cv2

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
    # roi_normalized = roi_resized / 255.0
    
    # Add a new axis for the channel (e.g., grayscale channel with depth 1)
    # roi_expanded = np.expand_dims(roi_normalized, axis=-1)  # Shape: (30, 30, 1)
    roi_expanded = np.expand_dims(roi_resized, axis=-1)  # Shape: (30, 30, 1)
    
    return roi_expanded  # Return the preprocessed ROI

# Function to extract bounding boxes from the annotations JSON data
def get_annotations_from_json(data):
    annotations_by_image = defaultdict(list)  # Create a dictionary to store bounding boxes by image ID
    for annotation in data["annotations"]:
        bbox = [int(value) for value in annotation['bbox']]
        # [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in annotation['bbox']]
        label = 0 if annotation['category_id'] == 1 else 1
        annotations_by_image[annotation["image_id"]].append(
            {'bbox': bbox,
             'label': label}
        )  # Group bounding boxes by image ID
    return annotations_by_image  # Return the dictionary of bounding boxes

def get_bboxes_and_labels_by_image(annotations_by_image, image_id):
    entries = annotations_by_image.get(image_id, [])

    # Separate bboxes and labels
    bboxes = [entry['bbox'] for entry in entries]
    labels = [entry['label'] for entry in entries]
    
    return bboxes, labels

def draw_labeled_bounding_boxes(image, bboxes, labels):
    # Draw bounding boxes and labels on the image
    for (x, y, w, h), predicted_class in zip(bboxes, labels):
        # Set bounding box color: green for empty, red for occupied (B, G, R)
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)  # Draw the bounding box
    return image

# Function to draw bounding boxes and predict classes for each ROI in an image
def draw_predicted_occupancy(image, bboxes, model):
    # Convert the input image to grayscale (required for single-channel input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize a list to store preprocessed ROIs
    rois = []
    
    # Process each bounding box
    for bbox in bboxes:
        roi = crop_roi(gray, bbox)  # Crop the ROI from the image
        roi_preprocessed = preprocess_for_prediction(roi)  # Preprocess the ROI for prediction
        if not rois:
            print(f'roi_preprocessed: {roi_preprocessed}')
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
    image = draw_labeled_bounding_boxes(image, bboxes, predicted_classes)

    return image  # Return the image with bounding boxes and predictions