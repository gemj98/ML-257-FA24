import os
from PIL import Image
from collections import defaultdict
import json
from tqdm import tqdm  # Import tqdm for the progress bar

# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
image_dir = os.path.join(script_dir, '..', 'data', 'PKLot')
save_dir = os.path.join(script_dir, '..', 'data', 'PKLot', 'cropped_dataset')

data_folders = ["train", "valid"]  # Define subfolders for training and validation data
categories = ["empty", "non-empty"]  # Define categories corresponding to parking space status

# Ensure the save directory and category subdirectories exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
for category in categories:
    if not os.path.exists(os.path.join(save_dir, category)):
        os.makedirs(os.path.join(save_dir, category), exist_ok=True)

# Function to crop a region of interest (ROI) from an image based on bounding box
# bbox is expected to be in the format [x, y, width, height]
def crop_roi(img, bbox):
    x, y, width, height = bbox
    right = x + width
    bottom = y + height
    return img.crop((x, y, right, bottom))

# Function to preprocess annotations and crop images based on bounding boxes
def crop_images(image_dir='path_to_images/', save_dir='path_to_save/', output_prefix="default"):
    # Define the path to the COCO annotations JSON file
    json_file_path = os.path.join(image_dir, folder, "_annotations.coco.json")

    # Open the annotations file and load the data
    with open(json_file_path, 'r') as file:
        data = json.load(file)  # Load JSON data into a Python dictionary

    # Step 1: Group annotations by image_id for efficient processing
    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    # Step 2: Iterate over each image and process its annotations
    with tqdm(total=len(data["images"]), desc=f"Processing images in '{folder}' folder:") as pbar:
        for image_info in data["images"]:
            image_id = image_info["id"]
            image_path = os.path.join(image_dir, folder, image_info["file_name"])

            try:
                # Open the image file
                with Image.open(image_path) as img:
                    # Retrieve annotations for the current image
                    annotations = annotations_by_image[image_id]

                    # Crop and save each annotated region
                    for annotation in annotations:
                        bbox = annotation["bbox"]  # Bounding box coordinates
                        cropped_img = crop_roi(img, bbox)  # Crop the image based on the bounding box

                        # Determine the category name based on category_id
                        category_name = categories[0] if annotation["category_id"] == 1 else categories[1]

                        # Create a unique name for the cropped image
                        cropped_img_name = f"{output_prefix}_{image_id}_{annotation['id']}.jpg"
                        cropped_img_path = os.path.join(save_dir, category_name, cropped_img_name)

                        # Save the cropped image to the appropriate category folder
                        cropped_img.save(cropped_img_path)

                # Update the progress bar after processing each image
                pbar.update(1)

            except FileNotFoundError:
                # Handle missing image files and update the progress bar
                print(f"Image file not found: {image_path}")
                pbar.update(1)

# Process images in both training and validation data folders
for folder in data_folders:
    crop_images(image_dir=image_dir, save_dir=save_dir, output_prefix=folder)
