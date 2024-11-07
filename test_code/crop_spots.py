import os
from PIL import Image
from collections import defaultdict
import json
from tqdm import tqdm  # Import tqdm for the progress bar


# Set the image directory path relative to the script location
script_dir = os.path.dirname(__file__)
image_dir = os.path.join(script_dir, 'data', 'PKLot')
save_dir = os.path.join(script_dir, 'data', 'PKLot', 'cropped_dataset')
# labels_cropped_path = os.path.join(save_dir, '_labels.json')

data_folders = ["train", "valid", "test"]
categories = ["empty", "non-empty"]

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
for category in categories:
    if not os.path.exists(os.path.join(save_dir, category)):
        os.makedirs(os.path.join(save_dir, category), exist_ok=True)



# Function to preprocess annotations and crop images based on bounding boxes
def crop_images(image_dir='path_to_images/', save_dir='path_to_save/', output_prefix="default"):
    json_file_path = os.path.join(image_dir, folder, "_annotations.coco.json")
    # Open the file and load the data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # Now `data` contains the JSON data as a Python dictionary

    # Step 1: Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
        
    # Step 2: Iterate over each image and process annotations
    with tqdm(total=len(data["images"]), desc=f"Processing images in '{folder}' folder:") as pbar:
        for image_info in data["images"]:
            image_id = image_info["id"]
            image_path = os.path.join(image_dir, folder, image_info["file_name"])   
    
            try:
                # Open the image once
                with Image.open(image_path) as img:
                    # Retrieve annotations for this image_id
                    annotations = annotations_by_image[image_id]
    
                    # Crop for each annotation
                    for annotation in annotations:
                        bbox = annotation["bbox"]
                        x, y, width, height = bbox
                        right = x + width
                        bottom = y + height
    
                        # Crop the image
                        cropped_img = img.crop((x, y, right, bottom))
                        
                        # Save the cropped image with a unique name
                        category_name = categories[0] if annotation["category_id"] == 1 else categories[1]
                        cropped_img_name = f"{output_prefix}_{image_id}_{annotation['id']}.jpg"
                        cropped_img_path = os.path.join(save_dir, category_name, cropped_img_name)
                        cropped_img.save(cropped_img_path)
                        
                # Update progress bar after processing each image
                pbar.update(1)

            except FileNotFoundError:
                print(f"Image file not found: {image_path}")
                pbar.update(1)  # Still update the progress bar if an image is missing

for folder in data_folders:
    crop_images(image_dir=image_dir, save_dir=save_dir, output_prefix=folder)
