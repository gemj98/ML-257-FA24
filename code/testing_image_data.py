# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:19:31 2024

@author: gemj9
"""

# Import necessary libraries
import os
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Get the directory of the script
script_dir = os.path.dirname(__file__)

# Define the path to the dataset directory
# The dataset contains cropped images of parking spaces (assumed structure is labeled subdirectories)
data_dir = os.path.join(script_dir, '..', 'data', 'PKLot', 'cropped_dataset_1')

# Define the path to save the trained model
model_path = os.path.join(script_dir, 'model_1.keras')

# Hyperparameters for the dataset loading
batch_size = 32  # Number of samples per batch
img_height = 30  # Image height in pixels (resized for uniformity)
img_width = 30   # Image width in pixels (resized for uniformity)

# Load the training dataset
# Splitting data into training and validation subsets using an 80-20 split
train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,       # 20% of the data for validation
  subset="training",         # Specify loading the training subset
  seed=123,                   # Seed for reproducibility
  image_size=(img_height, img_width),  # Resize all images to 30x30 pixels
  batch_size=batch_size,      # Batch size for training
  color_mode="grayscale")    # Load images as grayscale

# Iterate through the dataset to get one batch
for image_batch, label_batch in train_ds.take(1):
    # Extract the first image and its label from the batch
    image = image_batch[0].numpy()  # Convert Tensor to NumPy array
    label = label_batch[0].numpy()  # Convert Tensor to NumPy array

    # Print the image values and its corresponding label
    print("Image shape:", image.shape)
    print("Image pixel values:\n", image)
    print("Label:", label)