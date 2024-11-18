# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:25:39 2024

@author: gemj9
"""
# Import necessary libraries
import os
import matplotlib.pyplot as plt
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Get the directory of the script
script_dir = os.path.dirname(__file__)

# Define the path to the dataset directory
# The dataset contains cropped images of parking spaces (assumed structure is labeled subdirectories)
data_dir = os.path.join(script_dir, '..', 'data', 'PKLot', 'cropped_dataset_1')

# Define the path to save the trained model
model_path = os.path.join(script_dir, 'model_3.keras')

# Hyperparameters for the dataset loading
batch_size = 32  # Number of samples per batch
img_height = 15  # Image height in pixels (resized for uniformity)
img_width = 15   # Image width in pixels (resized for uniformity)

# Load the training dataset
# Splitting data into training and validation subsets using an 80-20 split
train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,       # 20% of the data for validation
  subset="training",         # Specify loading the training subset
  seed=123,                   # Seed for reproducibility
  image_size=(img_height, img_width),  # Resize all images to 30x30 pixels
  batch_size=batch_size,      # Batch size for training
  color_mode="grayscale",     # Load images as grayscale
  label_mode='binary')    

# Load the validation dataset
val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,       # 20% of the data for validation
  subset="validation",       # Specify loading the validation subset
  seed=123,                   # Seed for reproducibility
  image_size=(img_height, img_width),  # Resize all images to 30x30 pixels
  batch_size=batch_size,      # Batch size for validation
  color_mode="grayscale",     # Load images as grayscale
  label_mode='binary')     

# Normalize the images
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# Optimize for performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
# Function to visualize images with their labels
def visualize_dataset(dataset, title):
    plt.figure(figsize=(15, 60))
    for images, labels in dataset.take(1):  # Take one batch of data
        for i in range(10*4):  # Display 9 images
            if i == 0:
                print(images[i])
            ax = plt.subplot(10, 4, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(f"Label: {int(labels[i].numpy())}")
            plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Visualize training dataset
visualize_dataset(train_ds, "Training Dataset")

# Visualize validation dataset
visualize_dataset(val_ds, "Validation Dataset")