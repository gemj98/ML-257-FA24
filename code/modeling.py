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
data_dir = os.path.join(script_dir, '..', 'data', 'PKLot', 'cropped_dataset')

# Define the path to save the trained model
model_path = os.path.join(script_dir, 'model.keras')

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

# Load the validation dataset
val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,       # 20% of the data for validation
  subset="validation",       # Specify loading the validation subset
  seed=123,                   # Seed for reproducibility
  image_size=(img_height, img_width),  # Resize all images to 30x30 pixels
  batch_size=batch_size,      # Batch size for validation
  color_mode="grayscale")    # Load images as grayscale

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), 
           activation='relu', 
           input_shape=(img_height, img_width, 1)), # First convolutional layer
    MaxPooling2D(2, 2),                             # First max pooling layer
    Conv2D(64, (3, 3), activation='relu'),          # Second convolutional layer
    MaxPooling2D(2, 2),                             # Second max pooling layer
    Flatten(),                                      # Flatten the feature maps to a vector
    Dense(128, activation='relu'),                  # Fully connected layer with 128 neurons
    Dropout(0.5),                                   # Dropout for regularization to prevent overfitting
    Dense(1, activation='sigmoid')                  # Output layer with 1 neurons for binary classification
])

# Compile the model
# Using Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set the number of epochs for training
epochs = 3  # Number of passes through the entire dataset during training

# Train the model
history = model.fit(
    train_ds,                # Training data
    validation_data=val_ds,  # Validation data
    epochs=epochs)           # Number of epochs

# Save the trained model to the specified path
model.save(model_path)