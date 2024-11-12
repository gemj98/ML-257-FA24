# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:19:31 2024

@author: gemj9
"""
import os
from tensorflow.keras.utils import image_dataset_from_directory

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data', 'PKLot', 'PKLot_cropped')
model_path = os.path.join(script_dir, 'model.keras')

batch_size = 32
img_height = 30
img_width = 30

train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",        
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode="grayscale")

val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode="grayscale")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

model.save(model_path)

