"""
                                    -- FACIAL RECOGNITION MODEL TRAINING --
-> This script is used for the creation and training of the actual facial recognition model
-> It should be noted that the structure I have put together for this model is a general one
   that I have tested on multiple subjects and it gave as good results as a script of this size could give :)
-> This should make as a quick and fun project that anybody could create and test and actually use in
   personal projects and more
-> The aim of this project was mainly to create an algorithm that anybody could use!
-> In my mind this project could be used in many applications, especially ones that use low-power devices
   such as Raspberry Pi and Jetson Nano
-> A lot of interesting projects could arise from this one!

Prerequisites:
1. Ensure all other scripts have been run and that the /data folder actually has 2 subfolders True and False
"""


import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Constants used across the program
IMG_WIDTH = 48
IMG_HEIGHT = 48
BATCH_SIZE = 32
EPOCHS = 50

# Paths for necessary directories/files
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "../data")
true_dir = os.path.join(data_dir, "True")
false_dir = os.path.join(data_dir, "False")
models_dir = os.path.join(base_dir, "../models")

os.makedirs(models_dir, exist_ok=True)


def balance_classes(true_dir, false_dir):
    """
    -> Check and balance the number of images in the True and False folders for training
    -> Randomly select images from the larger folder to match the smaller folders size for training
    """

    true_images = [os.path.join(true_dir, f) for f in os.listdir(true_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    false_images = [os.path.join(false_dir, f) for f in os.listdir(false_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    true_count = len(true_images)
    false_count = len(false_images)

    print(f"True images: {true_count}, False images: {false_count}")

    if true_count > false_count:
        true_images = random.sample(true_images, false_count)
    elif false_count > true_count:
        false_images = random.sample(false_images, true_count)

    print("Balancing done!")

    return true_images, false_images


def create_data_generators(true_images, false_images, img_width, img_height, batch_size):
    """
    -> Create training and validation data generators from balanced datasets
    """

    # Combine true and false images with corresponding labels
    all_images = true_images + false_images
    all_labels = [1] * len(true_images) + [0] * len(false_images)

    # Shuffle the dataset
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # Split into training and validation sets 80/20
    split_idx = int(0.8 * len(all_images))
    train_images, val_images = all_images[:split_idx], all_images[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

    # Convert to arryas
    train_images = np.array([cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (img_width, img_height)) for img in train_images])
    val_images = np.array([cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (img_width, img_height)) for img in val_images])

    # Rescale pixel values
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Add grayscale channel dim images
    train_images = train_images[..., np.newaxis]
    val_images = val_images[..., np.newaxis]

    # Convert labels to arrays
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # Create tf datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).shuffle(len(train_images))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)

    return train_dataset, val_dataset


def build_model(img_width, img_height):
    """
    -> Build and compile the CNN model.
    -> The model is already structured by me but it should be a pretty good and robust one for succesfully achieving
       our goal of creating a facial recognition algorithm
    """

    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_width, img_height, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the actual model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_save_model(model, train_dataset, val_dataset, epochs, models_dir):
    """
    -> Train the model and save it to the models directory.
    """

    print("Starting model training!")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # Save the model in the models directory in the main Project
    model_path = os.path.join(models_dir, "my_model.h5")
    model.save(model_path, include_optimizer=True)
    print(f"Model saved to: {model_path}")

    return history


def main():
    # Step 1: Balance the classes
    true_images, false_images = balance_classes(true_dir, false_dir)

    # Step 2: Create data generators
    train_dataset, val_dataset = create_data_generators(true_images, false_images, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)

    # Step 3: Build the model
    model = build_model(IMG_WIDTH, IMG_HEIGHT)
    model.summary()

    # Step 4: Train the model and save it
    train_and_save_model(model, train_dataset, val_dataset, EPOCHS, models_dir)


if __name__ == "__main__":
    main()
