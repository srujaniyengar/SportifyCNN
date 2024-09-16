import os
from PIL import Image, UnidentifiedImageError
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model

def preprocess_images(data_dir, img_size, batch_size):
    """
    Preprocesses images from the specified directory.
    
    Args:
    - data_dir (str): Directory where the image data is stored.
    - img_size (int): Target size for resizing images.
    - batch_size (int): Number of images to yield per batch.

    Returns:
    - train_generator: A Keras ImageDataGenerator instance for training.
    """
    try:
        # Create an instance of ImageDataGenerator for preprocessing
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        # Create a generator for the training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            classes=['football', 'chess', 'cricket', 'table_tennis']
        )
        return train_generator
    except UnidentifiedImageError as e:
        print(f"Encountered an unidentified image error: {e}. Skipping the current image.")
        return None

def build_model(input_shape, num_classes):
    """
    Builds a Convolutional Neural Network (CNN) model.
    
    Args:
    - input_shape (tuple): Shape of the input images (height, width, channels).
    - num_classes (int): Number of output classes.

    Returns:
    - model: A Keras Sequential model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, train_generator, epochs):
    """
    Trains the CNN model on the training data.
    
    Args:
    - model: A Keras Sequential model.
    - train_generator: A Keras ImageDataGenerator instance for training data.
    - epochs (int): Number of epochs to train the model.

    Returns:
    - history: A Keras History object containing training metrics.
    """
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, verbose=1)
    return history

def save_model(model, output_path):
    """
    Saves the trained model to the specified path.
    
    Args:
    - model: A Keras Sequential model.
    - output_path (str): Path to save the model.
    """
    model.save(output_path)

# Configuration
data_dir = 'path/to/your/data'  # Update this path as needed
img_size = 150
batch_size = 32
output_model_path = 'path/to/save/your/model'

# Preprocess images and train model if preprocessing is successful
train_generator = preprocess_images(data_dir, img_size, batch_size)

if train_generator:
    input_shape = (img_size, img_size, 3)
    num_classes = len(train_generator.class_indices)
    model = build_model(input_shape, num_classes)
    epochs = 30
    history = train_model(model, train_generator, epochs)
    save_model(model, output_model_path)
else:
    print("Failed to preprocess images. Skipping model training.")
