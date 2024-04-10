import os
from PIL import Image, UnidentifiedImageError
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def preprocess_images(data_dir, img_size, batch_size):
    try:
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
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
        for file_path in train_generator.filenames:
            print("Corrupted file:", os.path.join(data_dir, file_path))
        return None

def build_model(input_shape, num_classes):
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
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, verbose=1)
    return history

data_dir = r"D:\sportsml\data"
img_size = 150
batch_size = 32
output_model_path = os.path.join(r"D:\sportsml", "final_trained_model_2")

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
