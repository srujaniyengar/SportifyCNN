# Sports Video Classifier

## Overview

Welcome to the Sports Video Classifier project! This project uses a Convolutional Neural Network (CNN) to classify sports activities in video files. It is built using TensorFlow and Keras for the model and OpenCV for video processing. The graphical user interface (GUI) is created using Tkinter, allowing users to select a video file and see real-time sports classification results.

**Note:** This project is intended for educational purposes to demonstrate image classification using deep learning and real-time video processing.

## Features

### Model Capabilities

- **Sport Classification:** Classifies video frames into one of four categories: Football, Chess, Cricket, or Table Tennis.
- **Real-Time Processing:** Displays the classified sport label on the video in real-time.

### GUI

- **Video Selection:** Allows users to select a video file from their system.
- **Frame Annotation:** Shows the classified sport name over the video frames.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `tkinter`, `opencv-python`, `numpy`, `tensorflow`
- A trained model saved in H5 format (see the **Training** section below)

### Training

1. **Prepare Your Data:**
   - Organize your dataset into directories for each sport with images for training.
   - Use the `train_model.py` script (not shown here) to train and save the model.

2. **Save Your Model:**
   - Ensure the model is saved as an H5 file in the specified path in `process_video()`.

### Running the Application

1. **Run the Video Classification Script:**

   ```bash
   python classify_video.py
   Use the GUI:
Click the "Browse" button to select a video file.
The selected video will be processed, and the classified sport will be displayed on the video.
Contact
For any questions or feedback, please reach out to srujanparthasarathyiyengar@gmail.com
