import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_frame(frame, input_shape):
    """
    Preprocesses a video frame by resizing and normalizing.

    Args:
    - frame (numpy.ndarray): The frame to preprocess.
    - input_shape (tuple): The target size for resizing the frame (height, width).

    Returns:
    - normalized_frame (numpy.ndarray): The resized and normalized frame.
    """
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

def process_video(video_path):
    """
    Processes a video file, classifying each frame using the pre-trained model.

    Args:
    - video_path (str): Path to the video file to process.
    """
    model_path = 'path/to/your/model.h5'  # Update this path to the saved model
    model = load_model(model_path)

    sports = {0: 'football', 1: 'chess', 2: 'cricket', 3: 'table_tennis'}

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    input_shape = model.input_shape[1:3]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = preprocess_frame(frame, input_shape)

        # Make prediction
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))

        # Determine predicted sport
        predicted_sport = sports[np.argmax(prediction)]

        # Annotate frame with prediction
        cv2.putText(frame, predicted_sport, (50, 50), font, font_scale, (0, 255, 0), font_thickness)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def browse_video():
    """
    Opens a file dialog to select a video file and processes it.
    """
    video_path = filedialog.askopenfilename()
    if video_path:
        process_video(video_path)

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Sports Video Classifier")

# Label for instructions
label = tk.Label(root, text="Select a video file to classify:", font=("Helvetica", 14))
label.pack(pady=20)

# Browse button to select video file
browse_button = tk.Button(root, text="Browse", command=browse_video)
browse_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
