"""
                                    -- FACIAL RECOGNITION MODEL TRAINING --
-> Testing part of the project
-> Here we will test the previously trained model on a real-time video stream
-> A live video feed will open and each face (up to 5) that get in the video will
   get bounded by a red or green box depending if the face is detected or not

Prerequisites:
1. Ensure all other scripts have been run and that the model is present in the models directory
"""

import cv2
import numpy as np
import tensorflow as tf
import os

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "../models")
model_path = os.path.join(models_dir, "my_model.h5")

# Load the trained model
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)


def preprocess_face(face_region):
    """
    -> Preprocess the detected face for prediction
    -> Convert to grayscale, resize 48x48, and normalize pixel values
    -> Return the reshaped face image
    """
    face_resized = cv2.resize(face_region, (48, 48))
    face_normalized = face_resized / 255.0
    face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))
    return face_reshaped


def predict_face(face_region):
    """
    -> Uses our created model to predict if the face is True or False
    """
    processed_face = preprocess_face(face_region)
    prediction = model.predict(processed_face)
    return prediction[0][0]


def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error!!! Could not open webcam!")
        return

    print("Press q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exit now!")
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            face_region = grayscale_frame[y:y + h, x:x + w]
            prediction = predict_face(face_region)

            if prediction >= 0.5:
                color = (0, 255, 0)
                label = "True"
            else:
                color = (0, 0, 255)
                label = "False"

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed!")

if __name__ == "__main__":
    main()
