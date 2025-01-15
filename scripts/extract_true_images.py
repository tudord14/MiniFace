"""
                                    -- TRUE DATA ACQUISITION --
-> This script is used for extracting frames from an already existing personal video recording OR
   it may be used for actually creating a video and auto-extracting the images required for the facial
   recognition algorithm
-> If the user chooses to use an already existing video he may need to include those videos in the pre-created
   "videos" directory which is used for storing all the videos used!
-> If the user chooses to record a new video he will need to choose the time for which the camera will be active
-> Using this video the script will extract each frame as a separate image and use OpenCV to detect the face, grayscale and resize

Prerequisites:
1. Ensure you have a small environment in which you can move or use to change lightning and background
2. If you already have a selfie style video make sure to insert it in the "videos" directory already created
"""

import cv2
import os
import time
import shutil


def list_videos(video_folder):
    """
    -> List existing videos in the videos directory and allow the user to select one or skip
    """

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    videos = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if videos:
        print("The available videos are:")
        for idx, video in enumerate(videos, start=1):
            print(f"{idx}. {video}")
        print("Type the number corresponding to the video you'd like to use or type 'skip' to record a new video!!!")
    else:
        print("No videos found in the videos directory! Let's record a new one!")
        return None

    user_input = input("Enter your choice: ").strip()
    if user_input == "skip":
        return None
    else:
        try:
            selected_index = int(user_input) - 1
            if 0 <= selected_index < len(videos):
                return os.path.join(video_folder, videos[selected_index])
            else:
                print("Invalid choice. Recording a new video.")
                return None
        except ValueError:
            print("Invalid input. Recording a new video.")
            return None


def record_video(output_path, record_seconds):
    """
    -> Record a video using the webcam for the specified number of seconds
    -> Save the video to the videos directory
    """

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Actually save the video for further uses!
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Recording for {record_seconds} seconds. Press 'q' to stop early.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # We show the video stream so you can actually see how you look ;)
        cv2.imshow("Recording (Press 'q' for stop)!!!!", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > record_seconds:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")


def extract_images_from_video(video_path, output_folder, face_cascade_path):
    """
    -> Input the video that was previously recorded or chosen
    -> Transform each frame into grayscale and resize 48x48
    """

    # We are gonna use the face cascade already present in the Project (not made by me!!!!)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Error: Could not load face cascade.")
        return

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(48, 48)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_region = grayscale[y:y + h, x:x + w]
            face_resized = cv2.resize(face_region, (48, 48))

            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, face_resized)
            print(f"Saved processed image to: {output_path}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Image extraction completed!!!!")


def main():
    # Define paths for the whole projects necessities
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(base_dir, "../videos")
    data_dir = os.path.join(base_dir, "../data")
    true_folder = os.path.join(data_dir, "True")
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Step 1: Check for existing videos or record
    selected_video = list_videos(video_folder)
    if selected_video:
        print(f"Using existing video: {selected_video}")
    else:
        os.makedirs(video_folder, exist_ok=True)
        selected_video = os.path.join(video_folder, "new_video.mp4")
        record_seconds = int(input("Enter the number of seconds to record: "))
        record_video(selected_video, record_seconds)

    # Step 2: Extract images
    extract_images_from_video(selected_video, true_folder, face_cascade_path)


if __name__ == "__main__":
    main()
