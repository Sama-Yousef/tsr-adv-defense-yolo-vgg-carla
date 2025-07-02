import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video and saves them as .jpg images.
    
    :param video_path: Path to the input video file.
    :param output_folder: Directory where extracted frames will be saved.
    :param frame_interval: Extract every nth frame (default is 1, meaning extract all frames).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames and saved in {output_folder}")

# Example usage
video_path = "C:/CarlaYolo/CudaPytorch/Scripts/Ue5v2.mp4"  # Replace with your video file path
output_folder = "Ue5v2"         # Replace with your desired output folder
frame_interval = 1              # Extract every 10th frame

extract_frames(video_path, output_folder, frame_interval)
