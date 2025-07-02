import cv2
import os

# Set the folder containing images
image_folder = 'D:frames_ZC_speedlimit30_defended'  # Change to your directory
video_name = 'D:frames_ZC_speedlimit30_defended.mp4'  # Output video file name
frame_rate =30 #25  # Frames per second

# Get all image files and sort them
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort()  # Ensure proper ordering

# Read the first image to get dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Loop through images and write them into the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)  # Add frame to video

# Release the video writer
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")
