import ffmpeg
import os
import glob
import cv2
from moviepy.editor import *

def extract_frames(video_path, output_dir='frames', interval=5, frame_size=(512, 512)):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    video = VideoFileClip(video_path)

    # Calculate the frame rate
    fps = video.fps

    # Calculate the number of frames to skip
    skip_frames = int(fps * interval)

    # Initialize the frame count
    frame_count = 0

    # Start frame extraction
    for frame_count in range(0, int(video.duration * fps), skip_frames):
        # Calculate the current time in seconds
        current_time = frame_count / fps

        # Format the timestamp as a string
        timestamp_str = f"{int(current_time):04d}"

        # Get the frame at the current frame count
        frame = video.get_frame(frame_count / fps)

        # Resize the frame to the specified size
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

        # Save the frame with the timestamp in the filename
        cv2.imwrite(f'{output_dir}/frame_{timestamp_str}.jpg', frame)

    # Close the video clip
    video.close()

if __name__ == "__main__":
    video_path = "data/mp4/"
    videos = glob.glob(video_path+"*")
    output_dir = "data/frames/"
    for video in videos:
        name = video.split("/")[-1].split(".")[0]
        tmp_output_dir = output_dir + name
        if not name.startswith('spaghetti'):
            continue
        extract_frames(video, tmp_output_dir)
