
import os
import cv2
from moviepy.editor import VideoFileClip
from release_time import process_video


def cut_video(input_video_path):
    try:
        output_folder = r'path/to/your/output/folder'
        os.makedirs(output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the release time from the process_video function
        release_time = process_video(input_video_path)
        print('Release time is:', release_time)

        pre_buffer_seconds = 1
        post_buffer_seconds = 1

        # Calculate frame indices for start and end times
        clip_start_frame = int(max(0, (release_time - pre_buffer_seconds) * fps))
        clip_end_frame = int(min(total_frames - 1, (release_time + post_buffer_seconds) * fps))

        # Set the video capture to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

        # Create VideoWriter object to save the output video
        output_path_cv2 = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_video_path))[0]}_throw_opencv.mp4")
        output_path_cv2 = output_path_cv2.replace("\\", "/")
        print('Output path (OpenCV):', output_path_cv2)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
        out = cv2.VideoWriter(output_path_cv2, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("Error: VideoWriter could not be opened.")
            return None

        # Read and write frames from the video
        frame_count = clip_end_frame - clip_start_frame + 1
        frames_written = 0

        while frames_written < frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Write the frame
            out.write(frame)
            frames_written += 1

        out.release()
        print("Video file written successfully with OpenCV.")

        # Release the video capture
        cap.release()

        return output_path_cv2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None