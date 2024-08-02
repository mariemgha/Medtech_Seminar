import cv2
import mediapipe as mp
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter, find_peaks

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

def calculate_angle(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude_v2 = math.sqrt(sum(a ** 2 for a in v2))
    angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle)

def extract_keypoints(landmarks):
    keypoints = {}
    keypoints['M1'] = (landmarks[20].x, landmarks[20].y, landmarks[20].z)
    keypoints['M2'] = (landmarks[16].x, landmarks[16].y, landmarks[16].z)
    keypoints['M3'] = (landmarks[14].x, landmarks[14].y, landmarks[14].z)
    keypoints['M4'] = (landmarks[12].x, landmarks[12].y, landmarks[12].z)
    keypoints['M5'] = (landmarks[11].x, landmarks[11].y, landmarks[11].z)
    return keypoints

def calculate_velocity(points):
    velocities = []
    for i in range(1, len(points)):
        velocity = math.sqrt(
            (points[i][0] - points[i-1][0])**2 +
            (points[i][1] - points[i-1][1])**2 +
            (points[i][2] - points[i-1][2])**2
        )
        velocities.append(velocity)
    return velocities

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    angles_list = []
    wrist_positions = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = extract_keypoints(landmarks)

            angles = {}
            angles['θr.shoulder'] = calculate_angle(
                (keypoints['M4'][0] - keypoints['M3'][0], keypoints['M4'][1] - keypoints['M3'][1], keypoints['M4'][2] - keypoints['M3'][2]),
                (keypoints['M4'][0] - keypoints['M5'][0], keypoints['M4'][1] - keypoints['M5'][1], keypoints['M4'][2] - keypoints['M5'][2])
            )
            angles['θr.elbow'] = calculate_angle(
                (keypoints['M3'][0] - keypoints['M2'][0], keypoints['M3'][1] - keypoints['M2'][1], keypoints['M3'][2] - keypoints['M2'][2]),
                (keypoints['M3'][0] - keypoints['M4'][0], keypoints['M3'][1] - keypoints['M4'][1], keypoints['M3'][2] - keypoints['M4'][2])
            )
            
            wrist_positions.append((keypoints['M2'][0], keypoints['M2'][1], keypoints['M2'][2]))
            angles_list.append(angles)
    
    cap.release()
    wrist_positions = wrist_positions[:-50]
    angles_list = angles_list[:-50]
    
    # Calculate velocities
    wrist_velocities = calculate_velocity(wrist_positions)

    # Apply Savitzky-Golay filter to smooth the velocity data
    wrist_velocities_smoothed = savgol_filter(wrist_velocities, window_length=91, polyorder=3)

    # Find all peaks
    peaks, _ = find_peaks(wrist_velocities_smoothed)

    # Exclude peaks in the first 100 frames
    min_frame = 110
    peaks = [peak for peak in peaks if peak >= min_frame]

    # Get peak values
    peak_values = wrist_velocities_smoothed[peaks]

    # Sort peaks by their values in descending order
    sorted_peak_indices = np.argsort(peak_values)[::-1]
    top_peak_indices = sorted_peak_indices[:15]  # Check more than 6 peaks initially

    # Convert to actual peak indices
    top_peaks = [peaks[idx] for idx in top_peak_indices]

    # Filter peaks based on minimum distance of 1 second (assuming fps = 30)
    fps = 30
    min_distance = 2* fps  # Minimum distance in frames
    filtered_peaks = []

    for peak in sorted(top_peaks):
        if all(abs(peak - fp) >= min_distance for fp in filtered_peaks):
            filtered_peaks.append(peak)
            if len(filtered_peaks) == 6:
                break

    # Handle cases where fewer than 6 peaks are found
    if len(filtered_peaks) < 6:
        print("Warning: Less than 6 valid peaks found. Ensure video data is sufficient and try adjusting parameters.")
        # Optionally, handle fewer peaks or choose the closest valid peaks

    # Sort filtered peaks for correct cutting order
    filtered_peaks.sort()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wrist_velocities_smoothed, label='Smoothed Wrist Velocities')
    for peak in filtered_peaks:
        plt.axvline(x=peak, color='r', linestyle='--')
    plt.xlabel('Frame Index')
    plt.ylabel('Velocity')
    plt.title('Smoothed Wrist Velocities over Frames')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Filtered peak indices:", filtered_peaks)
    output_angles = []
    for frame in filtered_peaks:
        release_time = frame / fps
        output_angles.append({
            'frame': int(frame),
            'release_time': release_time,
            'θr.shoulder': angles_list[frame]['θr.shoulder'],
            'θr.elbow': angles_list[frame]['θr.elbow']
        })
    
    output_angles = sorted(output_angles, key=lambda x: x['frame'])
    
    excel_file = 'output_angles.xlsx'
    if os.path.exists(excel_file):
        df_existing = pd.read_excel(excel_file)
        df_new = pd.DataFrame(output_angles)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(excel_file, index=False, engine='openpyxl')
    else:
        df = pd.DataFrame(output_angles)
        df.to_excel(excel_file, index=False, engine='openpyxl')

    return filtered_peaks, fps

def cut_video(input_video_path, release_times, fps):
    try:
        output_folder = r'C:\Users\mmari\Desktop\Codes\mp_qMRI_Frankfurt\deep-darts\test_vids'
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pre_buffer_seconds = 1
        post_buffer_seconds = 1

        for i, release_frame in enumerate(release_times):
            clip_start_frame = int(max(0, release_frame - pre_buffer_seconds * fps))
            clip_end_frame = int(min(total_frames - 1, release_frame + post_buffer_seconds * fps))

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

            output_path_cv2 = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_video_path))[0]}_throw_{i+1}.mp4")
            output_path_cv2 = output_path_cv2.replace("\\", "/")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_cv2, fourcc, fps, (frame_width, frame_height))

            if not out.isOpened():
                print("Error: VideoWriter could not be opened.")
                return None

            frame_count = clip_end_frame - clip_start_frame + 1
            frames_written = 0

            while frames_written < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1

            out.release()
            print(f"Video file {i+1} written successfully with OpenCV.")

        cap.release()

        return output_folder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    video_folder = r'C:\Users\mmari\Pictures\Camera Roll\Testing'
    for i in range(16,19):
        video_path = os.path.join(video_folder, f"Test_{i}.mp4")
        release_times, fps = process_video(video_path)
        cut_video(video_path, release_times, fps)
