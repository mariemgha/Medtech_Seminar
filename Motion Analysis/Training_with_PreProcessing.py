import cv2
import mediapipe as mp
import json
import os
import math
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import glob
import joblib

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Added to draw landmarks

def calculate_angle(v1, v2):
    dot_product = sum(p*q for p,q in zip(v1, v2))
    magnitude1 = math.sqrt(sum([val**2 for val in v1]))
    magnitude2 = math.sqrt(sum([val**2 for val in v2]))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def extract_keypoints(landmarks):
    keypoints = {}
    # Mapping landmarks to your specific keypoints
    keypoints['M1'] = (landmarks[20].x, landmarks[20].y, landmarks[20].z)
    keypoints['M2'] = (landmarks[16].x, landmarks[16].y, landmarks[16].z)
    keypoints['M3'] = (landmarks[14].x, landmarks[14].y, landmarks[14].z)
    keypoints['M4'] = (landmarks[12].x, landmarks[12].y, landmarks[12].z)
    keypoints['M5'] = (landmarks[11].x, landmarks[11].y, landmarks[11].z)
    keypoints['M6'] = (landmarks[24].x, landmarks[24].y, landmarks[24].z)
    keypoints['M7'] = (landmarks[23].x, landmarks[23].y, landmarks[23].z)
    keypoints['M8'] = (landmarks[26].x, landmarks[26].y, landmarks[26].z)
    keypoints['M9'] = (landmarks[25].x, landmarks[25].y, landmarks[25].z)
    keypoints['M10'] = (landmarks[28].x, landmarks[28].y, landmarks[28].z)
    keypoints['M11'] = (landmarks[27].x, landmarks[27].y, landmarks[27].z)
    keypoints['M12'] = (landmarks[32].x, landmarks[32].y, landmarks[32].z)
    keypoints['M13'] = (landmarks[31].x, landmarks[31].y, landmarks[31].z)
    return keypoints

def calculate_velocity(points):
    velocities = []
    for i in range(1, len(points)):
        velocity = math.sqrt((points[i][0] - points[i-1][0])**2 +
                             (points[i][1] - points[i-1][1])**2 +
                             (points[i][2] - points[i-1][2])**2)
        velocities.append(velocity)
    return velocities

def moving_average(data, window_size=3):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def smooth_angles(angles, window_size=3):
    smoothed_angles = []
    for angle_seq in angles:
        smoothed_seq = []
        for i in range(len(angle_seq)):
            if i < window_size - 1:
                smoothed_seq.append(angle_seq[i])
            else:
                smoothed_window = {key: moving_average([angle_seq[j][key] for j in range(i-window_size+1, i+1)])[-1] for key in angle_seq[i]}
                smoothed_seq.append(smoothed_window)
        smoothed_angles.append(smoothed_seq)
    return smoothed_angles

def process_video(video_path, testing):
    cap = cv2.VideoCapture(video_path)
    angles_list = []
    wrist_positions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}")
        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = extract_keypoints(landmarks)

            # Calculate angles
            angles = {}
            angles['θr.shoulder'] = calculate_angle(
                (keypoints['M4'][0] - keypoints['M3'][0], keypoints['M4'][1] - keypoints['M3'][1], keypoints['M4'][2] - keypoints['M3'][2]),
                (keypoints['M4'][0] - keypoints['M5'][0], keypoints['M4'][1] - keypoints['M5'][1], keypoints['M4'][2] - keypoints['M5'][2])
            )
            angles['θr.elbow'] = calculate_angle(
                (keypoints['M3'][0] - keypoints['M2'][0], keypoints['M3'][1] - keypoints['M2'][1], keypoints['M3'][2] - keypoints['M2'][2]),
                (keypoints['M3'][0] - keypoints['M4'][0], keypoints['M3'][1] - keypoints['M4'][1], keypoints['M3'][2] - keypoints['M4'][2])
            )
            angles['θr.wrist'] = calculate_angle(
                (keypoints['M2'][0] - keypoints['M1'][0], keypoints['M2'][1] - keypoints['M1'][1], keypoints['M2'][2] - keypoints['M1'][2]),
                (keypoints['M2'][0] - keypoints['M3'][0], keypoints['M2'][1] - keypoints['M3'][1], keypoints['M2'][2] - keypoints['M3'][2])
            )
            
            # Append wrist position for velocity calculation
            wrist_positions.append((keypoints['M2'][0], keypoints['M2'][1], keypoints['M2'][2]))
            angles_list.append(angles)
    
    cap.release()
    if testing == False:
        if not wrist_positions:
            print("Error: No wrist positions detected in the video.")
            return None, None, None

        print("Wrist positions detected.")

        # Calculate wrist velocities
        wrist_velocities = calculate_velocity(wrist_positions)

        if not wrist_velocities:
            print("Error: No wrist velocities calculated.")
            return None, None, None

        print("Wrist velocities calculated.")

        # Detect dart release frame based on peak velocity
        release_frame = np.argmax(wrist_velocities)  # Example: detecting maximum velocity point
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: Frames per second property is zero. Assuming FPS = 30.")
            fps = 30  # Assuming 30 FPS if not available from the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in the video: {total_frames}")
        
    return angles_list

def save_angles_to_json(video_files, output_dir):
    all_angles = {}
    for video in video_files:
        angles = process_video(video, False)
        video_name = os.path.splitext(os.path.basename(video))[0]
        with open(os.path.join(output_dir, f"{video_name}_angles.json"), 'w') as f:
            json.dump(angles, f, indent=4)
        all_angles[video_name] = angles
    return all_angles

def load_angles_from_json(json_dir):
    all_angles = []
    video_names = []
    for json_file in os.listdir(json_dir):
        if json_file.endswith("_angles.json"):
            video_name = os.path.splitext(json_file)[0].replace('_angles', '')
            with open(os.path.join(json_dir, json_file), 'r') as f:
                angles = json.load(f)
                all_angles.append(angles)
                video_names.append(video_name)
    print(f"Loaded angles from {len(all_angles)} JSON files")
    return all_angles, video_names

def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq += [{}] * (max_len - len(seq))
        padded.append(seq)
    return padded

def preprocess_data(angles, scores):
    # Flatten all angles for scaling
    all_angles_flat = []
    for angles_seq in angles:
        for angle in angles_seq:
            all_angles_flat.extend([angle[key] for key in sorted(angle.keys())])
    
    scaler = StandardScaler()
    scaler.fit(np.array(all_angles_flat).reshape(-1, 1))
    
    # Normalize and smooth angles
    normalized_angles = []
    for angles_seq in angles:
        normalized_seq = []
        for angle in angles_seq:
            normalized_angle = scaler.transform(np.array([angle[key] for key in sorted(angle.keys())]).reshape(-1, 1)).flatten()
            normalized_seq.append(dict(zip(sorted(angle.keys()), normalized_angle)))
        normalized_angles.append(normalized_seq)
    
    # Apply smoothing
    smoothed_angles = smooth_angles(normalized_angles, window_size=3)
    
    max_len = max(len(a) for a in smoothed_angles)
    padded_angles = pad_sequences(smoothed_angles, max_len)
    
    angles_flat = []
    for angles_seq in padded_angles:
        seq = []
        for angles in angles_seq:
            if angles:
                seq.append([angles[key] for key in sorted(angles.keys())])
            else:
                seq.append([0]*3)  # Use 0s for padding
        angles_flat.append(seq)
    
    X = np.array(angles_flat)
    
    # Normalize the scores
    score_scaler = MinMaxScaler()
    y = score_scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    
    print(f"Preprocessed data shape: {X.shape}")
    
    return X, y, scaler, score_scaler

def build_lstm_model(input_shape):
    import tensorflow as tf
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

def predict_new_video(model, video_path, scaler, score_scaler):
    print(f"Processing new video for prediction: {video_path}")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    angles = process_video(video_path, False)

    if not angles:
        print("No angles extracted from the video.")
        return None

    max_len = model.input_shape[1] 
    num_features = model.input_shape[2]

    # Normalize angles using the same scaler used during training
    normalized_angles = []
    for angle in angles:
        normalized_angle = scaler.transform(np.array([angle[key] for key in sorted(angle.keys())]).reshape(-1, 1)).flatten()
        normalized_angles.append(dict(zip(sorted(angle.keys()), normalized_angle)))
    
    # Apply smoothing
    smoothed_angles = smooth_angles([normalized_angles], window_size=3)[0]
    
    padded_angles = pad_sequences([smoothed_angles], max_len)
    angles_flat = []
    for angles_seq in padded_angles:
        seq = []
        for angles in angles_seq:
            if angles:
                seq.append([angles[key] for key in sorted(angles.keys())])
            else:
                seq.append([0] * num_features)  # Use 0s for padding
        angles_flat.append(seq)

    X_new = np.array(angles_flat, dtype=np.float32)
    X_new = X_new.reshape((1, max_len, num_features))

    print("Shape of X_new:", X_new.shape)
    print("Type of X_new:", type(X_new))

    prediction = model.predict(X_new)
    predicted_score = score_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
    return predicted_score

if __name__ == "__main__":
    video_files = glob.glob(r'.\Train_model_dataset\*.mp4')
    output_dir = r'.s\All Training JSONS'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #save the angles in JSON files
    print("Extracting angles from videos...")
    save_angles_to_json(video_files, output_dir)
    
    json_dir = r'.\All Training JSONS'
    excel_file = r'.\Final_scores.xlsx'
    
    #Extract angles from JSON files
    print("Loading angles from JSON files...")
    all_angles, video_names = load_angles_from_json(json_dir)
    
    scores_df = pd.read_excel(excel_file)
    scores = []
    for video_name in video_names:
        score = scores_df.loc[scores_df['video_name'] == video_name, 'score'].values
        if len(score) > 0:
            scores.append(score[0])
        else:
            print(f"Warning: No score found for video {video_name}")
            scores.append(None)
    
    filtered_angles = [angles for angles, score in zip(all_angles, scores) if score is not None]
    filtered_scores = [score for score in scores if score is not None]

    print("Preprocessing data...")
    X, y, scaler, score_scaler = preprocess_data(filtered_angles, filtered_scores)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print("Building and training LSTM model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)
    
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    
    model.save('dart_throw_score_predictor_normalized_2Layers.h5')
    print("Model saved.")
    # Save the scalers
    joblib.dump(scaler, 'feature_scaler_2Layers.pkl')
    joblib.dump(score_scaler, 'score_scaler_2Layers.pkl')
    print("saved scalers.")
    
    # Uncomment to test on a single video
    # from tensorflow.keras.models import load_model
    # import joblib
    # model = load_model('dart_throw_score_predictor_normalized.h5')
    # # Load the scalers
    # scaler = joblib.load('feature_scaler.pkl')
    # score_scaler = joblib.load('score_scaler.pkl')

    # new_video_path = r'path/to/test/video.mp4'
    # print(f"Predicting score for new video: {new_video_path}")
    # predicted_score = predict_new_video(model, new_video_path, scaler, score_scaler)
    # print(f"Predicted score for the new throw: {predicted_score}")
