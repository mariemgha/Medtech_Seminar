import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from Training_with_PreProcessing import predict_new_video, extract_keypoints, process_video
from tensorflow.keras.models import load_model
import mediapipe as mp
import joblib

from cut_videos import cut_video

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

class VideoPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dart Throw Score Predictor")
        self.model = load_model(r'.\dart_throw_score_predictor_normalized_2Layers.h5') #path to your model
          # Load the scalers
        self.scaler = joblib.load(r'.\feature_scaler_2Layers.pkl') #path to your feature scaler
        self.score_scaler = joblib.load(r'.\score_scaler_2Layers.pkl') #path to your score scaler
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        self.camera_button = tk.Button(root, text="Open Camera", command=self.open_camera)
        self.camera_button.pack()

        self.stop_button = tk.Button(root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack()

        self.visualize_button = tk.Button(root, text="Visualize Angles", command=self.visualize_angles)
        self.visualize_button.pack() 

        self.predict_button = tk.Button(root, text="Predict Score", command=self.predict_score)
        self.predict_button.pack()

        self.prediction_label = tk.Label(root, text="Prediction: N/A")
        self.prediction_label.pack()

        self.video_path = None
        self.cap = None
        self.out = None
        self.recording = False

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.display_video()

    def open_camera(self):
        self.video_path = "live_camera_feed.mp4"
        self.cap = cv2.VideoCapture(0)
        self.out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 480))
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        self.recording = True
        self.process_camera()

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.recording = False
            self.cap.release()
            if self.out:
                self.out.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Info", "Video recording stopped and saved.")
        else:
            messagebox.showerror("Error", "Camera is not open or already stopped")

    def process_camera(self):
        if not self.recording:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.out.write(frame)
            self.process_and_display_frame(frame)
            self.root.after(10, self.process_camera)
        else:
            self.stop_camera()

    def display_video(self):
        def show_frame():
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display_frame(frame)
                self.root.after(10, show_frame)
            else:
                self.cap.release()

        show_frame()

    def resize_frame(self, frame, max_width=640, max_height=480):
        height, width = frame.shape[:2]
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame
    
    def process_and_display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks.landmark)
            for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
                x, y, z = keypoints[key]
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
            # Draw connections between keypoints
            connections = [
                ('M1', 'M2'),
                ('M2', 'M3'),
                ('M3', 'M4'),
                ('M4', 'M5')
            ]
            for (key1, key2) in connections:
                x1, y1, z1 = keypoints[key1]
                x2, y2, z2 = keypoints[key2]
                cv2.line(frame, (int(x1 * frame.shape[1]), int(y1 * frame.shape[0])), 
                                (int(x2 * frame.shape[1]), int(y2 * frame.shape[0])), (255, 0, 0), 2)
        frame_resized = self.resize_frame(frame)
        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def predict_score(self):
        print("Predict button clicked")
        if not self.video_path:
            messagebox.showerror("Error", "No video selected or recorded")
            print("No video path found")
            return

        print(f"Video path: {self.video_path}")
        
        preprocess = messagebox.askyesno("Preprocess Video", "Do you want to preprocess your video?")
        
        try:
            if preprocess:
                processed_video = cut_video(self.video_path)
                predicted_score = predict_new_video(self.model, processed_video,self.scaler, self.score_scaler)
            else:
                predicted_score = predict_new_video(self.model, self.video_path,self.scaler, self.score_scaler)
            self.prediction_label.config(text=f"Prediction: {predicted_score:.2f}")
            print(f"Prediction: {predicted_score:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")
            print(f"An error occurred during prediction: {e}")

    def visualize_angles(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected or recorded")
            return
        
        try:
            angles_data = process_video(self.video_path, True)
            self.cap = cv2.VideoCapture(self.video_path)
            self.display_video_with_angles(angles_data)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while visualizing angles: {e}")
            print(f"An error occurred while visualizing angles: {e}")

    def display_video_with_angles(self, angles_data):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Frame rate of the output video
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video_path = 'processed_video.avi'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        def show_frame_with_angles(frame_idx=0):
            if frame_idx < len(angles_data):
                ret, frame = self.cap.read()
                if not ret:
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    keypoints = extract_keypoints(results.pose_landmarks.landmark)
                    # Only process the specific keypoints for the angles of interest
                    joints_of_interest = ['M1', 'M2', 'M3', 'M4', 'M5']
                    for key in joints_of_interest:
                        if key in keypoints:
                            x, y, z = keypoints[key]
                            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

                    # Draw connections between the keypoints of interest
                    connections = [
                        ('M1', 'M2'),
                        ('M2', 'M3'),
                        ('M3', 'M4'),
                        ('M4', 'M5')
                    ]
                    for (key1, key2) in connections:
                        if key1 in keypoints and key2 in keypoints:
                            x1, y1, z1 = keypoints[key1]
                            x2, y2, z2 = keypoints[key2]
                            cv2.line(frame, (int(x1 * frame.shape[1]), int(y1 * frame.shape[0])),
                                    (int(x2 * frame.shape[1]), int(y2 * frame.shape[0])), (255, 0, 0), 2)

                # Calculate midpoint for putting text
                text_offset_x = 10
                text_offset_y = 10
                # Calculate the positions of the middle landmarks:
                x1, y1, z1 = keypoints['M4']  # shoulder
                x2, y2, z2 = keypoints['M3']  # Elbow
                x3, y3, z3 = keypoints['M2']  # Wrist

                # Determine the correct landmark and adjust text position accordingly
                text_position_shoulder = (int(x1 * frame.shape[1]) + text_offset_x, int(y1 * frame.shape[0]) + text_offset_y)
                text_position_elbow = (int(x2 * frame.shape[1]) + text_offset_x, int(y2 * frame.shape[0]) + text_offset_y)
                text_position_wrist = (int(x3 * frame.shape[1]) + text_offset_x, int(y3 * frame.shape[0]) + text_offset_y)

                # Draw angle values from JSON data
                angle_shoulder = angles_data[frame_idx]["θr.shoulder"]
                angle_elbow = angles_data[frame_idx]["θr.elbow"]
                angle_wrist = angles_data[frame_idx]["θr.wrist"]

                cv2.putText(frame, f"Shoulder: {angle_shoulder:.1f} deg", text_position_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Elbow: {angle_elbow:.1f} deg", text_position_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Wrist: {angle_wrist:.1f} deg", text_position_wrist, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #print(f"Angles at frame {frame_idx}: Shoulder={angle_shoulder:.1f}, Elbow={angle_elbow:.1f}, Wrist={angle_wrist:.1f}")
                # Write the frame into the output video file
                out.write(frame)
                frame_resized = self.resize_frame(frame)
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.root.after(10, show_frame_with_angles, frame_idx+1)  # Schedule next frame processing at 30 FPS
            else:
                # Finish processing and save the video
                self.cap.release()
                out.release()
                # Display the saved video with angles visualized
                self.cap = cv2.VideoCapture(output_video_path)
                #self.display_processed_angles_video()

        show_frame_with_angles()

    def display_processed_angles_video(self):
        def show_frame():
            ret, frame = self.cap.read()
            if ret:
                frame_resized = self.resize_frame(frame)
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.root.after(10, show_frame)  # Schedule next frame processing at 30 FPS
            else:
                self.cap.release()

        show_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPredictorApp(root)
    root.mainloop()