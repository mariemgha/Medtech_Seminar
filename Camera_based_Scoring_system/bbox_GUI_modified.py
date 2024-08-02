import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from yacs.config import CfgNode as CN
import argparse
import os.path as osp
import pandas as pd
from train import build_model
from predict_v1_mofdified import predict


class DeepDartsApp:
    def __init__(self, root, model, cfg):
        self.root = root
        self.root.title("DeepDarts Test GUI")
        self.root.geometry("1080x980") 
         # Create a frame for the scores
        self.score_frame = tk.Frame(self.root, bg="white", padx=10, pady=10)
        self.score_frame.pack(side=tk.TOP, fill=tk.X)
        # Create a frame for the controls
        self.control_frame = tk.Frame(self.root, bg="white", padx=10, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = tk.Button(self.control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.live_button = tk.Button(self.control_frame, text="Open Live Stream", command=self.open_live_stream)
        self.live_button.pack(side=tk.LEFT, padx=5)
        
        self.start_button = tk.Button(self.control_frame, text="Start", command=self.start_processing, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the video display
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(expand=True, fill=tk.BOTH)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True)
        # Set a fixed size for the video label
        self.video_label.config(width=1080, height=1000)
        
        # Add labels for displaying scores in the score frame
        self.score1_label = tk.Label(self.score_frame, text="Score 1: 0", font=("Helvetica", 16), bg="white", padx=10)
        self.score1_label.pack(side=tk.LEFT)
        
        self.score2_label = tk.Label(self.score_frame, text="Score 2: 0", font=("Helvetica", 16), bg="white", padx=10)
        self.score2_label.pack(side=tk.LEFT)
        
        self.score3_label = tk.Label(self.score_frame, text="Score 3: 0", font=("Helvetica", 16), bg="white", padx=10)
        self.score3_label.pack(side=tk.LEFT)

        self.cap = None
        self.model = model
        self.cfg = cfg
        self.roi = None
        self.orb = None
        self.keypoints = None
        self.descriptors = None
        self.bf = None
        self.dart_bboxes = []  # List to store bounding boxes of detected darts
        self.stable_bboxes = []  # List to store bounding boxes that have been stable over a few frames
        self.frames_stable = []  # List to count frames each bbox has been stable
        self.stability_threshold = 2  # Number of frames a bbox must remain stable to be considered a fixed dart
        self.dart_count=0
        self.filtered_bboxes=[]
        self.previous_image=None
        #self.processed_frames_dir = 'processed_frames'
        #if not os.path.exists(self.processed_frames_dir):
         #   os.makedirs(self.processed_frames_dir)

        # Ensure max_darts is set
        if not hasattr(self.cfg.model, 'max_darts'):
            self.cfg.model.max_darts = 3

        self.video_path = None
        self.fps = 30  # Default FPS
        self.frame_skip_count = 10  # Assuming 30 fps, skip 60 frames to process every 2 seconds
        self.current_frame = 0
        self.saved_scores = [0, 0, 0]  # Placeholder for three darts' scores
        self.processing = False
        # Initialize variables and setup
        self.trackers = []  # Initialize trackers attribute


    def load_video(self):
        self.video_path = filedialog.askopenfilename()
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Handle case if FPS is not available
        self.frame_skip_count = int(self.fps)  # Process every second
        self.processing = True
        self.initialize_template_matching()
        self.process_video()
    
    def find_usb_camera_index():
        index = 1
        while index < 10:  # Limit the search to 10 devices
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"Camera found at index {index}")
                cap.release()
                return index
            cap.release()
            index += 1
        print("No camera found")
        return None

    def open_live_stream(self):
        webcam_index=self.find_usb_camera_index()
        if webcam_index is None:
            print("Failed to find an available USB webcam.")
            return
        self.cap = cv2.VideoCapture(webcam_index)  # 0 is usually the built-in webcam
        if not self.cap.isOpened():
            print("Failed to open webcam.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Handle case if FPS is not available
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.show_live_stream()  
        self.stop_button.config(state=tk.DISABLED)

    def show_live_stream(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize the frame to fit the Tkinter window
                frame = self.resize_to_fit_window(frame)
                # Convert to PIL Image and display in Tkinter
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Call this function again after a short delay to create a live video effect
            self.root.after(int(1000 / self.fps), self.show_live_stream)

    def initialize_template_matching(self):
        #please load the paths to your templates
        template_path_1 = r'template1.jpg'
        template_path_2 = r'template2.png'
        template_path_3 = r'template3.jpg'
        
        # Try loading and processing templates
        self.template = cv2.imread(template_path_1)
        if self.template is None:
            print("Error: Could not read template image 1.")
            # Attempt with template 2 and 3
            self.try_template_matching(template_path_2, template_path_3)
            
            return
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Perform template matching with template 1
        if not self.match_template_and_process():
            # Attempt with template 2 and 3
            self.try_template_matching(template_path_2, template_path_3)
        
    def match_template_and_process(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return False

            self.height, self.width, _ = frame.shape
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
            best_match = None
            best_val = -1
            best_loc = None

            for scale in scales:
                resized_template = cv2.resize(self.template_gray, (0, 0), fx=scale, fy=scale)
                # Skip if resized template is larger than the frame
                if resized_template.shape[0] > gray_frame.shape[0] or resized_template.shape[1] > gray_frame.shape[1]:
                    continue
                result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_match = resized_template
                    best_loc = max_loc

            if best_match is None or best_val < 0.5: 
                print("Warning: No match found for template 1.")
                return False
            
            self.top_left = best_loc
            h, w = best_match.shape
            self.bottom_right = (self.top_left[0] + w, self.top_left[1] + h)
            cv2.rectangle(frame, self.top_left, self.bottom_right, (0, 255, 0), 2)

            self.roi = frame[self.top_left[1]:self.bottom_right[1], self.top_left[0]:self.bottom_right[0]]
            self.orb = cv2.ORB_create()
            self.keypoints, self.descriptors = self.orb.detectAndCompute(self.roi, None)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
           
            return True

        else:
            print("Error: Video capture is not open.")
            return False

    def try_template_matching(self, template_path, fallback_template_path):
        self.template = cv2.imread(template_path)
        if self.template is None:
            print(f"Error: Could not read template image {template_path}.")
            # Attempt with fallback template
            self.template = cv2.imread(fallback_template_path)
            if self.template is None:
                print(f"Error: Could not read fallback template image {fallback_template_path}.")
                self.cap.release()
                self.root.quit()  
                return
            
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        if not self.match_template_and_process():
            print(f"Warning: No match found for template {template_path}.")
            if template_path != fallback_template_path:
                print(f"Attempting with fallback template {fallback_template_path}...")
                self.template = cv2.imread(fallback_template_path)
                if self.template is None:
                    print(f"Error: Could not read fallback template image {fallback_template_path}.")
                    self.cap.release()
                    self.root.quit()  
                    return
                
                self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                if not self.match_template_and_process():
                    print(f"Warning: No match found for fallback template {fallback_template_path}.")
                    self.cap.release()
                    self.root.quit() 

    def process_live_video(self):
        if not self.processing:
            return
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from live stream.")
            self.cap.release()
            return

        # Detect keypoints and compute descriptors in the current frame
        kp_frame, des_frame = self.orb.detectAndCompute(frame, None)

        # Match descriptors between ROI and current frame
        matches = self.bf.match(self.descriptors, des_frame)

        # Sort matches by distance (lower distance is better)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw the top matches
        match_img = cv2.drawMatches(self.roi, self.keypoints, frame, kp_frame, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display matches
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", self.width, self.height)
        cv2.imshow("Matches", match_img)

        if self.current_frame % self.frame_skip_count == 0:
            if self.bbox:
                x, y, w, h = self.bbox
                cropped_frame = frame[y:y + h, x:x + w]
                img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                img_rgb = self.resize_with_padding(img_rgb)
                preds, img = predict(self.model, img_rgb, self.cfg)
                self.update_scores(preds)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.current_frame += 1
        self.root.after(int(1000 / self.fps), self.process_live_video)

    def start_processing(self):
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        if self.cap is not None and self.cap.isOpened():
            # Initialize template matching
            self.initialize_template_matching()
            # Start processing the live video
            self.process_video()

    def stop_processing(self):
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    
    def filter_bboxes(self,img, decimal_places=2):
        try:

            if self.previous_image is not None:
                
                # Convert images to grayscale
                current_image_gray = cv2.cvtColor(self.previous_image, cv2.COLOR_BGR2GRAY)
                next_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Compute image difference
                image_diff = cv2.absdiff(current_image_gray, next_image_gray)
                # Apply Gaussian blur
                blurred_image_diff = cv2.GaussianBlur(image_diff, (5, 5), 0)
                # Apply morphological closing to remove small white points (noise)
                kernel = np.ones((5, 5), np.uint8)
                morph_image_diff = cv2.morphologyEx(blurred_image_diff, cv2.MORPH_CLOSE, kernel, iterations=2)
                # Threshold the image to create a binary image
                _, image_diff_thresh = cv2.threshold(morph_image_diff, 15, 255, cv2.THRESH_BINARY)
                img_matrix = np.array(image_diff_thresh)
                # Find contours of connected components
                contours, _ = cv2.findContours(img_matrix.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                local_bboxes=[]
                for contour in contours:
                    if cv2.contourArea(contour)<1500:
                        continue
                    # Create a copy of the original image to draw bounding boxes on
                    #original_image = np.zeros_like(img_matrix, dtype=np.uint8)
                    # Iterate through each contour (each connected component of ones)
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw bounding box on the original image
                    cv2.rectangle(image_diff_thresh, (x, y), (x + w, y + h), (255), 2)
                    box=(x/800,y/800,(x+w)/800,(y+h)/800)
                    local_bboxes.append(box)
                
                # Find the biggest box based on area
                if local_bboxes:
                    max_box = max(local_bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    if not self.frames_stable:
                        # If frames_stable is empty, append the max_box and the current frame
                        self.filtered_bboxes.append(max_box)
                        self.frames_stable.append(self.current_frame)
                    else:
                        # If frames_stable is not empty, perform the check
                        if (self.current_frame - self.frames_stable[-1]) < 2:
                            self.filtered_bboxes = self.filtered_bboxes[:-1]  # Remove the last element
                            self.filtered_bboxes.append(max_box)              # Append the max_box
                            self.frames_stable.append(self.current_frame)     # Append the current frame
                        else:
                            self.frames_stable = []                          # Reset frames_stable
                            self.filtered_bboxes.append(max_box)              # Append the max_box
                            self.frames_stable.append(self.current_frame)     # Append the current frame

                    
                # Show the original image with bounding boxes
                #cv2.imshow('Bounding Boxes', image_diff_thresh)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            # Update the previous image
            self.previous_image = img
            return self.filtered_bboxes
        except Exception as e:

            print(f"Error occurred: {e}")

            return self.filtered_bboxes

    def process_video(self):
        if not self.processing:
            return
        
        ret, frame = self.cap.read()

        try:
            if not ret:
                print("Reached end of video or failed to read frame.")
                self.cap.release()  # Release the video capture object
                return
            
            #Detect the dart appearing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not hasattr(self, 'prev_frame'):
                self.prev_frame = gray_frame
                self.current_frame += 1
                self.root.after(int(1000 / self.fps), self.process_video)
                return
            frame_height,frame_width=self.prev_frame.shape[:2]
            # Compute the absolute difference between the current frame and the previous frame
            diff_frame = cv2.absdiff(self.prev_frame, gray_frame)
            # Threshold the difference to get the regions with significant changes
            _, thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)
            # Debug: Display the diff and threshold frames
            #cv2.imshow("Difference Frame", diff_frame)
            #cv2.imshow("Threshold Frame", thresh_frame)
            #cv2.waitKey(1)
             # Apply morphological operations to remove noise
            #kernel = np.ones((5, 5), np.uint8)
            #thresh_frame = cv2.dilate(thresh_frame, kernel, iterations=2)
            #thresh_frame = cv2.erode(thresh_frame, kernel, iterations=1)

            # Find contours of the thresholded regions
            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_dart_bbox = None
            similar_frames=True
            # Uncomment if the individual frames are to be saved
            # a=0
            # b=0
            # c=0
            # d=0
            for contour in contours:
                if cv2.contourArea(contour) < 1100:  # Filter out small contours
                   continue
                # Get the bounding box for the contour
                (x, y, w, h) = cv2.boundingRect(contour)
                # Draw the bounding box on the original frame
                #cv2.rectangle(thresh_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Uncomment if the individual frames are to be saved
                # a=x
                # b=y
                # c=w
                # d=h
                 # Assuming only one dart appears at a time, take the first large enough contour
                #new_dart_bbox = (x/frame_width, y/frame_height, (x + w)/frame_width, (y + h)/frame_height)
                new_dart_bbox=(x/frame_width, y/frame_height, (x+w)/frame_width,(y+h)/frame_height)
                self.dart_count+=1
                similar_frames=False
                self.dart_bboxes.append(new_dart_bbox)
                #print('new dart:  ', new_dart_bbox)
                break
            #if new_dart_bbox:
             #self.update_stable_bboxes(new_dart_bbox)
          
            # Detect keypoints and compute descriptors in the current frame
            kp_frame, des_frame = self.orb.detectAndCompute(frame, None)

            # Match descriptors between ROI and current frame
            matches = self.bf.match(self.descriptors, des_frame)

            # Sort matches by distance (lower distance is better)
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw the top matches
            match_img = cv2.drawMatches(self.roi, self.keypoints, frame, kp_frame, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Resize the match_img to fit the Tkinter window
            match_img = self.resize_to_fit_window(match_img)

            # Convert to PIL Image and display in Tkinter
            img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Uncomment if the individual frames are to be saved
            #save = f"bbox_{self.current_frame}.png"
            #cv2.imwrite(save, cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 0), 2))

            # Extract the ROI from the current frame
            x, y, w, h = self.top_left[0], self.top_left[1], self.bottom_right[0] - self.top_left[0], self.bottom_right[1] - self.top_left[1]
            current_roi = frame[y:y + h, x:x + w]    
            if current_roi is not None:
                    img_rgb = cv2.cvtColor(current_roi, cv2.COLOR_BGR2RGB)
                    img_rgb = self.resize_with_padding(img_rgb)
                    #cv2.imshow('img',img_rgb)
                    if self.current_frame<2:
                        self.previous_image=img_rgb
                    else:
                        self.filtered_bboxes=self.filter_bboxes(img_rgb)
                        print(len(self.filtered_bboxes))
                        print(self.filtered_bboxes)
            if self.current_frame % self.frame_skip_count == 0:
                # Process current ROI frame
                    if similar_frames==True and self.dart_count>1:
                        if len(self.stable_bboxes)>0:
                            is_unique = True
                            for bbox in self.stable_bboxes:
                                if self.is_same_bbox(bbox,self.dart_bboxes[-1]):
                                 is_unique = False
                                 break
                            # Add the new bounding box only if it is unique
                            if is_unique:
                                self.stable_bboxes.append(self.dart_bboxes[-1])
                        else: 
                            self.stable_bboxes.append(self.dart_bboxes[-1])
                        if len(self.stable_bboxes) <= 3:
                            print('stable: ',self.stable_bboxes)
                            preds, img = predict(self.model, img_rgb, self.cfg,self.stable_bboxes,self.filtered_bboxes)
                            
                            # Update dart positions and scores based on prediction results
                            self.update_scores(preds)
                            
                            # Save processed frame if needed
                            #filename = os.path.join(self.processed_frames_dir, f"processed_frame_{self.current_frame}.png")
                            #cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))# Save processed frame if needed
                            filename = f"processed_frame_{self.current_frame}.png"
                            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                # Draw the stable bounding boxes on the frame
                #for bbox in self.stable_bboxes:
                #  x1, y1, x2, y2 = bbox
                #   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
        except Exception as e:
            print(f"Error occurred: {e}")
        
        # Update the previous frame
        self.prev_frame = gray_frame

        self.current_frame += 1
        self.root.after(int(1000 / self.fps), self.process_video)  # Schedule next frame processing
    
    def is_same_bbox(self,bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_new, y1_new, x2_new, y2_new = bbox2
        return abs(x1 - x1_new) ==0 and abs(y1 - y1_new) ==0 and abs(x2 - x2_new) ==0 and abs(y2 - y2_new) ==0


    def resize_to_fit_window(self, frame):
        window_width = 1080  # Fixed width for the video label
        window_height = 1000  # Fixed height for the video label
        frame_height, frame_width = frame.shape[:2]

        # Determine the scaling factor
        scale = min(window_width / frame_width, window_height / frame_height)
        
        # Compute the new dimensions of the frame
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Create a new image with the window size and a black background
        result = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Center the resized frame within the result image
        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2
        result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
        
        return result

    def update_scores(self, scores):
        try:
            # Remove '0' elements from the input scores
            scores = [element for element in scores if element != '0']
            print('scores: ',scores)
            # Iterate over the new scores
            for new in scores:
                # Find an index in saved_scores where the current score is '0'
                for i, saved in enumerate(self.saved_scores):
                    if saved == 0:
                        # Update the score if it's not already in saved_scores
                        if new not in self.saved_scores:
                            self.saved_scores[i] =  str(new)
                            break  # Move to the next new score after updating
            print('saved scores: ', self.saved_scores)
            # Update GUI labels with the current scores
            self.score1_label.config(text=f"Score 1: {self.saved_scores[0]}")
            self.score2_label.config(text=f"Score 2: {self.saved_scores[1]}")
            self.score3_label.config(text=f"Score 3: {self.saved_scores[2]}")
            
        except Exception as e:
            print(f"Error updating scores: {e}")

    def resize_with_padding(self, frame, target_size=800):
        h, w = frame.shape[:2]
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))

        resized_img = cv2.resize(frame, (new_w, new_h))
        padded_img = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        return padded_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d2')
    parser.add_argument('-s', '--split', default='val')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    model = build_model(cfg)
    model.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    root = tk.Tk()
    app = DeepDartsApp(root, model, cfg)
    root.mainloop()