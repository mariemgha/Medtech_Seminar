import glob
import os
from Training_with_PreProcessing import predict_new_video
import pandas as pd

# Directory containing the videos
video_files = glob.glob(r'\Train_model_dataset\Video_*.mp4')  # Adjust the path accordingly (for test dataset change Video_ to Test_)

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('dart_throw_score_predictor_normalized_2Layers.h5')
import joblib
# Load the scalers
scaler = joblib.load('feature_scaler_2Layers.pkl')
score_scaler = joblib.load('score_scaler_2Layers.pkl')
# Predict scores for all videos
predicted_scores = {}
for video in video_files:
    video_name = os.path.splitext(os.path.basename(video))[0]
    print(f"Predicting score for video: {video_name}")
    predicted_score = predict_new_video(model, video,scaler, score_scaler)
    predicted_scores[video_name] = predicted_score
    print(f"Predicted score for {video_name}: {predicted_score}")

# Load actual scores from the Excel file
excel_file = r'.\Training_scores.xlsx'
if not os.path.exists(excel_file):
    print(f"Error: Excel file not found at {excel_file}")
    exit(1)

try:
    scores_df = pd.read_excel(excel_file)
    print("Scores DataFrame loaded successfully:")
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit(1)

# Extract actual scores from the DataFrame
actual_scores = {}
for idx, row in scores_df.iterrows():
    video_name = row['video_name']
    score = row['score']
    actual_scores[video_name] = score

# Compare predicted scores with actual scores
common_videos = set(predicted_scores.keys()) & set(actual_scores.keys())
if not common_videos:
    print("No common videos found between predictions and actual scores.")
    exit(1)

y_pred = [predicted_scores[video] for video in common_videos]
y_true = [actual_scores[video] for video in common_videos]

# Plot predicted scores vs. actual scores
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Predicted Scores vs. Actual Scores')
plt.legend()
plt.show()

# Save predicted scores to a CSV file for reference
results_df = pd.DataFrame({'video_name': list(common_videos), 'actual_score': y_true, 'predicted_score': y_pred})
results_df.to_csv('predicted_vs_actual_scores_2Layers.csv', index=False)
print("Results saved to predicted_vs_actual_scores_2Layers.csv")