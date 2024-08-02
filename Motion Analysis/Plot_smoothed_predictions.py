import pandas as pd
import matplotlib.pyplot as plt

# Load the saved CSV file
results_df = pd.read_csv(r'.s\predicted_vs_actual_scores_2Layers.csv') #path to your csv file predicted vs actual scores

# Sort the DataFrame by video name (assuming video names have numerical order in their names)
results_df = results_df.sort_values(by='video_name').reset_index(drop=True)

# Add a new column for video number
results_df['video_number'] = range(1, len(results_df) + 1)

# Define the window size for the moving average
window_size = 10

# Calculate the moving average for actual and predicted scores
results_df['actual_score_smooth'] = results_df['actual_score'].rolling(window=window_size, min_periods=1).mean()
results_df['predicted_score_smooth'] = results_df['predicted_score'].rolling(window=window_size, min_periods=1).mean()

# Plot actual and predicted scores as line plots
plt.figure(figsize=(14, 8))

# Plot the smoothed data
plt.plot(results_df['video_number'], results_df['actual_score_smooth'], marker='o', color='blue', label='Actual Score (Smoothed)')
plt.plot(results_df['video_number'], results_df['predicted_score_smooth'], marker='x', color='red', linestyle='--', label='Predicted Score (Smoothed)')

# Optionally plot the original data points for reference
# plt.plot(results_df['video_number'], results_df['actual_score'], marker='o', color='blue', alpha=0.3, label='Actual Score')
# plt.plot(results_df['video_number'], results_df['predicted_score'], marker='x', color='red', linestyle='--', alpha=0.3, label='Predicted Score')

plt.xlabel('Video Number')
plt.ylabel('Score')
plt.title('Actual vs Predicted Scores (Smoothed)')
plt.legend()
plt.grid(True)
plt.show()