import pandas as pd
import numpy as np
from functions import plot_cm
df = pd.read_csv("./submission.csv")

threshold = 0.5  # Adjust to change binary prediction threshold

# Get preds with new threshold and see how many of the new preds are correct
df["new_prediction"] = np.where(df["Prediction Value"] > threshold, "Malignant", "Benign")
df["Correct Prediction"] = df["new_prediction"] == df["Actual"]

# Calculate accuracy
value_counts = df["Correct Prediction"].value_counts()
correct_predictions = value_counts.iloc[0]
incorrect_predictions = value_counts.iloc[1]
accuracy = correct_predictions / (correct_predictions + incorrect_predictions) * 100
print(f"Accuracy Score: {accuracy}%")

# Plot adjusted confusion matrix
plot_cm(df["new_prediction"], df["Actual"], f"Inception V3 ({threshold} Threshold)")
