import pandas as pd
import numpy as np
from functions import plot_cm

CHOICES = ["Inception V3", "VGG16", "tuned_model"]
MODEL_CHOICE = CHOICES[1]  # Select one of three architectures tested. Architecture of choice: VGG16
df = pd.read_csv("generated_preds.csv")

"""Due to the nature of this challenge being very sensitive to Type II errors, it should be considered to 
   lower the binary threshold, potentially sacrificing some accuracy to in turn ensure the highest possible 
   recall (ensure that as many actually malignant cases as possible are classified as malignant) """

THRESHOLD = 0.40  # Adjust to change binary prediction threshold

# Get preds with new threshold and see how many of the new preds are correct
df["new_prediction"] = np.where(df["Prediction Value"] > THRESHOLD, "Malignant", "Benign")
df["Correct Prediction"] = df["new_prediction"] == df["Actual"]

# Calculate accuracy
value_counts = df["Correct Prediction"].value_counts()
correct_predictions = value_counts.iloc[0]
incorrect_predictions = value_counts.iloc[1]
accuracy = correct_predictions / (correct_predictions + incorrect_predictions) * 100
print(f"Accuracy Score: {accuracy}%")

# Plot adjusted confusion matrix
plot_cm(df["new_prediction"], df["Actual"], f"{MODEL_CHOICE} ({THRESHOLD} Threshold)")
