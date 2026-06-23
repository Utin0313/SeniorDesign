import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("your_file.csv")

# Add correctness
df["correct"] = (df["real"] == df["pred"]).astype(int)

cols = ["breast", "control", "prostate", "skin", "confidence", "correct"]

# Correlation
corr = df[cols].corr()

# Covariance
cov = df[cols].cov()

print("Correlation Matrix:\n", corr)
print("\nCovariance Matrix:\n", cov)

# Plot correlation heatmap
plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation of Model Outputs")
plt.show()