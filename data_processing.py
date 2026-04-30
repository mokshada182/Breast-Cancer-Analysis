
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load your dataset here, e.g., df = pd.read_csv("data.csv")

# Map diagnosis to numeric: M = 1 (Malignant), B = 0 (Benign)
df['diagnosis_numeric'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Target variable
y = df['diagnosis_numeric']

# Features (excluding diagnosis columns)
X = df.drop(columns=['diagnosis', 'diagnosis_numeric'])

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Correlation Matrix
plt.figure(figsize=(18, 16))
correlation_matrix = df.select_dtypes(include=['number']).corr()
mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool))

sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=False,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.3
)
plt.title("Correlation Matrix of Breast Cancer Features", fontsize=20)
plt.show()

# Feature Importance Visualization
important_features = ['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean']
for feature in important_features:
    plt.figure(figsize=(9, 7))
    sns.violinplot(
        x=y.map({0: 'Benign', 1: 'Malignant'}),
        y=df[feature],
        palette='RdPu'
    )
    plt.title(f'{feature.replace("_", " ").title()} by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel(feature.replace("_", " ").title())
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
