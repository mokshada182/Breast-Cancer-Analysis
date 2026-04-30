import numpy as np

new_data = np.array([[
    14.2, 20.2, 92.4, 600.7, 0.1, 0.12, 0.1, 0.07, 0.2, 0.07,
    0.5, 1.2, 3.5, 40.0, 0.007, 0.03, 0.03, 0.01, 0.02, 0.005,
    16.0, 30.0, 110.0, 800.0, 0.14, 0.3, 0.3, 0.15, 0.4, 0.1
]])

prediction = models['Logistic Regression'].predict(new_data)
probability = models['Logistic Regression'].predict_proba(new_data)[0][1]

print("Prediction: Malignant" if prediction[0] == 0 else "Prediction: Benign")
print(f"Probability of being benign: {probability:.2f}")
