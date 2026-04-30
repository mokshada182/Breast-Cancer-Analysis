plt.figure(figsize=(10, 7))
for name, (y_true, y_score) in roc_data.items():
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:][::-1]
        features = X.columns[indices]
        plt.figure(figsize=(8, 5))
        plt.barh(features, importances[indices])
        plt.xlabel("Importance")
        plt.title(f"Top Features - {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

plot_feature_importance(models['Random Forest'], "Random Forest")
plot_feature_importance(models['XGBoost'], "XGBoost")

y_risk = models["Linear Regression"].predict(X_test)
plt.figure(figsize=(8, 5))
plt.hist(y_risk, bins=20, color='orange', edgecolor='black')
plt.title("Predicted Tumor Risk Scores (Linear Regression)")
plt.xlabel("Risk Score (0 = Benign, 1 = Malignant)")
plt.ylabel("Number of Cases")
plt.grid(True)
plt.tight_layout()
plt.show()
