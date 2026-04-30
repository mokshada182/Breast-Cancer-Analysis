roc_data = {}

print("=== Model Evaluation ===\n")
for name, model in models.items():
    model.fit(X_train, y_train)

    if "Linear Regression" in name:
        y_pred = model.predict(X_test)
        y_class = (y_pred >= 0.5).astype(int)
        print(f"{name}:")
        print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"  R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"  Accuracy (threshold 0.5): {accuracy_score(y_test, y_class):.4f}")
        roc_data[name] = (y_test, y_pred)
        print()
    else:
        y_class = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"{name}:")
        print(f"  Accuracy: {accuracy_score(y_test, y_class):.4f}")
        print(f"  Precision: {precision_score(y_test, y_class):.4f}")
        print(f"  Recall: {recall_score(y_test, y_class):.4f}")
        print(f"  AUC: {roc_auc_score(y_test, y_prob):.4f}")
        roc_data[name] = (y_test, y_prob)
        print()
