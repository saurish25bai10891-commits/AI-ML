import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# ====================== 1. Load or Create Data ======================
if not os.path.exists('dataset.csv'):
    print("dataset.csv not found → Creating a synthetic dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic student data
    Attendance_Pct = np.random.uniform(55, 100, n_samples)
    Study_Hours_Per_Week = np.random.uniform(4, 45, n_samples)
    Previous_Score = np.random.uniform(35, 98, n_samples)
    
    # Create target (Pass_Final) with logical relationships + noise
    logit = (0.06 * (Attendance_Pct - 75) +
             0.09 * (Study_Hours_Per_Week - 20) +
             0.07 * (Previous_Score - 70))
    
    prob = 1 / (1 + np.exp(-logit))
    Pass_Final = np.random.binomial(1, prob)
    
    data = pd.DataFrame({
        'Attendance_Pct': Attendance_Pct,
        'Study_Hours_Per_Week': Study_Hours_Per_Week,
        'Previous_Score': Previous_Score,
        'Pass_Final': Pass_Final
    })
    data.to_csv('dataset.csv', index=False)
    print(f"✅ Synthetic dataset created ({len(data)} samples)")
else:
    data = pd.read_csv('dataset.csv')
    print(f"✅ Loaded existing dataset ({len(data)} samples)")

# ====================== 2. Preprocess ======================
X = data[['Attendance_Pct', 'Study_Hours_Per_Week', 'Previous_Score']]
y = data['Pass_Final']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================== 3. Train Model ======================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ====================== 4. Evaluate ======================
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print("               MODEL EVALUATION")
print("="*50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ====================== 5. Feature Importance ======================
importance = model.coef_[0]
feature_names = X.columns

print("\n" + "-"*50)
print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
print("-"*50)
for name, coef in zip(feature_names, importance):
    print(f"{name:25} : {coef:+.4f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(feature_names, importance, color='skyblue', edgecolor='navy')
plt.title('Feature Importance for Predicting Student Success (Pass/Fail)')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("\n✅ Feature importance chart saved as 'feature_importance.png'")

# Bonus: Show probabilities for first few test samples
print("\nSample Predictions (first 5 test samples):")
probs = model.predict_proba(X_test[:5])[:, 1]
for i in range(5):
    print(f"Sample {i+1:2d}:  Predicted = {predictions[i]} | "
          f"Pass Probability = {probs[i]:.3f} | Actual = {y_test.iloc[i]}")
