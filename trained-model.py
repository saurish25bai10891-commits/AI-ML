import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load Data
# Assuming dataset.csv has columns: 'Attendance_Pct', 'Study_Hours_Per_Week', 'Previous_Score', 'Pass_Final' (0 or 1)
try:
    data = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error: dataset.csv not found. Please ensure the data directory is populated.")
    exit()

# 2. Preprocess Data
X = data[['Attendance_Pct', 'Study_Hours_Per_Week', 'Previous_Score']]
y = data['Pass_Final']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
# Utilizing Logistic Regression as it is a fundamental, highly interpretable classification algorithm
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, predictions))

# 5. Feature Importance Visualization
importance = model.coef_[0]
plt.bar(X.columns, importance)
plt.title('Feature Importance for Predicting Student Success')
plt.ylabel('Coefficient Value')
plt.savefig('feature_importance.png')
print("Feature importance chart saved as 'feature_importance.png'.")
