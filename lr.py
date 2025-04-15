import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("D:/Downloads/student.csv")

# Display column names for verification
print("Available columns:", list(df))

# Prepare feature matrix and target vector
features_df = df[['study_hours', 'assignmnet_score', 'attendance ', 'mse_score']]
target = df['subject_grade']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on test data
grade_predictions = regressor.predict(X_test)

# Extract model parameters
intercept_val = regressor.intercept_
coeff_vals = regressor.coef_
feature_names = features_df.columns

# Construct regression equation string
regression_eq = f"subject_grade = {intercept_val:.2f}"
for feature, coeff in zip(feature_names, coeff_vals):
    regression_eq += f" + ({coeff:.2f} * {feature.strip()})"

# Calculate evaluation metrics
r_squared = r2_score(y_test, grade_predictions)
mean_actual = np.mean(y_test)
SST = np.sum((y_test - mean_actual) ** 2)
SSE = np.sum((y_test - grade_predictions) ** 2)
SSR = SST - SSE

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, grade_predictions, color='skyblue', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Comparison of Actual and Predicted Grades")
plt.grid(True)
plt.tight_layout()
plt.show()

# Display model summary and evaluation
print("\nFinal Regression Equation:")
print(regression_eq)
print(f"\nR² Score (Accuracy): {r_squared:.4f}")
print(f"SST (Total Variance): {SST:.4f}")
print(f"SSR (Explained Variance): {SSR:.4f}")
print(f"SSE (Unexplained Error): {SSE:.4f}")

# Interactive prediction
print("\n-- Predict Subject Grade Based on New Input --")
try:
    input_hours = float(input("Study hours per day: "))
    input_assignment = float(input("Assignment score: "))
    input_attendance = float(input("Attendance (%): "))
    input_mse = float(input("MSE score: "))

    custom_input = np.array([[input_hours, input_assignment, input_attendance, input_mse]])
    estimated_grade = regressor.predict(custom_input)

    print(f"\nPredicted Subject Grade: {estimated_grade[0]:.2f}")
except Exception as err:
    print(f"Error in input: {err}")
