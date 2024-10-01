import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
# Assuming you have two CSV files: 'employees.csv' and 'payments.csv'
employees_df = pd.read_csv('../recommendation/employees.csv')  # Contains Employee ID, Min Salary, Max Salary
payments_df = pd.read_csv('../recommendation/payments.csv')    # Contains Employee ID, Month, Amount Paid, Transfer Type

# Step 2: Merge DataFrames
merged_df = pd.merge(payments_df, employees_df, on='Employee ID')

# Step 3: Feature Engineering
# Calculate expected base salary (you can adjust this logic based on your needs)
merged_df['Expected Base Salary'] = (merged_df['Min Salary'] + merged_df['Max Salary']) / 2

# Step 4: Prepare Data for Model
# Select relevant features for anomaly detection
features = merged_df[['Amount Paid', 'Expected Base Salary']]
features['Anomaly Score'] = np.nan

# Step 5: Train Isolation Forest Model
model = IsolationForest(contamination=0.05)  # Adjust contamination based on expected anomaly rate
model.fit(features[['Amount Paid', 'Expected Base Salary']])

# Step 6: Predict Anomalies
features['Anomaly Score'] = model.predict(features[['Amount Paid', 'Expected Base Salary']])
features['Anomaly'] = features['Anomaly Score'].map({1: 0, -1: 1})  # Convert to binary (0: normal, 1: anomaly)

# Step 7: Analyze Results
anomalies = features[features['Anomaly'] == 1]
print("Detected Anomalies:")
print(anomalies)

# Optional: Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=features, x='Amount Paid', y='Expected Base Salary', hue='Anomaly', palette={0: 'blue', 1: 'red'})
plt.title('Anomaly Detection in Employee Payments')
plt.xlabel('Amount Paid')
plt.ylabel('Expected Base Salary')
plt.axhline(y=merged_df['Expected Base Salary'].mean(), color='green', linestyle='--', label='Mean Expected Salary')
plt.legend()
plt.show()