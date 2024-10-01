import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Sample DataFrames
employees = pd.DataFrame({
    'EmployeeID': [1,2,3,4,5],
    'Position': ['Software Engineer', 'Data Analyst', 'Project Manager', 'HR Specialist', 'Marketing Manager'],
    'YearsOfExperience': [5,3,8,2,6],
    'EmployeeGrade': ['A', 'B', 'A', 'C', 'B'],
    'Salary': [70000,60000,90000,50000,75000]
})

trainings = pd.DataFrame({
    'TrainingID': ['T1','T2','T3','T4','T5'],
    'TrainingName': ['Advanced Python', 'Data Analysis with Excel', 'Project Management Essentials', 'Effective Communication', 'Marketing Strategies']
})

reviews = pd.DataFrame({
    'ReviewID': [1,2,3,4,5],
    'EmployeeID': [1,2,3,4,5],
    'TrainingID': ['T1','T2','T3','T4','T5'],
    'Q1_Score': [5,4,5,3,4],
    'Q2_Score': [4,3,5,2,4],
    'Q3_Score': [5,4,5,3,4],
    'Q4_Score': [4,3,5,2,4],
    'Q5_Score': [5,4,5,3,4],
    'Q6_Score': [4,3,5,2,4],
    'Q7_Score': [5,4,5,3,4],
    'Q8_Score': [4,3,5,2,4],
    'Q9_Score': [5,4,5,3,4],
    'Q10_Score': [4,3,5,2,4]
})

# Merge datasets
merged_data = reviews.merge(employees[['EmployeeID']], on='EmployeeID').merge(trainings[['TrainingID']], on='TrainingID')

# Create a pivot table for KNN
pivot_table = merged_data.pivot_table(index='EmployeeID', columns='TrainingID', values=['Q1_Score', 'Q2_Score', 'Q3_Score',
                                                                                      'Q10_Score'], aggfunc='mean').fillna(0)

# Flatten the multi-level columns
pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]

# Prepare data for KNN
X = pivot_table.values

# Fit KNN model
knn = NearestNeighbors(n_neighbors=2) # You can adjust the number of neighbors
knn.fit(X)

# Making recommendations for a specific employee
employee_index = pivot_table.index.get_loc(0) # Example Employee ID (0 corresponds to EmployeeID=1)
distances, indices = knn.kneighbors(X[employee_index].reshape(1,-1))

# Get recommended trainings based on nearest neighbors
recommended_trainings = []
for i in indices.flatten():
    if i != employee_index: # Avoid recommending the same training
        training_id = merged_data.loc[merged_data['EmployeeID'] == pivot_table.index[i], 'TrainingID'].values[0]
        recommended_trainings.append(training_id)

print("Recommended Trainings for Employee ID (1):")
for training in recommended_trainings:
    print(training)