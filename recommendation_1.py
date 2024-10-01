import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

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
    # Add other question scores similarly...
})

# Merging datasets
merged_data = reviews.merge(employees[['EmployeeID']], on='EmployeeID').merge(trainings[['TrainingID']], on='TrainingID')

# Prepare data for Surprise library
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(merged_data[['EmployeeID', 'TrainingID', 'Q1_Score']], reader)

# Train-test split
trainset, testset = train_test_split(data.build_full_trainset(), test_size=0.25)

# Model training using SVD
model = SVD()
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
rmse = mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions], squared=False)
print(f'RMSE: {rmse}')

# Making recommendations for a specific employee
employee_id = "1" # Example Employee ID
training_ids = trainings['TrainingID'].tolist()
predictions = [(training_id, model.predict(employee_id, training_id).est) for training_id in training_ids]
recommended_trainings = sorted(predictions, key=lambda x: x[1], reverse=True)

print("Recommended Trainings:")
for training in recommended_trainings:
    print(f"Training ID: {training[0]}, Predicted Score: {training[1]}")