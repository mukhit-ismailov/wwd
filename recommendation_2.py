import pandas as pd
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import Dataset

# Sample Employee Data
employee_data = {
    'employee_id': [1, 2, 3, 4, 5],
    'position': ['Engineer', 'Manager', 'Analyst', 'Engineer', 'HR'],
    'years_of_experience': [3, 5, 2, 6, 4],
    'employee_grade': [3, 4, 2, 4, 3],
    'salary': [70000, 90000, 60000, 80000, 75000]
}

# Sample Training Data
training_data = {
    'training_id': [1, 2, 3],
    'training_name': ['Python for Data Science', 'Leadership Skills', 'Data Analysis with Excel']
}

# Sample Review Data
review_data = {
    'employee_id': [1, 1, 2, 3, 4, 5],
    'training_id': [1, 2, 1, 3, 2, 3],
    'review_score': [4.5, 5.0, 4.0, 3.5, 4.8, 4.2]
}

reviews = pd.DataFrame(review_data)
print(reviews)

trainings = pd.DataFrame(training_data)
print(trainings)

employees = pd.DataFrame(employee_data)
print(employees)


# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(reviews[['employee_id', 'training_id', 'review_score']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build the SVD model
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Compute and print RMSE (Root Mean Square Error)
rmse = accuracy.rmse(predictions)

print(f'RMSE: {rmse}')

