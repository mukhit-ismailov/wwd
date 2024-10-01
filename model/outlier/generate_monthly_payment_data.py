import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
months = pd.date_range(start='2023-01-01', periods=12, freq='M')
recurring_types = ['base_salary', 'housing', 'transport', 'food', 'phone_bill']

# Create base recurring payments
base_payments = {
    'base_salary': 3000,
    'housing': 800,
    'transport': 200,
    'food': 300,
    'phone_bill': 100
}

# Generate sample data
data = []
for month in months:
    # Add recurring payments
    for payment_type, amount in base_payments.items():
        # Introduce a promotion anomaly every few months
        if month.month in [3, 6]:  # March and June as promotion months
            amount += np.random.randint(100, 500)  # Increase salary and hence other payments
        # Randomly introduce errors in some months (e.g., mistakes)
        elif np.random.rand() < 0.1:  # 10% chance of error
            amount += np.random.randint(-100, 100)  # Random error

        data.append({'month': month, 'payment_type': payment_type, 'amount': amount})

# Add some non-recurring payments (e.g., loan, business trip)
for month in months:
    if np.random.rand() < 0.5:  # 50% chance of having a non-recurring payment
        data.append({'month': month, 'payment_type': 'non_recurring', 'amount': np.random.randint(200, 1000)})

# Create DataFrame
df = pd.DataFrame(data)

# Display sample data
print(df.head(20))

df.to_csv("/Users/mukhit.ismailov/Work/MLModel/monthly_payment_data.csv", index=False)
