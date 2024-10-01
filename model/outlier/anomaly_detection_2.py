from scipy import stats
import pandas as pd

# Sample data
df = pd.read_csv("/monthly_payment_data.csv")
# Pivot the DataFrame to have payment types as columns
pivot_df = df.pivot(index='month', columns='payment_type', values='amount')

# Calculate z-scores for each payment type across months
z_scores = stats.zscore(pivot_df.fillna(0), axis=0)

# Create a DataFrame to store z-scores and identify anomalies
z_score_df = pd.DataFrame(z_scores, index=pivot_df.index, columns=pivot_df.columns)

# Flag anomalies (z-score > threshold)
threshold = 3
anomalies = z_score_df[(z_score_df > threshold) | (z_score_df < -threshold)]

# Display anomalies found
anomalous_months = anomalies.dropna(how='all')
print("Anomalies detected:")
print(anomalous_months)