import pandas as pd

# Load the dataset
df = pd.read_csv('climate_change_faqs.csv')

# Data cleaning (example)
df.drop_duplicates(subset='faq', inplace=True)  # Remove duplicates
df = df[df['faq'].notnull()]  # Drop rows with missing questions/answers

# Save the cleaned dataset
df.to_csv('climate_change_faqs_cleaned.csv', index=False)
