# explore_data.py
import pandas as pd

# Load the dataset
df = pd.read_csv('data/news.csv')

# Show basic info
print("📊 Dataset Info:")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

print("\n📈 Label Distribution:")
print(df['label'].value_counts())

print("\n📝 Sample news text:")
print(df['text'].iloc[0][:200], "...")