# inspect_dataset.py
import pandas as pd
import os

print("🔍 Inspecting your Fake News Dataset")
print("="*50)

# Load both datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

print(f"\n📊 FAKE News Dataset:")
print(f"  - Shape: {fake_df.shape}")
print(f"  - Columns: {fake_df.columns.tolist()}")
print(f"  - First 2 rows:")
print(fake_df[['title', 'subject', 'date']].head(2))

print(f"\n📊 TRUE News Dataset:")
print(f"  - Shape: {true_df.shape}")
print(f"  - Columns: {true_df.columns.tolist()}")
print(f"  - First 2 rows:")
print(true_df[['title', 'subject', 'date']].head(2))

print(f"\n📈 Total Articles: {len(fake_df) + len(true_df)}")
print(f"  - FAKE: {len(fake_df)}")
print(f"  - TRUE: {len(true_df)}")