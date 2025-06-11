#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# Load the CSV files from splits directory
splits_dir = Path('./splits')
train_df = pd.read_csv(splits_dir / 'train.csv')
val_df = pd.read_csv(splits_dir / 'val_fixed.csv')
test_df = pd.read_csv(splits_dir / 'test_fixed.csv')

# Class mapping
class_names = {0: 'clear', 1: 'rain', 2: 'fog', 3: 'snow'}

# Simple text display of class distribution
print("TRAIN SET:")
for label, name in class_names.items():
    count = len(train_df[train_df['label'] == label])
    print(f"  {name}: {count}")

print("\nVALIDATION SET:")
for label, name in class_names.items():
    count = len(val_df[val_df['label'] == label])
    print(f"  {name}: {count}")

print("\nTEST SET:")
for label, name in class_names.items():
    count = len(test_df[test_df['label'] == label])
    print(f"  {name}: {count}")

# Total counts
print(f"\nTotal train images: {len(train_df)}")
print(f"Total validation images: {len(val_df)}")
print(f"Total test images: {len(test_df)}")
print(f"Total dataset size: {len(train_df) + len(val_df) + len(test_df)}") 