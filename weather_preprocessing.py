#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified Weather Classification Dataset Preprocessing
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# Dataset path (current directory where the weather folders are located)
DATA_PATH = Path(os.getcwd())

# Output directory for splits
SPLITS_DIR = Path('./splits')
SPLITS_DIR.mkdir(exist_ok=True)

# Class mapping
CLASS_MAPPING = {
    'clear': 0,
    'rain': 1,
    'fog': 2,
    'snow': 3
}

# Splits percentages
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def scan_valid_images(data_path, class_mapping):
    """Scan the dataset and create a DataFrame with only valid images and labels."""
    image_files = []
    labels = []
    for class_name, label in class_mapping.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            continue
        extensions = ['.jpg', '.jpeg', '.png']
        for ext in extensions:
            for img_file in class_dir.glob(f'*{ext}'):
                try:
                    img = Image.open(img_file)
                    img.verify()
                    img.close()
                    rel_path = f"{class_name}/{img_file.name}"
                    image_files.append(rel_path)
                    labels.append(label)
                except Exception as e:
                    print(f"Invalid image {img_file}: {e}")
    df = pd.DataFrame({'filepath': image_files, 'label': labels})
    return df

def main():
    print("Scanning original dataset for valid images...")
    df = scan_valid_images(DATA_PATH, CLASS_MAPPING)
    print(f"Valid original images: {len(df)}")

    print("\nScanning augmented dataset for valid images...")
    aug_df = scan_valid_images(DATA_PATH / 'augmented_images', CLASS_MAPPING)
    print(f"Valid augmented images: {len(aug_df)}")

    # Split only the original images
    min_count = df['label'].value_counts().min()
    print(f"\nDownsampling original images to {min_count} per class for balance in splits.")
    balanced_df = df.groupby('label').sample(n=min_count, random_state=42).reset_index(drop=True)

    print("\nSplitting balanced original dataset...")
    train_df, temp_df = train_test_split(
        balanced_df,
        train_size=TRAIN_SPLIT,
        random_state=42,
        stratify=balanced_df['label']
    )
    relative_val_size = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=42,
        stratify=temp_df['label']
    )

    # Add augmented images ONLY to the training set
    print(f"\nAdding {len(aug_df)} augmented images to the training set only.")
    train_df = pd.concat([train_df, aug_df], ignore_index=True)

    # Downsample the final training set for perfect balance
    min_train_count = train_df['label'].value_counts().min()
    print(f"\nDownsampling final training set to {min_train_count} per class for perfect balance.")
    train_df = train_df.groupby('label').sample(n=min_train_count, random_state=42).reset_index(drop=True)

    print("\nClass distribution in splits:")
    print("Training:")
    print(train_df['label'].value_counts())
    print("Validation:")
    print(val_df['label'].value_counts())
    print("Test:")
    print(test_df['label'].value_counts())

    # Plot and save distribution
    def plot_split_distribution(df, ax, title):
        class_counts = df['label'].value_counts().sort_index()
        bars = ax.bar(range(len(CLASS_MAPPING)), class_counts.values)
        ax.set_title(title)
        ax.set_xticks(range(len(CLASS_MAPPING)))
        ax.set_xticklabels(CLASS_MAPPING.keys(), rotation=45)
        ax.set_ylabel('Number of Images')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plot_split_distribution(train_df, ax1, 'Training Set')
    plot_split_distribution(val_df, ax2, 'Validation Set')
    plot_split_distribution(test_df, ax3, 'Test Set')
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()
    print("Distribution plot saved as 'dataset_distribution.png'")

    # Save each split to CSV
    print("\nSaving splits to CSV...")
    train_df.to_csv(SPLITS_DIR / 'train.csv', index=False)
    val_df.to_csv(SPLITS_DIR / 'val.csv', index=False)
    test_df.to_csv(SPLITS_DIR / 'test.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    main() 