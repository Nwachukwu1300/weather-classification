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

def scan_dataset(data_path, class_mapping):
    """Scan the dataset and create a DataFrame with filepaths and labels."""
    
    image_files = []
    labels = []
    
    # Base directory path as string for removing from absolute paths
    base_dir_str = str(data_path)
    
    # Iterate through each class folder
    for class_name, label in class_mapping.items():
        class_dir = data_path / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            continue
            
        # Get all image files in the class directory
        extensions = ['.jpg', '.jpeg', '.png']
        
        for ext in extensions:
            for img_file in class_dir.glob(f'*{ext}'):
                # Get absolute path and convert to relative path
                abs_path = str(img_file.absolute())
                # Create relative path by keeping just the class name and filename
                rel_path = f"{class_name}/{img_file.name}"
                
                image_files.append(rel_path)
                labels.append(label)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'filepath': image_files,
        'label': labels
    })
    
    return df

def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Split the dataset into train, validation, and test sets."""
    
    # First split: separate train from the rest
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    # Second split: divide the remaining data into validation and test
    relative_val_size = val_size / (val_size + test_size)
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=relative_val_size, 
        random_state=random_state, 
        stratify=temp_df['label']
    )
    
    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, val_df, test_df

def plot_distribution(train_df, val_df, test_df, class_mapping):
    """Plot the distribution of classes across splits."""
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Function to plot distribution for a single split
    def plot_split_distribution(df, ax, title):
        class_counts = df['label'].value_counts().sort_index()
        bars = ax.bar(range(len(class_mapping)), class_counts.values)
        ax.set_title(title)
        ax.set_xticks(range(len(class_mapping)))
        ax.set_xticklabels(class_mapping.keys(), rotation=45)
        ax.set_ylabel('Number of Images')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
    
    # Plot distributions
    plot_split_distribution(train_df, ax1, 'Training Set')
    plot_split_distribution(val_df, ax2, 'Validation Set')
    plot_split_distribution(test_df, ax3, 'Test Set')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    plt.close()

def main():
    # First, scan the original dataset
    print("Scanning original dataset...")
    df = scan_dataset(DATA_PATH, CLASS_MAPPING)
    
    # Then, scan the augmented dataset
    print("\nScanning augmented dataset...")
    aug_df = scan_dataset(DATA_PATH / 'augmented_images', CLASS_MAPPING)
    
    # Combine original and augmented datasets
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    # Print total counts
    print(f"\nOriginal images: {len(df)}")
    print(f"Augmented images: {len(aug_df)}")
    print(f"Total images: {len(combined_df)}")
    
    # Split the combined dataset
    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(
        combined_df, 
        train_size=TRAIN_SPLIT, 
        val_size=VAL_SPLIT, 
        test_size=TEST_SPLIT
    )
    
    # Print split sizes
    print(f"\nTraining: {len(train_df)} images ({100*len(train_df)/len(combined_df):.1f}%)")
    print(f"Validation: {len(val_df)} images ({100*len(val_df)/len(combined_df):.1f}%)")
    print(f"Test: {len(test_df)} images ({100*len(test_df)/len(combined_df):.1f}%)")
    
    # Plot and save distribution
    print("\nPlotting dataset distribution...")
    plot_distribution(train_df, val_df, test_df, CLASS_MAPPING)
    print("Distribution plot saved as 'dataset_distribution.png'")
    
    # Save each split to CSV
    print("\nSaving splits to CSV...")
    train_df.to_csv(SPLITS_DIR / 'train.csv', index=False)
    val_df.to_csv(SPLITS_DIR / 'val.csv', index=False)
    test_df.to_csv(SPLITS_DIR / 'test.csv', index=False)
    
    print("Done!")

if __name__ == "__main__":
    main() 