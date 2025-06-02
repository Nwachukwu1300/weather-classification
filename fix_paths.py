import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil

def validate_and_fix_paths(csv_path, output_csv_path):
    """
    Validate image paths in CSV and create a new CSV with only valid images
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    print(f"Original number of images: {len(df)}")
    
    # Convert paths to absolute
    base_dir = Path(os.getcwd())
    df['filepath'] = df['filepath'].apply(lambda x: str(base_dir / x))
    
    # Validate images
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            # Try to open and verify the image
            img = Image.open(row['filepath'])
            img.verify()  # Verify it's an image
            img.close()
            
            # If we get here, the image is valid
            valid_rows.append(row)
        except Exception as e:
            print(f"Invalid image {row['filepath']}: {str(e)}")
    
    # Create new DataFrame with only valid images
    valid_df = pd.DataFrame(valid_rows)
    print(f"Number of valid images: {len(valid_df)}")
    
    # Convert back to relative paths for the CSV
    valid_df['filepath'] = valid_df['filepath'].apply(lambda x: str(Path(x).relative_to(base_dir)))
    
    # Save the new CSV
    valid_df.to_csv(output_csv_path, index=False)
    print(f"Saved valid paths to {output_csv_path}")

if __name__ == "__main__":
    # Fix paths for all CSV files
    base_dir = Path(os.getcwd())
    
    # Process train_augmented.csv
    train_csv = base_dir / 'train_augmented.csv'
    train_fixed = base_dir / 'train_augmented_fixed.csv'
    validate_and_fix_paths(train_csv, train_fixed)
    
    # Process validation and test CSVs
    splits_dir = base_dir / 'splits'
    for csv_name in ['val.csv', 'test.csv']:
        csv_path = splits_dir / csv_name
        fixed_path = splits_dir / f"{csv_name.replace('.csv', '_fixed.csv')}"
        validate_and_fix_paths(csv_path, fixed_path) 