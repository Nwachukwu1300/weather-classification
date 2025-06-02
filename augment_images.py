import os
from PIL import Image, ImageEnhance
import pandas as pd
import random
from pathlib import Path

# Set up paths using workspace base directory
BASE_DIR = Path(os.getcwd())
SPLITS_DIR = BASE_DIR / 'splits'
TRAIN_CSV = SPLITS_DIR / 'train.csv'
OUTPUT_CSV = BASE_DIR / 'train_augmented.csv'

# Create augmented directory
AUGMENTED_DIR = BASE_DIR / 'augmented_images'
AUGMENTED_DIR.mkdir(exist_ok=True)

print(f"Loading dataset from {TRAIN_CSV}")
# Load the original training CSV file
df = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(df)} images from training set")

# Function to shift hue
def change_hue(img, delta=0.05):
    img = img.convert("RGB")
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda p: int((p + delta * 255) % 255))
    return Image.merge("HSV", (h, s, v)).convert("RGB")

# Do augmentations
augmented_rows = []
counter = 0

# Focus on underrepresented classes (assuming class imbalance)
# Change this filter as needed for your specific case
target_labels = [1, 3]  # rain and snow (assuming they're underrepresented)
subset_df = df[df["label"].isin(target_labels)]
print(f"Selected {len(subset_df)} images for augmentation (classes: {target_labels})")

for _, row in subset_df.iterrows():
    filepath = row["filepath"]
    label = row["label"]
    
    # Get the original image path (relative to workspace)
    full_path = BASE_DIR / filepath
    filename = os.path.basename(filepath)
    base = os.path.splitext(filename)[0]
    
    # Create class directory in augmented folder if it doesn't exist
    class_name = ["clear", "rain", "fog", "snow"][label]
    class_dir = AUGMENTED_DIR / class_name
    class_dir.mkdir(exist_ok=True)

    try:
        img = Image.open(full_path)

        # 1. Flip
        flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_name = f"{base}_flip.jpg"
        flip_path = class_dir / flip_name
        flip.save(flip_path)
        augmented_rows.append((str(Path('augmented_images') / class_name / flip_name), label))

        # 2. Brightness
        enhancer = ImageEnhance.Brightness(img)
        bright = enhancer.enhance(random.uniform(0.5, 1.6))  # for darker shadows and brighter highlights
        bright_name = f"{base}_bright.jpg"
        bright_path = class_dir / bright_name
        bright.save(bright_path)
        augmented_rows.append((str(Path('augmented_images') / class_name / bright_name), label))

        # 3. Hue
        delta = random.uniform(-0.15, 0.15)
        hue = change_hue(img, delta=delta)
        hue_name = f"{base}_hue.jpg"
        hue_path = class_dir / hue_name
        hue.save(hue_path)
        augmented_rows.append((str(Path('augmented_images') / class_name / hue_name), label))

        counter += 1
        if counter % 20 == 0:
            print(f"Augmented {counter} images so far...")

    except Exception as e:
        print(f"Error with {filepath}: {e}")

# Create DataFrame for augmented images
aug_df = pd.DataFrame(augmented_rows, columns=["filepath", "label"])
print(f"Created {len(aug_df)} augmented images")

# Combine original and augmented data
final_df = pd.concat([df, aug_df])
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone! Saved augmented dataset with {len(final_df)} images to: {OUTPUT_CSV}")
