import pandas as pd

# Load the CSV files
train_df = pd.read_csv('splits/train.csv')
val_df = pd.read_csv('splits/val.csv')
test_df = pd.read_csv('splits/test.csv')

# Extract image names
train_images = set(train_df['Image'])
val_images = set(val_df['Image'])
test_images = set(test_df['Image'])

# Check for overlaps
train_val_overlap = train_images & val_images
train_test_overlap = train_images & test_images
val_test_overlap = val_images & test_images

# Report results
print("Train-Val overlap:", len(train_val_overlap), "images")
if train_val_overlap:
    print("Overlapping images:", sorted(train_val_overlap))

print("\nTrain-Test overlap:", len(train_test_overlap), "images")
if train_test_overlap:
    print("Overlapping images:", sorted(train_test_overlap))

print("\nVal-Test overlap:", len(val_test_overlap), "images")
if val_test_overlap:
    print("Overlapping images:", sorted(val_test_overlap))

# Total unique images
total_unique = len(train_images | val_images | test_images)
print(f"\nTotal unique images: {total_unique}")
print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
