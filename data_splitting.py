'''
This script splits the .jsonl dataset file into training, validation, and test sets.
We use an 80-10-10 split.
'''
import os
import json
import random

# Configuration
INPUT_JSON = "./dataset.jsonl"
OUTPUT_DIR = "./data_splits"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
TEST_FILE = "test.jsonl"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
with open(INPUT_JSON, "r") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data) # Shuffle data before splitting
total_samples = len(data)
train_end = int(total_samples * TRAIN_RATIO)
val_end = train_end + int(total_samples * VAL_RATIO)
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Check that splits sum to total and abort
if len(train_data) + len(val_data) + len(test_data) != total_samples:
    raise ValueError("Data split sizes do not sum to total samples!")

# Save splits
def save_split(data_split, file_name):
    with open(os.path.join(OUTPUT_DIR, file_name), "w") as f:
        for entry in data_split:
            f.write(json.dumps(entry) + "\n")

save_split(train_data, TRAIN_FILE)
save_split(val_data, VAL_FILE)
save_split(test_data, TEST_FILE)

print(f"Data splitting complete: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples saved in {OUTPUT_DIR}")
