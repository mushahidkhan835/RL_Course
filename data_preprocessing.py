import os
import json
from PIL import Image
from tqdm import tqdm


# Configuration
IMAGE_DIR = "./circuit_images"
CODE_DIR = "./circuit_ops"
OUTPUT_JSON = "./dataset.jsonl"
RESIZED_DIR = "./circuit_images_resized"
IMAGE_SIZE = (224, 224)  # Resize images to 256x256
MAX_SAMPLES = None  # Set to None to process all samples

# Create new output dir if not exists
os.makedirs(RESIZED_DIR, exist_ok=True)

# Function to clean code text
def clean_code_text(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip()] # strip lines and remove empty ones
    return "\n".join(lines)

# Process dataset
entries = []
for i, file_name in enumerate(tqdm(sorted(os.listdir(IMAGE_DIR)))):
    if not file_name.endswith(".png"):
        continue
    if MAX_SAMPLES and i >= MAX_SAMPLES: # stop if reached max samples
        break

    idx = os.path.splitext(file_name)[0].replace("data", "")
    img_path = os.path.join(IMAGE_DIR, file_name)
    txt_path = os.path.join(CODE_DIR, f"data{idx}.txt")

    if not os.path.exists(txt_path):
        print(f"Warning: Missing text file for {file_name}, skipping.")
        continue

    # Resize and normalize image
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    
    resized_path = os.path.join(RESIZED_DIR, file_name)
    img.save(resized_path) 

    # Read code
    code_text = clean_code_text(txt_path)

    entries.append({
        "image": resized_path,
        "code": code_text
    })

# Save as JSONL
with open(OUTPUT_JSON, "w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"Preprocessing complete: {len(entries)} samples saved to {OUTPUT_JSON}")
