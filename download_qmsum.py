from datasets import load_dataset
import os
import json

print("Downloading QMSum from HuggingFace (cleaned version)...")

# Download cleaned dataset
dataset = load_dataset("pszemraj/qmsum-cleaned")

print(f"✓ Train: {len(dataset['train'])} examples")
print(f"✓ Val: {len(dataset['validation'])} examples")
print(f"✓ Test: {len(dataset['test'])} examples")

# Save as JSONL files
os.makedirs("data/raw/qmsum", exist_ok=True)

# Save train
print("Saving train.jsonl...")
with open("data/raw/qmsum/train.jsonl", "w", encoding="utf-8") as f:
    for item in dataset['train']:
        f.write(json.dumps(item) + "\n")

# Save validation
print("Saving val.jsonl...")
with open("data/raw/qmsum/val.jsonl", "w", encoding="utf-8") as f:
    for item in dataset['validation']:
        f.write(json.dumps(item) + "\n")

# Save test
print("Saving test.jsonl...")
with open("data/raw/qmsum/test.jsonl", "w", encoding="utf-8") as f:
    for item in dataset['test']:
        f.write(json.dumps(item) + "\n")

print("\n✅ Files saved to data/raw/qmsum/")
print("✅ Ready to preprocess!")