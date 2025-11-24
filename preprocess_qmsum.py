import json
from datasets import Dataset, DatasetDict
import os

# Paths
RAW_DIR = f"data\raw\qmsum"
OUT_DIR = "data/processed/qmsum"
os.makedirs(OUT_DIR, exist_ok=True)

def load_file(path):
    """Load JSONL file and extract query-answer pairs as dialogue-summary"""
    print(f"Loading: {path}")
    records = []
    
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path}")
        return records
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                
                # Extract input (query/transcript context)
                dialogue = item.get("input", "").strip()
                
                # Extract output (answer/summary)
                summary = item.get("output", "").strip()
                
                # Only add if both exist and are non-empty
                if dialogue and summary:
                    records.append({
                        "id": item.get("id", f"record_{line_num}"),
                        "dialogue": dialogue,
                        "summary": summary
                    })
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                print(f"⚠️  Line {line_num}: Error - {e}")
    
    print(f"→ {len(records)} records loaded")
    return records

# Load all splits
print("\n" + "="*50)
print("QMSum Dataset Preprocessing")
print("="*50 + "\n")

train_data = load_file(f"{RAW_DIR}/train.jsonl")
val_data = load_file(f"{RAW_DIR}/val.jsonl")
test_data = load_file(f"{RAW_DIR}/test.jsonl")

# Create datasets
if not train_data and not val_data and not test_data:
    print("\n❌ ERROR: No data loaded from any split!")
    print("\nPlease ensure you have downloaded the files correctly.")
    print("Run: python download_qmsum.py")
    exit(1)

dataset_dict = {
    "train": Dataset.from_list(train_data) if train_data else Dataset.from_dict({"id": [], "dialogue": [], "summary": []}),
    "validation": Dataset.from_list(val_data) if val_data else Dataset.from_dict({"id": [], "dialogue": [], "summary": []}),
    "test": Dataset.from_list(test_data) if test_data else Dataset.from_dict({"id": [], "dialogue": [], "summary": []}),
}

dataset = DatasetDict(dataset_dict)

# Print dataset info
print("\n" + "="*50)
print("Dataset Summary:")
print("="*50)
print(dataset)

# Show sample
if len(train_data) > 0:
    print("\n" + "="*50)
    print("SAMPLE RECORD:")
    print("="*50)
    print(f"ID: {train_data[0]['id']}")
    print(f"Dialogue (first 200 chars): {train_data[0]['dialogue'][:200]}...")
    print(f"Summary (first 200 chars): {train_data[0]['summary'][:200]}...")

# Save to disk
print(f"\n📁 Saving to: {OUT_DIR}")
dataset.save_to_disk(OUT_DIR)

print("\n✅ QMSum processed and saved successfully!")
print(f"📊 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
print(f"📂 Location: {OUT_DIR}")