import csv
import sys
import os

def get_ids(filename):
    ids = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle potential BOM or whitespace in keys
                id_key = next((k for k in row.keys() if k and 'id' in k), 'id')
                if row.get(id_key):
                    ids.add(row[id_key])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return ids

# Use absolute paths or relative to CWD
train_file = os.path.abspath("hover_experiment/original_hover_fewshot.csv")
test_file = os.path.abspath("hover_experiment/original_hover_fewshot_test.csv")

print(f"Checking files:\n{train_file}\n{test_file}")

train_ids = get_ids(train_file)
test_ids = get_ids(test_file)

print(f"Train IDs: {len(train_ids)}")
print(f"Test IDs: {len(test_ids)}")

overlap = train_ids.intersection(test_ids)
print(f"Overlap count: {len(overlap)}")

if overlap:
    print("WARNING: Data leakage detected!")
    print(f"First 10 overlapping IDs: {list(overlap)[:10]}")
else:
    print("No overlap detected.")
