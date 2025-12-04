import csv
import os

train_file = "hover_experiment/hover_fewshot.csv"
test_file = "hover_experiment/hover_fewshot_test.csv"
clean_test_file = "hover_experiment/hover_fewshot_test_clean.csv"

def get_ids(filename):
    ids = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_key = next((k for k in row.keys() if k and 'id' in k), 'id')
                if row.get(id_key):
                    ids.add(row[id_key])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return ids

print(f"Reading training IDs from {train_file}...")
train_ids = get_ids(train_file)
print(f"Found {len(train_ids)} training IDs.")

print(f"Filtering {test_file}...")
kept = 0
removed = 0

with open(test_file, 'r', encoding='utf-8') as fin, \
     open(clean_test_file, 'w', newline='', encoding='utf-8') as fout:
    
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for row in reader:
        id_key = next((k for k in row.keys() if k and 'id' in k), 'id')
        if row.get(id_key) in train_ids:
            removed += 1
        else:
            writer.writerow(row)
            kept += 1

print(f"Done.")
print(f"Removed: {removed}")
print(f"Kept: {kept}")
print(f"Clean test set saved to: {clean_test_file}")
