import os
import glob
import random
import csv

BASE = os.path.dirname(os.path.abspath(__file__))

CLEAN_DIR = os.path.join(BASE, "..", "clean_data")
DATASETS_DIR = os.path.join(BASE, "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def ok_pair(src, tgt):
    src, tgt = src.strip(), tgt.strip()
    if not src or not tgt:
        return False
    if len(src) < 2 or len(tgt) < 2:
        return False
    if len(src) > 250 or len(tgt) > 250:
        return False
    return True

pairs = []

for path in glob.glob(os.path.join(CLEAN_DIR, "*.txt")):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            src, tgt = parts[0], parts[1]
            if ok_pair(src, tgt):
                pairs.append((src.strip(), tgt.strip()))

print(f"✅ Loaded {len(pairs)} pairs from {CLEAN_DIR}")

random.seed(42)
random.shuffle(pairs)

n = len(pairs)
n_train = int(0.90 * n)
n_valid = int(0.05 * n)

train = pairs[:n_train]
valid = pairs[n_train:n_train + n_valid]
test  = pairs[n_train + n_valid:]

def write_csv(name, data):
    out_path = os.path.join(DATASETS_DIR, f"{name}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(data)
    print(f"✅ Wrote {out_path} ({len(data)} rows)")

write_csv("train", train)
write_csv("valid", valid)
write_csv("test", test)
