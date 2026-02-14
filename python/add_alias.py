import os

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE, "..", "clean_data")

for filename in os.listdir(CLEAN_DIR):
    if not filename.endswith(".txt"):
        continue

    lang = os.path.splitext(filename)[0]
    tag = f">>{lang}<< "

    path = os.path.join(CLEAN_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            continue

        src, tgt = parts
        new_line = f"{tag}{src.strip()}\t{tgt.strip()}\n"
        new_lines.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"✅ Updated {filename}")
