import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "..", "raw_data")
CLEAN_DIR = os.path.join(BASE_DIR, "..", "clean_data")

os.makedirs(CLEAN_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".txt"):
        continue

    src_path = os.path.join(RAW_DIR, filename)
    dst_path = os.path.join(CLEAN_DIR, filename)

    with open(src_path, "r", encoding="utf-8") as src, \
         open(dst_path, "w", encoding="utf-8") as dst:

        for line in src:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                src_sentence = parts[0].strip()
                tgt_sentence = parts[1].strip()
                dst.write(f"{src_sentence}\t{tgt_sentence}\n")

    print(f"✔ Nettoyé : {filename}")