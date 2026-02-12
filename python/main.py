import zipfile
import os

src = "."
dst = "extracted"
os.makedirs(dst, exist_ok=True)

for fname in os.listdir(src):
    if fname.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(src, fname)) as z:
            for name in z.namelist():
                if name.endswith(".txt") and not name.startswith("_about"):
                    z.extract(name, dst)