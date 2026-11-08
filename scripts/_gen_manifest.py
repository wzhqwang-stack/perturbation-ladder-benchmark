import hashlib
from pathlib import Path
PKG = Path(r"D:\资料\课题\虚拟细胞\论文②投稿包")
out = PKG / "MANIFEST.csv"
rows = ["file,sha256,bytes"]
for f in sorted(PKG.rglob("*")):
    if f.is_file() and f.name != "MANIFEST.csv":
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        rows.append(f"{f.relative_to(PKG).as_posix()},{h.hexdigest()},{f.stat().st_size}")
out.write_text("\n".join(rows), encoding="utf-8")
print(f"MANIFEST: {len(rows)-1} files")