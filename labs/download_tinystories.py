"""Chunk 3.1 — download TinyStories to D: and write the manifest."""
import datetime
import os
from datasets import load_dataset


def dir_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / (1024 ** 3)


def main():
    print("downloading roneneldan/TinyStories (about 1 GB)...")
    ds = load_dataset("roneneldan/TinyStories")
    cache = os.environ.get("HF_HOME", "?")
    lines = []
    lines.append("# Data manifest — TinyStories")
    lines.append("")
    lines.append(f"date={datetime.date.today().isoformat()}")
    lines.append("source=https://huggingface.co/datasets/roneneldan/TinyStories")
    lines.append("license=CDLA-Sharing-1.0 (verify at source before sharing weights)")
    lines.append(f"hf_cache={cache}")
    for split in ds:
        lines.append(f"split_{split}_rows={len(ds[split])}")
    try:
        lines.append(f"cache_size_gb={dir_gb(cache):.2f}")
    except Exception as e:
        lines.append(f"cache_size=unknown ({e})")
    os.makedirs("data", exist_ok=True)
    with open("data/MANIFEST.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
