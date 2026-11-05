"""Chunk 3.3b — drop val/test stories already in train (honest exam)."""
import hashlib
import json
from datasets import load_dataset


def norm(t):
    return " ".join(t.split())


def h(t):
    return hashlib.md5(norm(t).encode("utf-8")).hexdigest()


def main():
    train = load_dataset("roneneldan/TinyStories", split="train")
    val_full = load_dataset("roneneldan/TinyStories", split="validation")
    print("hashing train (2.1M)...")
    train_hashes = set()
    for i in range(len(train)):
        train_hashes.add(h(train[i]["text"]))
        if (i + 1) % 500000 == 0:
            print(f"  {i + 1}/{len(train)}")
    n_half = len(val_full) // 2
    val_raw = [val_full[i]["text"] for i in range(n_half)]
    test_raw = [val_full[i]["text"] for i in range(n_half, len(val_full))]

    def clean(lst):
        seen, out, dropped = set(), [], 0
        for t in lst:
            k = h(t)
            if k in train_hashes or k in seen:
                dropped += 1
                continue
            seen.add(k)
            out.append(t)
        return out, dropped

    val_clean, val_drop = clean(val_raw)
    test_clean, test_drop = clean(test_raw)
    print(f"val_raw={len(val_raw)} val_clean={len(val_clean)} dropped={val_drop}")
    print(f"test_raw={len(test_raw)} test_clean={len(test_clean)} dropped={test_drop}")
    with open("data/splits_clean.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_rows": len(train),
            "val_raw": len(val_raw),
            "test_raw": len(test_raw),
            "val_clean": len(val_clean),
            "test_clean": len(test_clean),
            "val_dropped": val_drop,
            "test_dropped": test_drop,
        }, f, indent=2)
    # save clean text for tokenizing next
    with open("data/val_clean.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_clean))
    with open("data/test_clean.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_clean))
    print("wrote data/splits_clean.json + val_clean.txt + test_clean.txt")


if __name__ == "__main__":
    main()
