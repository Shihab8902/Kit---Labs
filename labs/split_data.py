"""Chunk 3.3 — split train/validation/test, check disjoint."""
import hashlib
import json
from datasets import load_dataset


def norm(t):
    return " ".join(t.split())


def h(t):
    return hashlib.md5(norm(t).encode("utf-8")).hexdigest()


def main():
    ds_train = load_dataset("roneneldan/TinyStories", split="train")
    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    n_val = len(ds_val)
    n_half = n_val // 2
    val_texts = [ds_val[i]["text"] for i in range(n_half)]
    test_texts = [ds_val[i]["text"] for i in range(n_half, n_val)]
    val_hashes = set(h(t) for t in val_texts)
    test_hashes = set(h(t) for t in test_texts)
    cross = len(val_hashes & test_hashes)
    print(f"train_rows={len(ds_train)} val_rows={len(val_texts)} test_rows={len(test_texts)}")
    print(f"val_test_overlap={cross}")

    held = val_hashes | test_hashes
    overlap_train = 0
    for i in range(len(ds_train)):
        if h(ds_train[i]["text"]) in held:
            overlap_train += 1
            if overlap_train <= 5:
                print(f"train_overlap idx={i}")
    print(f"train_held_overlap={overlap_train}")
    out = {
        "train_rows": len(ds_train),
        "val_rows": len(val_texts),
        "test_rows": len(test_texts),
        "val_test_overlap": cross,
        "train_held_overlap": overlap_train,
        "rule": "official train=train; official validation first half=validation, second half=test",
    }
    with open("data/splits.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote data/splits.json")


if __name__ == "__main__":
    main()
