"""Chunk 3.2 — inspect 20 random stories, propose cleaning rules."""
import random
import re
from datasets import load_dataset


def main():
    ds = load_dataset("roneneldan/TinyStories", split="train")
    rng = random.Random(7)
    idxs = rng.sample(range(len(ds)), 20)
    email_re = re.compile(r"\S+@\S+")
    phone_re = re.compile(r"\d{3}[-. ]\d{3}[-. ]\d{4}")
    report = []
    for k, i in enumerate(idxs):
        t = ds[int(i)]["text"]
        words = len(t.split())
        non_ascii = sum(1 for c in t if ord(c) > 127)
        flags = []
        if not t.strip():
            flags.append("EMPTY")
        if len(t) < 50:
            flags.append("VERY_SHORT")
        if email_re.search(t):
            flags.append("HAS_EMAIL_LIKE")
        if phone_re.search(t):
            flags.append("HAS_PHONE_LIKE")
        if non_ascii / max(len(t), 1) > 0.05:
            flags.append("NON_ASCII_HEAVY")
        prev = t[:150].replace("\n", " ")
        print(f"--- {k + 1} idx={i} chars={len(t)} words={words} flags={flags}")
        print(f"    {prev}")
        report.append((len(t), words, flags))
    shorts = sum(1 for _, _, f in report if "VERY_SHORT" in f or "EMPTY" in f)
    priv = sum(1 for _, _, f in report if "HAS_EMAIL_LIKE" in f or "HAS_PHONE_LIKE" in f)
    print(f"inspected=20 very_short_or_empty={shorts} private_like={priv}")
    rules = """# Cleaning rules (proposed from 20-story sample)
- Drop empty stories.
- Drop stories under 50 characters.
- Normalize line endings and whitespace (keep story breaks).
- Exact-deduplicate full stories.
- Flag email-like or phone-like text for removal.
- Keep the rest; re-check after tokenizing.
"""
    with open("data/cleaning_rules.md", "w", encoding="utf-8") as f:
        f.write(rules)
    print("wrote data/cleaning_rules.md")


if __name__ == "__main__":
    main()
