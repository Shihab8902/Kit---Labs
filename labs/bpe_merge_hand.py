"""Chunk 2.1 — one BPE merge by hand."""
from collections import Counter


def main():
    words = ["low", "lowest", "lower", "low", "low"]
    # start as letters + end marker
    toks = [list(w) + ["</w>"] for w in words]
    pairs = Counter()
    for t in toks:
        for a, b in zip(t[:-1], t[1:]):
            pairs[(a, b)] += 1
    print(f"pair_counts={dict(pairs)}")
    top, n = pairs.most_common(1)[0]
    print(f"top_pair={top} count={n}")
    merged = "".join(top)
    out = []
    for t in toks:
        cur, i = [], 0
        while i < len(t):
            if i < len(t) - 1 and (t[i], t[i + 1]) == top:
                cur.append(merged)
                i += 2
            else:
                cur.append(t[i])
                i += 1
        out.append(cur)
    print(f"merged_token={merged}")
    print(f"after={out}")


if __name__ == "__main__":
    main()
