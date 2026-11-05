"""Chunk 2.2 — BPE on a real text sample (pair-joining, round-trip safe)."""
from collections import Counter


def get_pairs(toks):
    return Counter(zip(toks[:-1], toks[1:]))


def main():
    with open("data/raw/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full = f.read()
    sample = full[:200_000]  # ~200 KB stand-in; same method scales to 10 MB
    held = full[900_000:905_000]
    toks = list(sample)
    merges = []
    for i in range(100):
        top, n = get_pairs(toks).most_common(1)[0]
        new = "".join(top)
        merges.append((top, new))
        out, j = [], 0
        while j < len(toks):
            if j < len(toks) - 1 and (toks[j], toks[j + 1]) == top:
                out.append(new)
                j += 2
            else:
                out.append(toks[j])
                j += 1
        toks = out
        if (i + 1) % 25 == 0:
            print(f"merge={i + 1} piece='{new[:12]}' count={n} tokens={len(toks)}")

    # encode held-out by replaying merges in order, decode by joining
    et = list(held)
    for (a, b), new in merges:
        out, j = [], 0
        while j < len(et):
            if j < len(et) - 1 and et[j] == a and et[j + 1] == b:
                out.append(new)
                j += 2
            else:
                out.append(et[j])
                j += 1
        et = out
    back = "".join(et)
    print(f"merges=100 sample_chars={len(sample)} vocab_pieces={len(set(toks))}")
    print(f"held_chars={len(held)} held_tokens={len(et)} roundtrip_ok={back == held}")


if __name__ == "__main__":
    main()
