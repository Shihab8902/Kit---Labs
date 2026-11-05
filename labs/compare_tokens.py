"""Chunk 2.4 — tokens per word: chars vs BPE pieces."""
from tokenizers import Tokenizer


def main():
    tok = Tokenizer.from_file("tokenizer/kit1-4k.json")
    with open("data/raw/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full = f.read()
    held = full[int(len(full) * 0.9):]
    lines = [ln for ln in held.splitlines() if ln.strip()][:1000]
    cw, bw, words = 0, 0, 0
    for ln in lines:
        w = len(ln.split())
        words += w
        cw += len(ln)  # chars = 1 token each in char world
        bw += len(tok.encode(ln).ids)
    print(f"lines={len(lines)} words={words}")
    print(f"chars_per_word={cw / words:.3f}")
    print(f"bpe_tokens_per_word={bw / words:.3f}")
    print(f"compression={cw / bw:.2f}x")


if __name__ == "__main__":
    main()
