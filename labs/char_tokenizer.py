"""Chunk 1.1 — character tokenizer: text <-> numbers."""


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return chars, stoi


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids: list[int], chars: list[str]) -> str:
    return "".join(chars[i] for i in ids)


if __name__ == "__main__":
    sample = "Hello, Kit! Once upon a time."
    chars, stoi = build_vocab(sample)
    ids = encode(sample, stoi)
    back = decode(ids, chars)
    print(f"vocab_size={len(chars)} ids={ids}")
    print(f"roundtrip_ok={back == sample}")
