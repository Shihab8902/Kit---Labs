"""Chunk 2.3 — production 4K tokenizer (train-only text)."""
import hashlib
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    with open("data/raw/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full = f.read()
    n = int(len(full) * 0.9)
    train_text, held = full[:n], full[n:]
    with open("data/raw/train_split.txt", "w", encoding="utf-8") as f:
        f.write(train_text)

    specials = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=4096, special_tokens=specials, show_progress=False)
    tok.train(files=["data/raw/train_split.txt"], trainer=trainer)
    tok.save("tokenizer/kit1-4k.json")
    digest = sha256_of("tokenizer/kit1-4k.json")
    with open("tokenizer/kit1-4k.sha256", "w") as f:
        f.write(digest + "\n")

    ok_special = all(len(tok.encode(s).ids) == 1 for s in specials)
    lines = [ln for ln in held.splitlines() if ln.strip()][:20]
    ok_rt = all(tok.decode(tok.encode(ln).ids) == ln for ln in lines)
    print(f"vocab_size={tok.get_vocab_size()} sha256={digest[:12]}...")
    print(f"special_single_ids={ok_special} roundtrip_20={ok_rt}")


if __name__ == "__main__":
    main()
