"""Chunk 6.4a — train the 8K tokenizer on TinyStories train text."""
import hashlib
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


def main():
    train = load_dataset("roneneldan/TinyStories", split="train")
    with open("data/tok2_train.txt", "w", encoding="utf-8") as f:
        for i in range(200_000):
            f.write(train[i]["text"] + "\n")
    print("wrote sample (200k stories)", flush=True)
    specials = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.train(files=["data/tok2_train.txt"],
              trainer=BpeTrainer(vocab_size=8192, special_tokens=specials, show_progress=False))
    tok.save("tokenizer/kit2-8k.json")
    h = hashlib.sha256(open("tokenizer/kit2-8k.json", "rb").read()).hexdigest()
    open("tokenizer/kit2-8k.sha256", "w").write(h + "\n")
    ok_s = all(len(tok.encode(s).ids) == 1 for s in specials)
    print(f"vocab={tok.get_vocab_size()} sha={h[:12]}... specials_single={ok_s}", flush=True)


if __name__ == "__main__":
    main()
