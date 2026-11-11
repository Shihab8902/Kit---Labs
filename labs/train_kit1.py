"""Kit-1 trainer: YAML config, full checkpoints (model+opt+scaler+RNG), resume."""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.kit.model import GPT, GPTConfig


def get_batch(data, batch_size, block_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = torch.tensor(np.stack([data[i:i + block_size] for i in ix]), dtype=torch.long)
    y = torch.tensor(np.stack([data[i + 1:i + block_size + 1] for i in ix]), dtype=torch.long)
    return x, y


def lr_at(step, warmup, total, peak, end):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * min(p, 1.0)))


def ckpt_data(m, opt, scaler, rng, step, val_loss):
    return {
        "model": m.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "val_loss": val_loss, "rng_np": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_py": random.getstate(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kit1.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run", default="runs/kit1-2026-11-11-a")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--peak-lr", type=float, default=None)
    ap.add_argument("--end-lr", type=float, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    total = args.steps or cfg["train"]["steps"]
    peak = args.peak_lr or cfg["train"]["peak_lr"]
    end = args.end_lr or cfg["train"]["end_lr"]
    warmup, wd, clip = cfg["train"]["warmup"], cfg["train"]["weight_decay"], cfg["train"]["grad_clip"]
    micro, accum, block = cfg["batch"]["micro"], cfg["batch"]["accum"], cfg["batch"]["block"]
    seed = cfg["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.fp32

    train = np.memmap(cfg["data"]["train"], dtype=np.uint16, mode="r")
    val = np.memmap(cfg["data"]["val"], dtype=np.uint16, mode="r")
    mc = cfg["model"]
    m = GPT(GPTConfig(mc["vocab_size"], mc["d_model"], mc["n_layer"], mc["n_head"], mc["n_ctx"],
                      norm=mc.get("norm", "layernorm"), pos=mc.get("pos", "learned"),
                      mlp=mc.get("mlp", "gelu"))).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=peak, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    start, best = 0, float("inf")

    os.makedirs(args.run + "/checkpoints", exist_ok=True)
    with open(args.run + "/config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    with open(args.run + "/env.json", "w") as f:
        json.dump({"torch": torch.__version__, "cuda": torch.version.cuda, "device": device}, f, indent=2)

    if args.resume:
        c = torch.load(args.resume, weights_only=False)
        m.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        scaler.load_state_dict(c["scaler"])
        rng.bit_generator.state = c["rng_np"]
        torch.set_rng_state(c["rng_torch"])
        torch.cuda.set_rng_state_all(c["rng_cuda"])
        random.setstate(c["rng_py"])
        start, best = c["step"], c["val_loss"]
        print(f"resumed from step {start}", flush=True)

    log = open(args.run + "/metrics.jsonl", "a")
    V = mc["vocab_size"]
    t0 = time.time()
    for s in range(start, total):
        m.train()
        opt.zero_grad()
        for _ in range(accum):
            x, y = get_batch(train, micro, block, rng)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(m(x)[:, :-1].reshape(-1, V), y[:, :-1].reshape(-1)) / accum
            scaler.scale(loss).backward()
        for g in m.parameters():
            if g.grad is not None:
                g.grad.clamp_(-clip, clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(s, warmup, total, peak, end)
        scaler.step(opt)
        scaler.update()

        if (s + 1) % 50 == 0 or s == 0 or (s + 1) == total:
            m.eval()
            with torch.no_grad():
                vl = 0.0
                for _ in range(10):
                    x, y = get_batch(val, micro, block, rng)
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        vl += F.cross_entropy(m(x)[:, :-1].reshape(-1, V), y[:, :-1].reshape(-1)).item()
                vl /= 10
            dt = time.time() - t0
            tps = (s + 1) * micro * accum * block / dt
            print(f"step={s + 1}/{total} val_loss={vl:.4f} tok_s={tps:.0f} elapsed={dt:.0f}s", flush=True)
            log.write(json.dumps({"step": s + 1, "val_loss": vl, "tok_s": tps, "lr": opt.param_groups[0]["lr"]}) + "\n")
            log.flush()
            ck = ckpt_data(m, opt, scaler, rng, s + 1, vl)
            torch.save(ck, args.run + "/checkpoints/last.pt")
            if vl < best:
                best = vl
                torch.save(ck, args.run + "/checkpoints/best.pt")
    print(f"done best_val={best:.4f} peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f} amp={use_amp}", flush=True)


if __name__ == "__main__":
    main()
