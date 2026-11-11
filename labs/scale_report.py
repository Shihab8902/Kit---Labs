"""Scaling report — params per config + best val per run; diff two runs."""
import json
import yaml
from src.kit.model import GPT, GPTConfig


def best(run):
    vals = [json.loads(l)["val_loss"] for l in open(f"runs/{run}/metrics.jsonl")]
    steps = [json.loads(l)["step"] for l in open(f"runs/{run}/metrics.jsonl")]
    return steps, vals, min(vals)


print("---best vals---")
for run in ["scale-05m", "scale-15m", "scale-5m", "scale-15m"]:
    steps, vals, b = best(run)
    print(f"{run}: best={b:.4f} final={vals[-1]:.4f}")

print("---param counts---")
for cf in ["configs/scale-05m.yaml", "configs/scale-15m.yaml", "configs/scale-5m.yaml", "configs/scale-15m.yaml"]:
    c = yaml.safe_load(open(cf))["model"]
    m = GPT(GPTConfig(c["vocab_size"], c["d_model"], c["n_layer"], c["n_head"], c["n_ctx"],
                      norm=c["norm"], pos=c["pos"], mlp=c["mlp"]))
    print(f"{cf}: params={m.count_params()}")

print("---15m vs 15m metrics identical?---")
a = open("runs/scale-15m/metrics.jsonl").read()
b = open("runs/scale-15m/metrics.jsonl").read()
print("files_identical=", a == b)
