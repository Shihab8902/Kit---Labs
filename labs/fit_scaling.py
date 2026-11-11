"""Chunk 6.9 — fit L(N) = a*N^-alpha + c through the four points."""
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.kit.model import GPT, GPTConfig

pairs = [("scale-05m", "configs/scale-05m.yaml"), ("scale-1p5m", "configs/scale-1p5m.yaml"),
         ("scale-5m", "configs/scale-5m.yaml"), ("scale-15m", "configs/scale-15m.yaml")]
Ns, Ls = [], []
for run, cf in pairs:
    c = yaml.safe_load(open(cf))["model"]
    m = GPT(GPTConfig(c["vocab_size"], c["d_model"], c["n_layer"], c["n_head"], c["n_ctx"],
                      norm=c["norm"], pos=c["pos"], mlp=c["mlp"]))
    vals = [float(json.loads(line)["val_loss"]) for line in open(f"runs/{run}/metrics.jsonl")]
    Ns.append(float(m.count_params()))
    Ls.append(float(min(vals)))
    print(f"{run}: N={Ns[-1]:.0f} best_val={Ls[-1]:.4f}")
Nv = np.array(Ns)
Lv = np.array(Ls)
best_ss, best_a, best_al, best_c = -1.0, 0.0, 0.0, 0.0
for cc in np.linspace(1.5, 2.3, 81):
    yy = np.log(Lv - cc)
    xx = np.log(Nv)
    A = np.vstack([xx, np.ones_like(xx)]).T
    res = np.linalg.lstsq(A, yy, rcond=None)
    slope = float(res[0][0])
    icept = float(res[0][1])
    pred = np.exp(icept) * Nv ** slope
    denom = float(((Lv - Lv.mean()) ** 2).sum())
    sscore = float(1 - ((Lv - (pred + cc)) ** 2).sum() / denom)
    if sscore > best_ss:
        best_ss, best_a, best_al, best_c = sscore, float(np.exp(icept)), float(-slope), float(cc)
print(f"L(N) = {best_a:.3f} * N^(-{best_al:.3f}) + {best_c:.3f}, R2={best_ss:.4f}")
xs = np.logspace(np.log10(Nv.min()), np.log10(Nv.max()), 50)
plt.figure()
plt.loglog(Nv, Lv, "o", label="runs (20M tokens each)")
plt.loglog(xs, best_a * xs ** (-best_al) + best_c, label="power-law fit")
plt.xlabel("parameters")
plt.ylabel("val loss")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("runs/scaling.png")
print("saved runs/scaling.png")
