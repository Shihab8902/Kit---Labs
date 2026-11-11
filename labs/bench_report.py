import json
for l in open("runs/bench100/metrics.jsonl"):
    d = json.loads(l)
    print(f"step={d['step']} lr={d['lr']:.6f} val={d['val_loss']:.3f}")
flops = 6 * 1344640 * 50_000_000
est = 50_000_000 / 162008
print(f"flops={flops:.2e} eff_tflops={flops / 271 / 1e12:.2f}")
print(f"est_50M_s={est:.0f} actual=271 err={abs(est - 271) / 271 * 100:.1f}%")
