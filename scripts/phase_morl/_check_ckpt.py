"""Quick checkpoint inspection script."""
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))
if "optimizer_state_dict" in ckpt:
    opt = ckpt["optimizer_state_dict"]
    n = len(opt.get("state", {}))
    print(f"Optimizer: {n} param states")
    k = list(opt["state"].keys())[0]
    s = opt["state"][k]
    print(f"Adam keys: {list(s.keys())}")
    print(f"Adam step: {s.get('step','N/A')}")
    print(f"exp_avg norm: {s['exp_avg'].norm().item():.6f}")
    print(f"exp_avg_sq norm: {s['exp_avg_sq'].norm().item():.6f}")
else:
    print("NO optimizer_state_dict!")
print(f"Saved iter: {ckpt.get('iter','N/A')}")
