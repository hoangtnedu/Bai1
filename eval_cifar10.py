
# eval_cifar10.py
# Evaluate a trained checkpoint on CIFAR-10 test set and export confusion matrix & metrics.
import os, argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from models_simple import SimpleCNN

def build_model(name: str, num_classes=10):
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
        return m
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif name == "vgg11_bn":
        m = models.vgg11_bn(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--model", type=str, required=True,
                    choices=["simplecnn","resnet18","mobilenetv2","efficientnet_b0","vgg11_bn"])
    ap.add_argument("--outdir", type=str, default="runs/eval")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tfm)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)

    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro")
    print(f"Test Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
    print(classification_report(y, p, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y, p)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (CIFAR-10)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    cm_path = os.path.join(args.outdir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {cm_path}")

    # Params and FLOPs (optional)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    try:
        from thop import profile
        x = torch.randn(1,3,32,32).to(device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        flops_g = flops/1e9
    except Exception:
        flops_g = None

    # Latency
    def measure(m, dev, iters=200, warmup=50):
        x = torch.randn(1,3,32,32).to(dev)
        for _ in range(warmup): _ = m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters): _ = m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        return (time.time() - t0)*1000/iters

    lat_cpu = None; lat_gpu = None
    m_cpu = build_model(args.model, num_classes=10).to("cpu")
    m_cpu.load_state_dict(ckpt["model"], strict=False)
    lat_cpu = measure(m_cpu, torch.device("cpu"))
    if torch.cuda.is_available():
        lat_gpu = measure(model, device)

    # Save JSON summary
    summary = {
        "model": args.model,
        "dataset": "cifar10",
        "test_acc": round(acc,4),
        "test_f1": round(f1m,4),
        "params_m": round(params_m,4),
        "flops_g": round(flops_g,4) if flops_g is not None else None,
        "latency_cpu_ms": round(lat_cpu,4) if lat_cpu is not None else None,
        "latency_gpu_ms": round(lat_gpu,4) if lat_gpu is not None else None,
        "checkpoint": args.checkpoint
    }
    import json
    with open(os.path.join(args.outdir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved eval summary to eval_summary.json")

if __name__ == "__main__":
    main()
