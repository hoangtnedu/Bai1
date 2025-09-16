
# train_cifar10.py
# Training multiple CNN architectures on CIFAR-10 with a unified, fair recipe.
import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from models_simple import SimpleCNN

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def cifar10_loaders(data_root, batch_size=128, val_ratio=0.1, seed=42, aug=True, num_workers=4):
    # CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = []
    if aug:
        train_tf += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    test_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.Compose(train_tf))
    test_set   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.Compose(test_tf))

    # Split train/val
    g = torch.Generator().manual_seed(seed)
    n = len(train_full); n_val = int(n * val_ratio); n_train = n - n_val
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

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

def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_optimizer(model, name, lr, wd, momentum=0.9):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError("opt must be sgd or adamw")

def get_scheduler(optimizer, name, epochs, step_size=30, gamma=0.1):
    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    elif name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return None

def train_one_epoch(model, loader, device, optimizer, scaler=None):
    model.train(); total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward(); optimizer.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.append(y.cpu()); all_p.append(pred.cpu())
    y_true = torch.cat(all_y).numpy(); y_pred = torch.cat(all_p).numpy()
    f1 = f1_score(y_true, y_pred, average="macro")
    return loss_sum/total, correct/total, f1

def measure_latency(model, device, input_size=(1,3,32,32), warmup=50, iters=200):
    x = torch.randn(*input_size).to(device)
    # Warmup
    for _ in range(warmup):
        _ = model(x)
    # Measure
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dt = (time.time() - t0) * 1000.0 / iters  # ms/iter
    return dt

def maybe_flops(model, device):
    try:
        from thop import profile
        x = torch.randn(1,3,32,32).to(device)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops/1e9  # GMac approx
    except Exception as e:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="runs/exp_cifar10")
    ap.add_argument("--model", type=str, default="simplecnn",
                    choices=["simplecnn","resnet18","mobilenetv2","efficientnet_b0","vgg11_bn"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd","adamw"])
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--sched", type=str, default="cosine", choices=["cosine","step","none"])
    ap.add_argument("--step_size", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = cifar10_loaders(
        args.data_root, batch_size=args.batch, val_ratio=args.val_ratio, seed=args.seed,
        aug=(not args.no_aug), num_workers=args.num_workers
    )

    model = build_model(args.model, num_classes=10).to(device)
    opt = get_optimizer(model, args.opt, args.lr, args.wd)
    sched = None if args.sched == "none" else get_scheduler(opt, args.sched, args.epochs, args.step_size, args.gamma)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    hparams = vars(args).copy()
    hparams["device"] = str(device)
    hparams["params_m"] = round(count_params_m(model), 4)
    with open(os.path.join(args.outdir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    log_path = os.path.join(args.outdir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1,lr\n")

    best_acc, best_ep = -1.0, -1
    for ep in range(1, args.epochs+1):
        tl, ta = train_one_epoch(model, train_loader, device, opt, scaler=scaler)
        vl, va, vf1 = evaluate(model, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        if sched is not None: sched.step()

        with open(log_path, "a") as f:
            f.write(f"{ep},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f},{vf1:.6f},{lr_now:.6f}\n")

        if va > best_acc:
            best_acc, best_ep = va, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "val_acc": va},
                       os.path.join(args.outdir, "best.pt"))

        print(f"[{ep:03d}/{args.epochs}] train_acc={ta:.4f} val_acc={va:.4f} val_f1={vf1:.4f} lr={lr_now:.5f}")

    # Final test eval using best
    ckpt = torch.load(os.path.join(args.outdir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    tl, ta, tf1 = evaluate(model, test_loader, device)

    # Latency (CPU & GPU if available)
    model.eval()
    lat_cpu = None; lat_gpu = None
    lat_dev = torch.device("cpu")
    model_cpu = build_model(args.model, num_classes=10).to(lat_dev)
    model_cpu.load_state_dict(ckpt["model"], strict=False)
    lat_cpu = measure_latency(model_cpu, lat_dev, input_size=(1,3,32,32))

    if torch.cuda.is_available():
        lat_gpu = measure_latency(model, device, input_size=(1,3,32,32))

    # FLOPs (optional)
    flops_g = maybe_flops(model, device)

    summary = {
        "model": args.model,
        "dataset": "cifar10",
        "epochs": args.epochs,
        "batch": args.batch,
        "opt": args.opt,
        "lr0": args.lr,
        "wd": args.wd,
        "sched": args.sched,
        "aug": int(not args.no_aug),
        "seed": args.seed,
        "val_best_acc": round(best_acc, 4),
        "test_acc": round(ta, 4),
        "test_f1": round(tf1, 4),
        "params_m": hparams["params_m"],
        "flops_g": round(flops_g, 4) if flops_g is not None else None,
        "latency_cpu_ms": round(lat_cpu, 4) if lat_cpu is not None else None,
        "latency_gpu_ms": round(lat_gpu, 4) if lat_gpu is not None else None,
        "outdir": args.outdir
    }
    print("=== RUN SUMMARY ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    # Append to a results.csv in outdir
    res_csv = os.path.join(args.outdir, "result_summary.csv")
    write_header = not os.path.exists(res_csv)
    with open(res_csv, "a") as f:
        if write_header:
            f.write("model,dataset,epochs,batch,opt,lr0,wd,sched,aug,seed,val_best_acc,test_acc,test_f1,params_m,flops_g,latency_cpu_ms,latency_gpu_ms,outdir\n")
        f.write(",".join(str(summary[k]) for k in ["model","dataset","epochs","batch","opt","lr0","wd","sched","aug","seed",
                                                   "val_best_acc","test_acc","test_f1","params_m","flops_g","latency_cpu_ms","latency_gpu_ms","outdir"]) + "\n")

if __name__ == "__main__":
    main()
