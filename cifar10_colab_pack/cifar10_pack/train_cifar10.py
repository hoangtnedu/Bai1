
# train_cifar10.py
import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import f1_score
from models_simple import SimpleCNN

def set_seed(seed: int = 42):
    import torch.backends.cudnn as cudnn
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False; cudnn.benchmark = True

def cifar10_loaders(data_root, batch_size=128, val_ratio=0.1, seed=42, aug=True, num_workers=2):
    mean = (0.4914, 0.4822, 0.4465); std  = (0.2470, 0.2435, 0.2616)
    train_tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if aug else []
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.Compose(train_tf))
    test_set   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.Compose(test_tf))
    g = torch.Generator().manual_seed(seed); n = len(train_full); n_val = int(n * val_ratio); n_train = n - n_val
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=g)
    dl = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers, pin_memory=True)
    return dl(train_set, True), dl(val_set, False), dl(test_set, False)

def build_model(name: str, num_classes=10):
    name = name.lower()
    if name == "simplecnn": return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":  m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if name == "mobilenetv2": m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.last_channel, num_classes); return m
    if name == "efficientnet_b0": m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); return m
    if name == "vgg11_bn": m = models.vgg11_bn(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes); return m
    raise ValueError(f"Unknown model: {name}")

def count_params_m(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_optimizer(model, name, lr, wd, momentum=0.9):
    name = name.lower()
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True) if name=="sgd" \
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def get_scheduler(optimizer, name, epochs, step_size=30, gamma=0.1):
    name = name.lower()
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0) if name=="cosine" \
        else (StepLR(optimizer, step_size=step_size, gamma=gamma) if name=="step" else None)

def train_one_epoch(model, loader, device, optimizer, scaler=None):
    model.train(); total=correct=0; loss_sum=0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(x); loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward(); optimizer.step()
        loss_sum += loss.item()*y.size(0); pred = logits.argmax(1)
        correct += (pred==y).sum().item(); total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total=correct=0; loss_sum=0.0; ys=[]; ps=[]
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x); loss = F.cross_entropy(logits, y)
        loss_sum += loss.item()*y.size(0); pred=logits.argmax(1)
        correct += (pred==y).sum().item(); total += y.size(0)
        ys.append(y.cpu()); ps.append(pred.cpu())
    import torch as T; y_true=T.cat(ys).numpy(); y_pred=T.cat(ps).numpy()
    return loss_sum/total, correct/total, f1_score(y_true, y_pred, average="macro")

def measure_latency(model, device, input_size=(1,3,32,32), warmup=30, iters=100):
    import time; x=torch.randn(*input_size).to(device)
    for _ in range(warmup): _=model(x)
    if device.type=="cuda": torch.cuda.synchronize()
    t0=time.time(); 
    for _ in range(iters): _=model(x)
    if device.type=="cuda": torch.cuda.synchronize()
    return (time.time()-t0)*1000.0/iters

def maybe_flops(model, device):
    try:
        from thop import profile
        x = torch.randn(1,3,32,32).to(device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return flops/1e9
    except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="runs/exp_cifar10")
    ap.add_argument("--model", type=str, default="resnet18",
                    choices=["simplecnn","resnet18","mobilenetv2","efficientnet_b0","vgg11_bn"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd","adamw"])
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--sched", type=str, default="cosine", choices=["cosine","step","none"])
    ap.add_argument("--step_size", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed); os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = cifar10_loaders(args.data_root, args.batch, args.val_ratio, args.seed, not args.no_aug, args.num_workers)
    model = build_model(args.model, 10).to(device)
    opt = get_optimizer(model, args.opt, args.lr, args.wd)
    sched = None if args.sched=="none" else get_scheduler(opt, args.sched, args.epochs, args.step_size, args.gamma)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type=="cuda") else None

    hparams = vars(args).copy(); hparams["device"]=str(device); hparams["params_m"]=round(count_params_m(model),4)
    json.dump(hparams, open(os.path.join(args.outdir,"hparams.json"),"w"), indent=2)

    log_path = os.path.join(args.outdir,"train_log.csv")
    open(log_path,"w").write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1,lr\n")

    best_acc=-1.0
    for ep in range(1, args.epochs+1):
        tl, ta = train_one_epoch(model, train_loader, device, opt, scaler)
        vl, va, vf1 = evaluate(model, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        if sched: sched.step()
        open(log_path,"a").write(f"{ep},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f},{vf1:.6f},{lr_now:.6f}\\n")
        if va>best_acc:
            best_acc=va; torch.save({"model": model.state_dict(), "epoch": ep, "val_acc": va}, os.path.join(args.outdir,"best.pt"))
        print(f"[{ep:03d}/{args.epochs}] train_acc={ta:.4f} val_acc={va:.4f} val_f1={vf1:.4f}")

    ckpt = torch.load(os.path.join(args.outdir,"best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    tl, ta, tf1 = evaluate(model, test_loader, device)

    model.eval()
    lat_cpu = measure_latency(model.to("cpu"), torch.device("cpu"))
    lat_gpu = measure_latency(model.to(device), device) if torch.cuda.is_available() else None
    flops_g = maybe_flops(model.to(device), device)

    summary = {
        "model": args.model, "dataset": "cifar10", "epochs": args.epochs, "batch": args.batch,
        "opt": args.opt, "lr0": args.lr, "wd": args.wd, "sched": args.sched, "aug": int(not args.no_aug), "seed": args.seed,
        "val_best_acc": round(best_acc,4), "test_acc": round(ta,4), "test_f1": round(tf1,4),
        "params_m": hparams["params_m"],
        "flops_g": round(flops_g,4) if flops_g is not None else None,
        "latency_cpu_ms": round(lat_cpu,4) if lat_cpu is not None else None,
        "latency_gpu_ms": round(lat_gpu,4) if lat_gpu is not None else None,
        "outdir": args.outdir
    }
    print("=== RUN SUMMARY ==="); [print(f"{k}: {v}") for k,v in summary.items()]
    res_csv = os.path.join(args.outdir,"result_summary.csv"); write_header=not os.path.exists(res_csv)
    with open(res_csv,"a") as f:
        if write_header:
            f.write("model,dataset,epochs,batch,opt,lr0,wd,sched,aug,seed,val_best_acc,test_acc,test_f1,params_m,flops_g,latency_cpu_ms,latency_gpu_ms,outdir\\n")
        f.write(",".join(str(summary[k]) for k in ["model","dataset","epochs","batch","opt","lr0","wd","sched","aug","seed","val_best_acc","test_acc","test_f1","params_m","flops_g","latency_cpu_ms","latency_gpu_ms","outdir"])+"\\n")

if __name__ == "__main__":
    main()
