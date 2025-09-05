# rewind_prune.py
# -*- coding: utf-8 -*-
import os, sys, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Custom optimizer (GD / EG)
# -------------------------
sys.path.append('./EG_optimiser')
from optim_eg import sgd_eg  # provides sgd_eg.SGD(update_alg='gd'|'eg')

# -------------------------
# Args
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--alg', type=str, default='gd', choices=['gd','eg'],
                   help="Optimizer: 'gd' or 'eg'")
    p.add_argument('--epochs', type=int, default=20, help='Total epochs INCLUDING warmup')
    p.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    p.add_argument('--batch_size', type=int, default=256, help='Batch size')

    p.add_argument('--target_prune', type=float, default=0.80,
                   help='Target overall pruning fraction for selected weights')
    p.add_argument('--prune_steps', type=int, default=5, help='Number of IMP steps')

    p.add_argument('--rewind_epoch', type=int, default=1,
                   help='Warmup epochs before pruning; snapshot ?_k at this epoch')
    p.add_argument('--rewind_ckpt_path', type=str, default='',
                   help='Optional path to pre-saved ?_k .pth; if missing, it will be created at rewind_epoch')
    p.add_argument('--save_prefix', type=str, default='run',
                   help='Prefix for outputs under prune_out/')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return p.parse_args()

args = get_args()
set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} | alg={args.alg} | seed={args.seed}")

# -------------------------
# Data (FashionMNIST)
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# -------------------------
# Model
# -------------------------
class Model(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3, num_classes=10, bidirectional=False):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=28,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * d, num_classes)
        # sensible init
        for name, p in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        x = x.squeeze(1)  # (B,1,28,28) -> (B,28,28)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)

model = Model().to(device)
print(model)

# -------------------------
# Optimizer & Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
def make_optimizer():
    return sgd_eg.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        update_alg=args.alg,
        weight_decay=args.weight_decay
    )
optimizer = make_optimizer()
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------------
# Prune/rewind helpers (scope matches supervisor guidance)
# - prune recurrent weights: weight_hh_l{l}
# - prune between-layer weights: weight_ih_l{l} for l >= 1
# - skip true input (weight_ih_l0) and output (fc.*)
# -------------------------
def params_to_prune(m: nn.Module):
    pairs = []
    for mod in m.modules():
        if isinstance(mod, nn.RNN):
            L = mod.num_layers
            bidir = getattr(mod, "bidirectional", False)
            for l in range(L):
                # recurrent (always)
                name = f"weight_hh_l{l}"
                if hasattr(mod, name):
                    pairs.append((mod, name))
                # between-layer (skip layer 0 input)
                if l >= 1:
                    name = f"weight_ih_l{l}"
                    if hasattr(mod, name):
                        pairs.append((mod, name))
                # reverse-direction params if bidirectional
                if bidir:
                    name = f"weight_hh_l{l}_reverse"
                    if hasattr(mod, name):
                        pairs.append((mod, name))
                    if l >= 1:
                        name = f"weight_ih_l{l}_reverse"
                        if hasattr(mod, name):
                            pairs.append((mod, name))
    return pairs

def prune_once(amount_step: float):
    prune.global_unstructured(
        params_to_prune(model),
        pruning_method=prune.L1Unstructured,
        amount=amount_step,
    )

def rewind_survivors_to(source_state_dict: dict):
    """Rewind pruned tensors' surviving entries to ?_k (source_state_dict)."""
    with torch.no_grad():
        for mod, pname in params_to_prune(model):
            mask = getattr(mod, f"{pname}_mask", None)
            if mask is None:
                continue
            # pruned modules hold *_orig after pruning is applied
            W_orig = getattr(mod, f"{pname}_orig")
            key = f"rnn.{pname}"
            if key not in source_state_dict:
                # in case some param is absent (e.g., bidir off), skip gracefully
                continue
            src = source_state_dict[key].to(W_orig.device)
            msk = mask.bool()
            W_orig.data[msk] = src[msk]

def current_sparsity_over_scope(m: nn.Module):
    """Compute remaining/pruned fraction over pruned tensors only."""
    total, nonzero = 0, 0
    for mod, pname in params_to_prune(m):
        mask = getattr(mod, f"{pname}_mask", None)
        if mask is None:
            W = getattr(mod, pname).detach()
            total += W.numel()
            nonzero += int((W != 0).sum().item())
        else:
            msk = mask.detach()
            total += msk.numel()
            nonzero += int(msk.sum().item())
    rem = nonzero / total if total > 0 else 0.0
    return rem, 1.0 - rem

def nonzero_and_total_over_scope(m: nn.Module):
    total, nonzero = 0, 0
    for mod, pname in params_to_prune(m):
        mask = getattr(mod, f"{pname}_mask", None)
        if mask is None:
            W = getattr(mod, pname).detach()
            total += W.numel()
            nonzero += int((W != 0).sum().item())
        else:
            msk = mask.detach()
            total += msk.numel()
            nonzero += int(msk.sum().item())
    return nonzero, total

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total

# -------------------------
# Output dirs & CSV loggers
# -------------------------
out_dir = "prune_out"
os.makedirs(out_dir, exist_ok=True)
mask_dir = os.path.join(out_dir, f"{args.save_prefix}_masks")
os.makedirs(mask_dir, exist_ok=True)

train_losses, test_losses, train_accs, test_accs = [], [], [], []
sparsity_log = []  # (epoch, remaining, pruned, nonzeros, total)

def save_metrics_csv(path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Epoch","Train Loss","Train Acc (%)","Test Loss","Test Acc (%)"])
        for ep, trl, tra, tel, tea in zip(range(1,len(train_losses)+1), train_losses, train_accs, test_losses, test_accs):
            w.writerow([ep, trl, tra*100.0, tel, tea*100.0])

def save_sparsity_csv(path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Epoch","Remaining Fraction","Pruned Fraction","Nonzeros","Total"])
        w.writerows(sparsity_log)

# -------------------------
# Plan pruning AFTER warmup
# -------------------------
if args.rewind_epoch >= args.epochs:
    raise ValueError("--rewind_epoch must be < --epochs")

remaining_epochs = args.epochs - args.rewind_epoch
per_step_amount = 1.0 - (1.0 - args.target_prune) ** (1.0 / args.prune_steps)

# spread prune steps evenly over the post-warmup phase
prune_epochs = []
for i in range(1, args.prune_steps + 1):
    e = args.rewind_epoch + int(round(i * remaining_epochs / args.prune_steps))
    prune_epochs.append(e)
prune_epochs = sorted(set(prune_epochs))
if prune_epochs[-1] != args.epochs:
    prune_epochs[-1] = args.epochs

print(f"Target overall prune={args.target_prune:.2%} via {args.prune_steps} steps -> per-step = {per_step_amount:.2%}")
print(f"Warmup (no pruning) until epoch {args.rewind_epoch}; prune at epochs: {prune_epochs}")

# -------------------------
# Prepare ?_k (load or create)
# -------------------------
theta_k = None
ckpt = args.rewind_ckpt_path.strip()
if ckpt:
    if not os.path.isfile(ckpt):
        print(f"[Rewind] ?_k file not found at '{ckpt}'. It will be CREATED automatically at epoch {args.rewind_epoch}.")
        ckpt = ''  # fall through to auto-create
    else:
        tmp = torch.load(ckpt, map_location='cpu')
        if isinstance(tmp, dict) and any(k.startswith('rnn.') for k in tmp.keys()):
            theta_k = tmp
        elif isinstance(tmp, dict) and 'state_dict' in tmp:
            theta_k = tmp['state_dict']
        else:
            raise ValueError("Provided --rewind_ckpt_path does not look like a valid state_dict.")
        print(f"[Rewind] Loaded ?_k from: {args.rewind_ckpt_path}")

# -------------------------
# Training loop
# -------------------------
for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = train_one_epoch()
    te_loss, te_acc = evaluate()
    scheduler.step()

    train_losses.append(tr_loss); train_accs.append(tr_acc)
    test_losses.append(te_loss);  test_accs.append(te_acc)

    print(f"Epoch {epoch}/{args.epochs} | LR {scheduler.get_last_lr()[0]:.6f} | "
          f"Train {tr_loss:.4f}/{tr_acc*100:.2f}% | Test {te_loss:.4f}/{te_acc*100:.2f}%")

    # Snapshot ?_k at the chosen warmup epoch if we didn't load it
    if (theta_k is None) and (epoch == args.rewind_epoch):
        theta_k = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        theta_path = os.path.join(out_dir, f"{args.save_prefix}_theta_k_ep{args.rewind_epoch}.pth")
        torch.save(theta_k, theta_path)
        print(f"[Rewind] Snapshotted and saved ?_k at epoch {epoch} -> {theta_path}")

    # Perform prune step(s) AFTER warmup
    if epoch in prune_epochs:
        print(f"\n>>> PRUNE at epoch {epoch}: removing about {per_step_amount:.2%} of current remaining selected weights")
        prune_once(per_step_amount)

        if theta_k is None:
            raise RuntimeError("?_k is not available. It should be loaded or snapshotted before the first prune.")

        # Rewind surviving weights to ?_k
        rewind_survivors_to(theta_k)
        model.rnn.flatten_parameters()

        # Save masks for this prune step
        masks = {}
        for mod, pname in params_to_prune(model):
            m = getattr(mod, f"{pname}_mask", None)
            if m is not None:
                masks[f"rnn.{pname}_mask"] = m.detach().cpu()
        torch.save(masks, os.path.join(mask_dir, f"mask_ep{epoch}.pt"))

        # Fresh optimizer/scheduler after pruning (standard in LTH)
        optimizer = make_optimizer()
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        rem, pruned = current_sparsity_over_scope(model)
        nnz, tot = nonzero_and_total_over_scope(model)
        print(f"Sparsity (scoped) -> Remaining {rem*100:.2f}% | Pruned {pruned*100:.2f}% | nnz={nnz} / total={tot}\n")
        sparsity_log.append((epoch, rem, pruned, nnz, tot))

# Save final masks once more
final_masks = {}
for mod, pname in params_to_prune(model):
    m = getattr(mod, f"{pname}_mask", None)
    if m is not None:
        final_masks[f"rnn.{pname}_mask"] = m.detach().cpu()
if final_masks:
    torch.save(final_masks, os.path.join(mask_dir, "mask_final.pt"))

# Make pruning permanent (remove masks)
for mod, pname in params_to_prune(model):
    if hasattr(mod, f"{pname}_mask"):
        prune.remove(mod, pname)

# -------------------------
# Save outputs
# -------------------------
metrics_path  = os.path.join(out_dir, f"{args.save_prefix}_lottery_ticket_metrics.csv")
sparsity_path = os.path.join(out_dir, f"{args.save_prefix}_sparsity_log.csv")
weights_path  = os.path.join(out_dir, f"{args.save_prefix}_lottery_ticket_trained_weights.pth")

with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["Epoch","Train Loss","Train Acc (%)","Test Loss","Test Acc (%)"])
    for ep, trl, tra, tel, tea in zip(range(1,len(train_losses)+1), train_losses, train_accs, test_losses, test_accs):
        w.writerow([ep, trl, tra*100.0, tel, tea*100.0])

with open(sparsity_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["Epoch","Remaining Fraction","Pruned Fraction","Nonzeros","Total"])
    w.writerows(sparsity_log)

torch.save(model.state_dict(), weights_path)

print(f"Saved: {metrics_path}, {sparsity_path}, {weights_path}")
print(f"Masks saved in: {mask_dir}")
