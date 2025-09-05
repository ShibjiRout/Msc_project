# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# ---- Custom optimiser (supports update_alg='gd' or 'eg') ----
sys.path.append('./EG_optimiser')
from optim_eg import sgd_eg

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# RNN Model
# ---------------------------
class FashionRNN(nn.Module):
    def __init__(self, hidden_units=256, rnn_layers=3, output_classes=10, bidirectional=False):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=28,
            hidden_size=hidden_units,
            num_layers=rnn_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_units * directions, output_classes)

        # Initialise weights
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = x.squeeze(1)                 # (B,1,28,28) -> (B,28,28)
        outputs, _ = self.rnn(x)         # (B,28,H)
        last_hidden = outputs[:, -1, :]  # last timestep
        return self.fc(last_hidden)      # logits


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, data_loader, device, loss_fn, optimizer):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, correct / total_samples


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='gd', choices=['gd', 'eg'],
                        help="Optimizer algorithm: 'gd' or 'eg'")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--rnn_layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_scheduler', type=str, default='true',
                        help="true/false to use StepLR(step_size=5, gamma=0.9)")
    parser.add_argument('--save_prefix', type=str, default=None,
                        help="Prefix for outputs (default: alg name)")
    args = parser.parse_args()

    use_scheduler = str(args.use_scheduler).lower() in ('1', 'true', 'yes', 'y', 't')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} | alg={args.alg} | seed={args.seed}')

    # Dataset & loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train n={len(train_dataset)} | Test n={len(test_dataset)}")

    # Model, loss, optimizer
    model = FashionRNN(hidden_units=args.hidden_units, rnn_layers=args.rnn_layers).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = sgd_eg.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        update_alg=args.alg,           
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9) if use_scheduler else None

    # Outputs
    os.makedirs('out', exist_ok=True)
    prefix = args.save_prefix if args.save_prefix else args.alg
    csv_path = os.path.join('out', f'{prefix}_training_metrics.csv')
    weights_path = os.path.join('out', f'{prefix}_final_weights.pth')

    # Training loop
    import pandas as pd
    training_history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        training_history.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc (%)": train_acc * 100.0,
            "Test Loss": test_loss,
            "Test Acc (%)": test_acc * 100.0,
            "LR": current_lr,
            "Alg": args.alg,
            "Seed": args.seed,
        })

        print(f"Epoch {epoch:02d}/{args.epochs} | LR {current_lr:.5f} | "
              f"Train {train_loss:.4f}/{train_acc*100:.2f}% | Test {test_loss:.4f}/{test_acc*100:.2f}%")

    
    # Save only metrics (no model weights)
    pd.DataFrame(training_history).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")



if __name__ == "__main__":
    main()
