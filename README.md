# MSc Project — Fashion-MNIST with Gradient Descent vs Exponential Gradient Optimisation  
**University of Leeds – MSc Advanced Computer Science**

---

## 📖 Overview

This project investigates the performance of **Gradient Descent (GD)** and **Exponential Gradient (EG)** optimisation methods on the **Fashion-MNIST dataset** using a recurrent neural network (RNN).  

The work explores:  
- Baseline model training with both optimisers.  
- Iterative Magnitude Pruning (IMP) with **weight rewinding**.  
- Comparison of model sparsity and performance.  
- Graph-theoretic and weight distribution analysis of trained and pruned networks.  
- Systematic learning-rate sweeps for optimiser tuning.  

This study is conducted as part of the **MSc Advanced Computer Science** program at the **University of Leeds**.

---

## ✨ Key Features

- **Baseline Training**: Run experiments with GD and EG optimisers.  
- **Lottery Ticket Hypothesis (LTH)**: Pruning with rewinding to earlier weight states.  
- **Mask Generation**: Iterative pruning masks stored for reproducibility.  
- **Graph Analysis**: Degree distribution, clustering, connected components.  
- **Weight Distribution Analysis**: Histograms of absolute/log weights across layers.  
- **Learning Rate Experiments**: Automated sweeps for GD and EG.

---

## 📂 Project Structure
```
├── base_train.py # Baseline training (GD/EG)
├── prune_train.py # Pruning + rewinding (IMP)
├── analysis.py # Graph and weight analysis
├── run_lr_experiments.sh # Automated LR sweeps
├── EG_optimiser/ # Custom optimiser (sgd_eg.SGD)
├── data/ # Fashion-MNIST (auto-downloaded)
├── out/ # Baseline outputs
└── prune_out/ # Pruning outputs
```


---

## ⚙️ Setup & Installation

### Requirements
- Python 3.9+  
- PyTorch (CPU or GPU)  
- Libraries: `numpy`, `pandas`, `matplotlib`, `networkx`, `scikit-learn`, `tqdm`

### Installation Steps

```bash
git clone https://github.com/ShibjiRout/Msc_project.git
cd Msc_project

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch torchvision torchaudio
pip install numpy pandas matplotlib networkx tqdm scikit-learn
```
Dataset is automatically downloaded into ./data.
## ▶️ Usage

### 1. Baseline Training

Run GD or EG baseline experiments. Outputs stored in `./out`.

```bash
# GD
python base_train.py --alg gd --epochs 20 --lr 0.022 --batch_size 256 --save_prefix gd

# EG
python base_train.py --alg eg --epochs 20 --lr 0.75 --batch_size 256 --save_prefix eg
```
Output file:
out/{prefix}_training_metrics.csv
### 2. Pruning with Weight Rewinding

Perform Iterative Magnitude Pruning (IMP).
```bash
# GD pruning (80% sparsity, 5 steps, rewind at epoch 1)
python prune_train.py --alg gd --epochs 20 --lr 0.002 --batch_size 256 \
  --target_prune 0.80 --prune_steps 5 --rewind_epoch 1 --save_prefix gd_prune

# EG pruning
python prune_train.py --alg eg --epochs 20 --lr 0.75 --batch_size 256 \
  --target_prune 0.80 --prune_steps 5 --rewind_epoch 1 --save_prefix eg_prune

```

**Generated in `./prune_out/`:**

- `{prefix}_lottery_ticket_metrics.csv`  
- `{prefix}_sparsity_log.csv`  
- `{prefix}_lottery_ticket_trained_weights.pth`  
- `{prefix}_theta_k_ep{rewind_epoch}.pth`  
- `{prefix}_masks/mask_final.pt` (and intermediate masks)  
### 3. Analysis

Analyse masks or weights for graph properties and weight distributions.
```bash
# Preferred: masks
python analysis.py --outdir analysis --prefer_masks 

```
**Outputs in `./analysis/`:**
- `rewind_analysis_results.csv`  
- `per_layer_results.csv`  
- Histograms: degree, clustering, abs weights, log abs weights  
- Graph visualisations: largest connected component  

---

### 4. Learning Rate Sweeps

Automated batch experiments.
```bash
bash run_lr_experiments.sh

```
- **EG**: `1.1 1.2 1.0 0.9 0.8 0.7 0.75 0.77`  
- **GD**: `0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06`  

---

## 📊 Training Details

- **Model**: RNN(tanh), hidden_units=256, rnn_layers=3  
- **Optimisers**: GD and EG (`sgd_eg.SGD`)  
- **Init**: orthogonal (`weight_hh`), Xavier (`weight_ih`)  
- **Scheduler**: StepLR (γ=0.9 baseline, γ=0.5→0.1 pruning)  
- **Gradient clipping**: max-norm=1.0  
- **Transforms**: `ToTensor`, `Normalize((0.2860,), (0.3530,))`  
- **Reproducibility**: deterministic seeds (Python/NumPy/Torch)  

---

## 📌 Troubleshooting

- Ensure you run scripts from the **repo root** (`EG_optimiser/` is imported).  
- `--rewind_epoch` must be less than `--epochs`.  
- Use `--prefer_masks` for more accurate analysis.  
- Pass explicit file paths if using custom `--save_prefix`.  

---

## 📚 Reference Dataset

- **Fashion-MNIST**: Xiao, Han, et al. *"Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."* (2017).  

---

## 👤 Author

Shibji Shekhar  
MSc Advanced Computer Science, University of Leeds  
