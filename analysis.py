# rewind_analysis.py
# Combines topology (graph) analysis of masks/weights and weight histograms
# for GD and EG runs produced by rewind_prune.py

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# Defaults aligned to your generated filenames
# ----------------------------
DEFAULT_PRUNE_DIR = "prune_out"
DEFAULT_OUTDIR = "analysis"

DEFAULT_GD_MASK = os.path.join(DEFAULT_PRUNE_DIR, "gd_prune_masks", "mask_final.pt")
DEFAULT_EG_MASK = os.path.join(DEFAULT_PRUNE_DIR, "eg_prune_masks", "mask_final.pt")

DEFAULT_GD_WEIGHTS = os.path.join(DEFAULT_PRUNE_DIR, "gd_prune_lottery_ticket_trained_weights.pth")
DEFAULT_EG_WEIGHTS = os.path.join(DEFAULT_PRUNE_DIR, "eg_prune_lottery_ticket_trained_weights.pth")

DEFAULT_GD_SPARSITY = os.path.join(DEFAULT_PRUNE_DIR, "gd_prune_sparsity_log.csv")
DEFAULT_EG_SPARSITY = os.path.join(DEFAULT_PRUNE_DIR, "eg_prune_sparsity_log.csv")

DEFAULT_GD_METRICS = os.path.join(DEFAULT_PRUNE_DIR, "gd_prune_lottery_ticket_metrics.csv")
DEFAULT_EG_METRICS = os.path.join(DEFAULT_PRUNE_DIR, "eg_prune_lottery_ticket_metrics.csv")


# ----------------------------
# Graph building helpers
# ----------------------------
def load_recurrent_weights(state_dict: dict) -> List[Tuple[str, int, np.ndarray]]:
    items = []
    for key, tensor in state_dict.items():
        if not key.startswith("rnn.weight_hh_l"):
            continue
        suffix = key.split("weight_hh_l", 1)[1]  # e.g. '0' or '0_reverse'
        is_rev = suffix.endswith("_reverse")
        layer_idx = int(suffix[:-8]) if is_rev else int(suffix)
        items.append((key, layer_idx, tensor.detach().cpu().numpy()))
    items.sort(key=lambda t: (t[1], t[0]))
    return items


def top_fraction_threshold(matrix: np.ndarray, keep_fraction: float) -> np.ndarray:
    mat = matrix.copy()
    np.fill_diagonal(mat, 0.0)
    if keep_fraction is None:
        return mat
    abs_vals = np.abs(mat).flatten()
    abs_vals = abs_vals[abs_vals > 0]
    if abs_vals.size == 0:
        return np.zeros_like(mat)
    k = max(1, int(np.ceil(keep_fraction * abs_vals.size)))
    threshold = np.partition(abs_vals, -k)[-k] if k < abs_vals.size else np.min(abs_vals)
    kept = np.where(np.abs(mat) >= threshold, mat, 0.0)
    np.fill_diagonal(kept, 0.0)
    return kept


def graphs_from_weights(state_dict: dict, keep_fraction: float, use_directed: bool) -> Tuple[List[nx.Graph], List[Tuple[int, int]]]:
    layers = load_recurrent_weights(state_dict)
    graphs, shapes = [], []
    for name, layer_idx, W in layers:
        tag = "R" if name.endswith("_reverse") else "F"
        Wkept = top_fraction_threshold(W, keep_fraction)
        H = Wkept.shape[0]
        shapes.append((H, H))
        G = nx.DiGraph() if use_directed else nx.Graph()
        for i in range(H):
            G.add_node(f"L{layer_idx}{tag}_N{i}", layer=layer_idx, dir=tag, idx=i)
        rows, cols = np.nonzero(Wkept)
        for i, j in zip(rows, cols):
            if i != j:
                G.add_edge(f"L{layer_idx}{tag}_N{i}", f"L{layer_idx}{tag}_N{j}")
        graphs.append(G)
    return graphs, shapes


def graphs_from_masks(mask_path: str, use_directed: bool) -> Tuple[List[nx.Graph], List[Tuple[int, int]]]:
    masks = torch.load(mask_path, map_location="cpu")
    graphs, shapes = [], []
    for key, mask_tensor in masks.items():
        if not (key.startswith("rnn.weight_hh_l") and key.endswith("_mask")):
            continue
        base = key[:-5]  # strip '_mask'
        suffix = base.split("weight_hh_l", 1)[1]  # e.g. '0' or '0_reverse'
        tag = "R" if suffix.endswith("_reverse") else "F"
        layer_idx = int(suffix[:-8]) if tag == "R" else int(suffix)

        M = (mask_tensor.detach().cpu().numpy() > 0).astype(np.uint8)
        np.fill_diagonal(M, 0)
        H = M.shape[0]
        shapes.append((H, H))

        G = nx.DiGraph() if use_directed else nx.Graph()
        for i in range(H):
            G.add_node(f"L{layer_idx}{tag}_N{i}", layer=layer_idx, dir=tag, idx=i)
        rows, cols = np.nonzero(M)
        for i, j in zip(rows, cols):
            if i != j:
                G.add_edge(f"L{layer_idx}{tag}_N{i}", f"L{layer_idx}{tag}_N{j}")
        graphs.append(G)
    return graphs, shapes


def compose_graph(graphs: List[nx.Graph], use_directed: bool) -> nx.Graph:
    pooled = nx.DiGraph() if use_directed else nx.Graph()
    for g in graphs:
        pooled = nx.compose(pooled, g)
    return pooled


# ----------------------------
# Metrics & plots
# ----------------------------
def largest_component_undirected(G: nx.Graph) -> nx.Graph:
    GU = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    if GU.number_of_nodes() == 0:
        return GU
    if nx.is_connected(GU):
        return GU
    comp = max(nx.connected_components(GU), key=len)
    return GU.subgraph(comp).copy()


def small_world_sigma(G: nx.Graph, random_samples: int = 20, seed: int = 42) -> Tuple[float, float, float, float, float]:
    GU = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    if GU.number_of_nodes() < 3 or GU.number_of_edges() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    GCC = largest_component_undirected(GU)
    C = nx.average_clustering(GCC)
    try:
        L = nx.average_shortest_path_length(GCC)
    except nx.NetworkXError:
        return float("nan"), C, float("nan"), float("nan"), float("nan")

    n, m = GCC.number_of_nodes(), GCC.number_of_edges()
    rng = np.random.default_rng(seed)
    c_list, l_list = [], []
    tries = 0
    while len(c_list) < random_samples and tries < 10 * random_samples:
        tries += 1
        GR = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
        if not nx.is_connected(GR):
            continue
        c_list.append(nx.average_clustering(GR))
        l_list.append(nx.average_shortest_path_length(GR))
    if not c_list or not l_list:
        return float("nan"), C, L, float("nan"), float("nan")
    Cr, Lr = float(np.mean(c_list)), float(np.mean(l_list))
    if Cr <= 0.0 or Lr <= 0.0:
        return float("nan"), C, L, Cr, Lr
    sigma = (C / Cr) / (L / Lr)
    return float(sigma), float(C), float(L), Cr, Lr


def undirected_sparsity(G: nx.Graph, layer_shapes: List[Tuple[int, int]]) -> float:
    GU = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    max_edges = sum(H * (H - 1) // 2 for H, _ in layer_shapes)
    return GU.number_of_edges() / float(max_edges) if max_edges > 0 else float("nan")


def directed_sparsity(G: nx.Graph, layer_shapes: List[Tuple[int, int]]) -> float:
    if not isinstance(G, nx.DiGraph):
        return float("nan")
    max_edges = sum(H * (H - 1) for H, _ in layer_shapes)
    return G.number_of_edges() / float(max_edges) if max_edges > 0 else float("nan")


def read_final_stage2_sparsity(csv_path: str) -> Tuple[float, float]:
    try:
        df = pd.read_csv(csv_path)
        last = df.iloc[-1]
        return float(last["Remaining Fraction"]), float(last["Pruned Fraction"])
    except Exception:
        return float("nan"), float("nan")


def save_clustering_histogram(G: nx.Graph, bins: int, out_path: str, title: str):
    coeffs = list(nx.clustering(G.to_undirected()).values())
    plt.figure(figsize=(6, 4))
    plt.hist(coeffs, bins=bins)
    plt.title(title)
    plt.xlabel("Clustering coefficient")
    plt.ylabel("Count")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_degree_histogram(G: nx.Graph, out_path: str, title: str):
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=50)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_lcc_plot(G: nx.Graph, out_path: str, title: str):
    GU = G.to_undirected()
    if GU.number_of_nodes() == 0:
        return
    comp = max(nx.connected_components(GU), key=len)
    GCC = GU.subgraph(comp)
    plt.figure(figsize=(6, 6))
    nx.draw(GCC, node_size=10)
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_histograms_from_weights(state_path: str, label: str, outdir: str):
    state = torch.load(state_path, map_location="cpu")
    mags = []
    for name, tensor in state.items():
        if any(k in name for k in ["rnn.weight_hh_l", "rnn.weight_ih_l", "fc.weight", "classifier.weight"]):
            w = tensor.detach().cpu().numpy().ravel()
            w = w[np.isfinite(w)]
            mags.append(np.abs(w))
    mags = np.concatenate(mags) if mags else np.array([])

    if mags.size == 0:
        print(f"[WARN] No weights found for {label}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(mags, bins=80)
    plt.xlabel("|weight|")
    plt.ylabel("Count")
    plt.title(f"{label}: |w| histogram")
    plt.grid(True, axis="y", alpha=0.4)
    plt.savefig(os.path.join(outdir, f"{label}_abs_weights_hist.png"))
    plt.close()

    mags_nz = mags[mags > 0]
    if mags_nz.size > 0:
        logs = np.log10(mags_nz)
        plt.figure(figsize=(8, 5))
        plt.hist(logs, bins=80)
        plt.xlabel("log10(|weight|)")
        plt.ylabel("Count")
        plt.title(f"{label}: log10(|w|) histogram")
        plt.grid(True, axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{label}_log_abs_weights_hist.png"))
        plt.close()


# ----------------------------
# Core analysis per model
# ----------------------------
def analyze_one(label: str,
                weights_path: str,
                masks_path: str,
                sparsity_csv: str,
                metrics_csv: str,
                out_dir: str,
                use_masks_first: bool,
                keep_fraction: float,
                use_directed: bool) -> Dict[str, float]:

    if use_masks_first and masks_path and os.path.exists(masks_path):
        layer_graphs, layer_shapes = graphs_from_masks(masks_path, use_directed)
    else:
        state = torch.load(weights_path, map_location="cpu")
        layer_graphs, layer_shapes = graphs_from_weights(state, keep_fraction, use_directed)

    pooled = compose_graph(layer_graphs, use_directed)

    # Per-layer table
    perlayer_rows = []
    for idx, G in enumerate(layer_graphs):
        sigma, C, L, Cr, Lr = small_world_sigma(G, random_samples=20)
        GU = G.to_undirected() if isinstance(G, nx.DiGraph) else G
        GCC = largest_component_undirected(G)
        coverage = (GCC.number_of_nodes() / GU.number_of_nodes()) if GU.number_of_nodes() else 0.0
        perlayer_rows.append({
            "Model": label, "Layer": idx,
            "nodes": GU.number_of_nodes(), "edges_undirected": GU.number_of_edges(),
            "sigma": sigma, "C": C, "L": L, "C_rand": Cr, "L_rand": Lr,
            "LCC_nodes": GCC.number_of_nodes(), "LCC_coverage": coverage
        })

    per_layer_csv = os.path.join(out_dir, "per_layer_results.csv")
    per_df = pd.DataFrame(perlayer_rows)
    if os.path.exists(per_layer_csv):
        old = pd.read_csv(per_layer_csv)
        per_df = pd.concat([old, per_df], ignore_index=True)
    per_df.to_csv(per_layer_csv, index=False)

    # Pooled graph stats
    sigma, C, L, Cr, Lr = small_world_sigma(pooled, random_samples=20)
    GU = pooled.to_undirected() if isinstance(pooled, nx.DiGraph) else pooled
    GCC = largest_component_undirected(pooled)
    coverage = (GCC.number_of_nodes() / GU.number_of_nodes()) if GU.number_of_nodes() else 0.0
    und_spars = undirected_sparsity(pooled, layer_shapes)
    dir_spars = directed_sparsity(pooled, layer_shapes) if use_directed else float("nan")

    train_remaining, train_pruned = read_final_stage2_sparsity(sparsity_csv)

    save_clustering_histogram(pooled, bins=40,
                              out_path=os.path.join(out_dir, f"{label}_clustering_hist.png"),
                              title=f"{label}: Clustering Coefficients")
    save_degree_histogram(pooled,
                          out_path=os.path.join(out_dir, f"{label}_degree_hist.png"),
                          title=f"{label}: Degree Distribution")
    save_lcc_plot(pooled,
                  out_path=os.path.join(out_dir, f"{label}_graph_LCC.png"),
                  title=f"{label}: Largest Connected Component")

    save_histograms_from_weights(weights_path, label, out_dir)

    last_acc = float("nan")
    if metrics_csv and os.path.exists(metrics_csv):
        try:
            mdf = pd.read_csv(metrics_csv)
            last_acc = float(mdf.iloc[-1]["Test Acc (%)"])
        except Exception:
            pass

    return {
        "sigma": sigma, "C": C, "L": L, "C_rand": Cr, "L_rand": Lr,
        "nodes": GU.number_of_nodes(), "edges": GU.number_of_edges(),
        "LCC_nodes": GCC.number_of_nodes(), "LCC_edges": GCC.number_of_edges(),
        "LCC_coverage": coverage,
        "graph_sparsity_undirected": und_spars,
        "graph_sparsity_directed": dir_spars,
        "train_remaining_frac": train_remaining,
        "train_pruned_frac": train_pruned,
        "final_test_acc_pct": last_acc
    }


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Topology + weight histogram analysis for GD/EG rewind pipeline")
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--prefer_masks", action="store_true",
                    help="Use masks if present; fall back to weights otherwise.")
    ap.add_argument("--edge_keep_frac", type=float, default=0.01,
                    help="Top-|W| fraction per layer when using weights (ignored if masks are used).")
    ap.add_argument("--directed", action="store_true",
                    help="Build directed graphs; metrics are computed on undirected views.")
    ap.add_argument("--gd_mask", type=str, default=DEFAULT_GD_MASK)
    ap.add_argument("--eg_mask", type=str, default=DEFAULT_EG_MASK)
    ap.add_argument("--gd_weights", type=str, default=DEFAULT_GD_WEIGHTS)
    ap.add_argument("--eg_weights", type=str, default=DEFAULT_EG_WEIGHTS)
    ap.add_argument("--gd_sparsity", type=str, default=DEFAULT_GD_SPARSITY)
    ap.add_argument("--eg_sparsity", type=str, default=DEFAULT_EG_SPARSITY)
    ap.add_argument("--gd_metrics", type=str, default=DEFAULT_GD_METRICS)
    ap.add_argument("--eg_metrics", type=str, default=DEFAULT_EG_METRICS)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Analyzing GD...")
    gd_stats = analyze_one("GD", args.gd_weights, args.gd_mask,
                           args.gd_sparsity, args.gd_metrics,
                           args.outdir, args.prefer_masks,
                           args.edge_keep_frac, args.directed)

    print("Analyzing EG...")
    eg_stats = analyze_one("EG", args.eg_weights, args.eg_mask,
                           args.eg_sparsity, args.eg_metrics,
                           args.outdir, args.prefer_masks,
                           args.edge_keep_frac, args.directed)

    results_df = pd.DataFrame.from_dict({"GD": gd_stats, "EG": eg_stats}, orient="index")
    print("\n=== REWIND ANALYSIS RESULTS ===")
    print(results_df.to_string())
    results_df.to_csv(os.path.join(args.outdir, "rewind_analysis_results.csv"))

    print(f"\nArtifacts written to: {args.outdir}")
    print("Tables: rewind_analysis_results.csv, per_layer_results.csv")
    print("Plots: *clustering_hist.png, *degree_hist.png, *graph_LCC.png, *_abs_weights_hist.png, *_log_abs_weights_hist.png")


if __name__ == "__main__":
    main()
