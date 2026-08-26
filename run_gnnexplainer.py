r"""
run_gnnexplainer.py
---------------------
Applies GNNExplainer (via PyTorch Geometric's built-in torch_geometric.explain
module) to the trained GAT model, producing:
  1. gnnexplainer_results.csv -- one row per explained test graph, with the
     retained (thresholded) explanatory edges, true/predicted class, and
     model confidence.
  2. A folder of pitch-diagram PNGs for a sample of explained instances,
     showing player positions and the highlighted explanatory edges.
  3. explanation_edge_sets.pt -- the raw retained edge sets per instance,
     saved for stability_analysis.py to consume.

IMPORTANT DESIGN NOTE ON EDGE IDENTITY ACROSS DIFFERENT GRAPHS:
Each graph in the test set comes from a different match, different players,
and a different (small) set of local node indices -- node 0 in one graph is
not the same player as node 0 in another. This means retained edges CANNOT
be compared across graphs by raw node index. For within-graph reporting
(the CSV and pitch diagrams), edges are described using each node's actual
role and normalised pitch position, which IS meaningful and human-readable.
For the cross-graph stability analysis (stability_analysis.py), edges are
instead identified by the ROLE-PAIR of the two connected nodes (e.g.
"DEF-MID"), since role-pair is the only edge descriptor that is
consistently comparable across different graphs with different players.
This is a deliberate methodological choice, not an oversight -- see the
corresponding note in Chapter 3, Section 3.8.

Usage:
    python run_gnnexplainer.py --graphs_path .\graphs_360_full\all_graphs.pt \
        --model_path .\model_output_360\football_gat_model.pt \
        --out_dir .\explanations_360 --n_diagrams 20
"""

import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["build_up", "pressing", "transition", "scoring_opp"]
ROLE_NAMES = ["GK", "DEF", "MID", "FWD"]
EDGE_THRESHOLD = 0.5
MIN_EDGES = 3
MAX_EDGES = 7


class GATClassifier(nn.Module):
    """Must match train_gat.py's architecture exactly for weights to load."""
    def __init__(self, in_channels, hidden_channels=64, heads=8,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels // heads,
                             heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, hidden_channels,
                             heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return self.fc(x)


def match_level_split(graphs, val_frac=0.15, test_frac=0.15, seed=42):
    """Same split logic as train_gat.py, so we explain the SAME test set
    the model was actually evaluated on."""
    forced_train_idx = [i for i, g in enumerate(graphs)
                         if getattr(g, "match_id", None) == "BLOCK_UNATTRIBUTED"]
    splittable_idx = [i for i, g in enumerate(graphs)
                       if getattr(g, "match_id", None) != "BLOCK_UNATTRIBUTED"]
    groups = [graphs[i].match_id for i in splittable_idx]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_pos, test_pos = next(gss1.split(splittable_idx, groups=groups))
    trainval_idx = [splittable_idx[p] for p in trainval_pos]
    test_idx = [splittable_idx[p] for p in test_pos]

    trainval_groups = [graphs[i].match_id for i in trainval_idx]
    relative_val_frac = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_pos, val_pos = next(gss2.split(trainval_idx, groups=trainval_groups))
    train_idx = [trainval_idx[p] for p in train_pos] + forced_train_idx
    val_idx = [trainval_idx[p] for p in val_pos]

    return train_idx, val_idx, test_idx


def role_of_node(feat_vector):
    """Node feature layout (see create_all_graphs_360.py):
    [x, y, vx, vy, dist_to_ball, team_indicator,
     role_GK, role_DEF, role_MID, role_FWD, possession_flag,
     opp_5m, mate_5m, pressure_score]
    Role one-hot occupies indices 6-9."""
    role_onehot = feat_vector[6:10]
    idx = int(np.argmax(role_onehot))
    return ROLE_NAMES[idx]


def get_retained_edges(edge_index, edge_mask, threshold=EDGE_THRESHOLD,
                        min_edges=MIN_EDGES, max_edges=MAX_EDGES):
    """
    Thresholds the edge mask, but adaptively falls back to top-k if
    thresholding alone yields too few or too many edges, so that
    explanations stay within the practically interpretable 3-7 edge
    range described in Chapter 3.
    """
    mask_np = edge_mask.detach().cpu().numpy()
    order = np.argsort(-mask_np)  # descending

    above_threshold = np.where(mask_np >= threshold)[0]

    if min_edges <= len(above_threshold) <= max_edges:
        keep = above_threshold
    elif len(above_threshold) > max_edges:
        keep = order[:max_edges]
    else:
        # too few survived thresholding -- take top min_edges regardless
        keep = order[:min(min_edges, len(mask_np))]

    return sorted(keep.tolist())


def draw_pitch_diagram(data, edge_index_np, retained_edge_ids, pred_label,
                        true_label, confidence, out_path):
    """Simple pitch diagram: player dots (coloured by predicted involvement)
    with retained explanatory edges drawn as lines, unretained edges faint."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Pitch outline (normalised 0-1 coordinates, matching feature encoding)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.5))
    ax.axvline(0.5, color="black", linewidth=0.8)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.08, fill=False, edgecolor="black", linewidth=0.8))

    xs = data.x[:, 0].numpy()
    ys = data.x[:, 1].numpy()

    # Draw all edges faintly first
    for i, (u, v) in enumerate(edge_index_np.T):
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], color="lightgray", linewidth=0.8, zorder=1)

    # Draw retained (explanatory) edges highlighted
    for eid in retained_edge_ids:
        u, v = edge_index_np[:, eid]
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], color="crimson", linewidth=2.5, zorder=2)

    # Draw nodes with role labels
    for i in range(len(xs)):
        role = role_of_node(data.x[i].numpy())
        ax.scatter(xs[i], ys[i], s=250, color="steelblue", edgecolor="black", zorder=3)
        ax.annotate(role, (xs[i], ys[i]), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=4)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"True: {CLASSES[true_label]} | Predicted: {CLASSES[pred_label]} "
                 f"(confidence {confidence:.2f})\nHighlighted edges = GNNExplainer's "
                 f"explanation", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_diagrams", type=int, default=20,
                         help="Number of pitch diagrams to render (from the test set)")
    parser.add_argument("--n_explain", type=int, default=500,
                         help="Number of test instances to run GNNExplainer on "
                              "(running on the full test set can be slow; a "
                              "representative sample is usually sufficient)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diagrams_dir = os.path.join(args.out_dir, "pitch_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")  # explanation optimisation is per-instance; CPU is fine

    print("Loading graphs...")
    graphs = torch.load(args.graphs_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    _, _, test_idx = match_level_split(graphs, seed=args.seed)
    print(f"Test set: {len(test_idx)} graphs")

    if len(test_idx) > args.n_explain:
        rng = np.random.default_rng(args.seed)
        sampled_idx = rng.choice(test_idx, size=args.n_explain, replace=False).tolist()
    else:
        sampled_idx = test_idx

    in_channels = graphs[0].x.shape[1]
    model = GATClassifier(in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    csv_rows = []
    edge_sets_for_stability = []  # list of dicts: {true_label, role_pair_set}
    n_diagrams_drawn = 0

    for count, idx in enumerate(sampled_idx):
        data = graphs[idx]
        x = data.x
        edge_index = data.edge_index

        if edge_index.shape[1] == 0:
            continue  # nothing to explain

        with torch.no_grad():
            logits = model(x, edge_index)
            probs = F.softmax(logits, dim=1)
            pred_label = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_label].item())

        true_label = int(data.y.item())

        explanation = explainer(x, edge_index)
        edge_mask = explanation.edge_mask

        retained = get_retained_edges(edge_index, edge_mask)
        edge_index_np = edge_index.numpy()

        # Build human-readable edge descriptions using role + approx position
        edge_descriptions = []
        role_pairs = []
        for eid in retained:
            u, v = edge_index_np[:, eid]
            role_u = role_of_node(x[u].numpy())
            role_v = role_of_node(x[v].numpy())
            edge_descriptions.append(f"{role_u}(node{u})-{role_v}(node{v})")
            role_pairs.append("-".join(sorted([role_u, role_v])))

        csv_rows.append({
            "graph_idx": idx,
            "match_id": getattr(data, "match_id", "unknown"),
            "true_label": CLASSES[true_label],
            "predicted_label": CLASSES[pred_label],
            "confidence": round(confidence, 4),
            "correct": int(true_label == pred_label),
            "num_nodes": x.shape[0],
            "num_retained_edges": len(retained),
            "retained_edges": "; ".join(edge_descriptions),
        })

        edge_sets_for_stability.append({
            "graph_idx": idx,
            "true_label": true_label,
            "role_pair_set": frozenset(role_pairs),
            "mean_features": x.mean(dim=0).numpy(),  # for clustering similar instances
        })

        if n_diagrams_drawn < args.n_diagrams:
            out_path = os.path.join(diagrams_dir, f"explanation_{idx}.png")
            draw_pitch_diagram(data, edge_index_np, retained, pred_label,
                                true_label, confidence, out_path)
            n_diagrams_drawn += 1

        if (count + 1) % 50 == 0:
            print(f"  Explained {count + 1}/{len(sampled_idx)} instances...")

    csv_path = os.path.join(args.out_dir, "gnnexplainer_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    torch.save(edge_sets_for_stability, os.path.join(args.out_dir, "explanation_edge_sets.pt"))

    n_correct = sum(r["correct"] for r in csv_rows)
    print(f"\n{'='*50}")
    print(f"Explained {len(csv_rows)} test instances")
    print(f"Explanation accuracy check: {n_correct}/{len(csv_rows)} "
          f"({100.0 * n_correct / len(csv_rows):.1f}%) predictions were correct")
    print(f"Pitch diagrams saved: {n_diagrams_drawn} -> {diagrams_dir}")
    print(f"Results CSV: {csv_path}")
    print(f"Edge sets for stability analysis: "
          f"{os.path.join(args.out_dir, 'explanation_edge_sets.pt')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
