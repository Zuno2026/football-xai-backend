r"""
backend.py
------------
FastAPI backend for Football XAI (22-Player Graph Architecture).
High-speed GNNExplainer execution with plain-English tactical rationales.
"""

import json
import io
import os
import hashlib
import secrets
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer

from graph_builder import build_graphs_from_uploaded_match, CLASSES

MODEL_PATH = "football_gat_model.pt"
USERS_FILE = "users.json"
ROLE_NAMES = ["GK", "DEF", "MID", "FWD"]

def role_from_onehot(role_onehot):
    if float(np.sum(role_onehot)) == 0.0:
        return "Unidentified"
    return ROLE_NAMES[int(np.argmax(role_onehot))]

def generate_rationale(feat_a, feat_b, predicted_class):
    """Translates spatial tensor features into tactical pass/pressing explanations."""
    x_a, y_a = feat_a[0], feat_a[1]
    x_b, y_b = feat_b[0], feat_b[1]
    opp_a, pressure_a = feat_a[11], feat_a[13]
    opp_b, pressure_b = feat_b[11], feat_b[13]
    role_a = role_from_onehot(feat_a[6:10])
    role_b = role_from_onehot(feat_b[6:10])

    reasons = []

    if predicted_class in ("build_up", "scoring_opp"):
        if x_b > x_a + 0.05:
            reasons.append("a forward pass into an advanced attacking pocket")
        if opp_b < opp_a:
            reasons.append(f"bypassing defensive pressure ({int(opp_b)} nearby defenders vs {int(opp_a)})")
        if pressure_b < pressure_a:
            reasons.append("exploiting open space away from tight marking")
        if not reasons:
            reasons.append("a clear passing channel between positional lines")
        return f"GNNExplainer highlighted pass towards {role_b} due to " + ", ".join(reasons) + "."

    if predicted_class in ("pressing", "transition"):
        if opp_a >= 2:
            reasons.append(f"aggressive high-press trapping {role_a} ({int(opp_a)} defenders converging)")
        if pressure_a > pressure_b:
            reasons.append("closing down passing angles under heavy pressure")
        if not reasons:
            reasons.append("a rapid turnover in midfield transition")
        return f"Defensive interaction highlighted: " + ", ".join(reasons) + "."

    return f"Tactical vector established between {role_a} and {role_b} based on positioning."

active_tokens = {}

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password: str, salt: Optional[str] = None):
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return salt, digest.hex()

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

REAL_METRICS = {
    "accuracy": 0.7442,
    "macro_f1": 0.5802,
    "per_class_f1": {
        "build_up": 0.8491, "pressing": 0.2873,
        "transition": 0.4019, "scoring_opp": 0.7887,
    },
    "stability": {
        "build_up": 0.4657, "pressing": 0.4489,
        "transition": 0.4649, "scoring_opp": 0.5429,
        "overall": 0.4659,
    },
    "test_set_size": 22237,
    "training_matches": 426,
    "training_graphs": 146456,
}

class GATClassifier(nn.Module):
    def __init__(self, in_channels=15, hidden_channels=64, heads=8, num_classes=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=dropout)
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

app = FastAPI(title="Football XAI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_explainer = None

def get_model():
    global _model, _explainer
    if _model is None:
        _model = GATClassifier(in_channels=15)
        try:
            _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at '{MODEL_PATH}'."
            )
        _model.eval()
        # Epochs reduced to 30 for high-speed response while preserving explanation quality
        _explainer = Explainer(
            model=_model,
            algorithm=GNNExplainer(epochs=30, lr=0.02),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
        )
    return _model, _explainer

def get_retained_edges(edge_mask, threshold=0.4, min_edges=2, max_edges=5):
    mask_np = edge_mask.detach().cpu().numpy()
    order = np.argsort(-mask_np)
    above = np.where(mask_np >= threshold)[0]
    if min_edges <= len(above) <= max_edges:
        keep = above
    elif len(above) > max_edges:
        keep = order[:max_edges]
    else:
        keep = order[:min(min_edges, len(mask_np))]
    return sorted(keep.tolist())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register")
def register(req: RegisterRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    users = load_users()
    if email in users:
        raise HTTPException(status_code=409, detail="Account already exists.")

    salt, pw_hash = hash_password(req.password)
    users[email] = {"salt": salt, "hash": pw_hash}
    save_users(users)

    token = secrets.token_hex(24)
    active_tokens[token] = email
    return {"token": token, "email": email}

@app.post("/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    users = load_users()

    user = users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="No account found with this email.")

    _, computed_hash = hash_password(req.password, salt=user["salt"])
    if computed_hash != user["hash"]:
        raise HTTPException(status_code=401, detail="Incorrect password.")

    token = secrets.token_hex(24)
    active_tokens[token] = email
    return {"token": token, "email": email}

@app.post("/logout")
def logout(token: str):
    active_tokens.pop(token, None)
    return {"status": "logged out"}

@app.get("/model-info")
def model_info():
    return REAL_METRICS

@app.post("/analyze-match")
async def analyze_match(
    events_file: UploadFile = File(...),
    lineup_file: UploadFile = File(...),
    three_sixty_file: Optional[UploadFile] = File(None),
):
    try:
        events_json = json.loads(await events_file.read())
        lineup_json = json.loads(await lineup_file.read())
        three_sixty_json = json.loads(await three_sixty_file.read()) if three_sixty_file else None
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file format: {e}")

    windows = build_graphs_from_uploaded_match(events_json, lineup_json, three_sixty_json)
    if not windows:
        raise HTTPException(status_code=422, detail="No valid graph windows constructed.")

    model, explainer = get_model()
    results = []

    for data, id_map, window_start_sec in windows:
        if data.x.shape[1] == 14:
            zeros = torch.zeros((data.x.shape[0], 1), dtype=data.x.dtype)
            data.x = torch.cat([data.x, zeros], dim=1)

        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        explanation = explainer(data.x, data.edge_index)
        retained = get_retained_edges(explanation.edge_mask)

        edge_index_np = data.edge_index.numpy()
        highlighted_interactions = []
        for eid in retained:
            u, v = edge_index_np[:, eid]
            name_u = id_map.get(u, {}).get("player_name", f"Player {u}")
            name_v = id_map.get(v, {}).get("player_name", f"Player {v}")
            feat_u = data.x[u].numpy()
            feat_v = data.x[v].numpy()
            rationale = generate_rationale(feat_u, feat_v, CLASSES[pred_idx])
            highlighted_interactions.append({
                "player_a": name_u, "player_b": name_v,
                "node_a": int(u), "node_b": int(v),
                "rationale": rationale,
            })

        players = []
        for node_idx in range(data.x.shape[0]):
            feat = data.x[node_idx].numpy()
            role_onehot = feat[6:10]
            role = role_from_onehot(role_onehot)
            is_named = feat[14] > 0.5 if len(feat) > 14 else True
            name = id_map.get(node_idx, {}).get("player_name", f"Player {node_idx}") if is_named else "Unidentified"
            players.append({
                "node": node_idx,
                "name": name,
                "x": round(float(feat[0]), 4),
                "y": round(float(feat[1]), 4),
                "team_indicator": round(float(feat[5]), 1),
                "role": role,
                "is_named": bool(is_named),
            })

        results.append({
            "window_start_seconds": window_start_sec,
            "predicted_class": CLASSES[pred_idx],
            "confidence": round(confidence, 4),
            "class_probabilities": {c: round(float(probs[0, i]), 4) for i, c in enumerate(CLASSES)},
            "num_players_involved": data.x.shape[0],
            "highlighted_interactions": highlighted_interactions,
            "players": players,
        })

    return {
        "num_windows_analyzed": len(results),
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)