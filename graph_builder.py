r"""
graph_builder.py
-------------------
Reusable graph construction logic, adapted from create_all_graphs_360.py
to work on in-memory uploaded match data (rather than reading files by
match_id from a local StatsBomb archive). Used by backend.py to turn a
user's own uploaded match into graphs the trained model can classify.

This must stay logically IDENTICAL to create_all_graphs_360.py's graph
construction (same feature layout, same edge rules) -- otherwise the
trained model, which learned on graphs from that script, will receive
inputs it wasn't trained to understand.
"""

import numpy as np
import torch
from torch_geometric.data import Data

WINDOW_SECONDS = 5
MIN_EVENTS_PER_WINDOW = 2
MIN_PLAYERS_PER_WINDOW = 3
PROXIMITY_THRESHOLD_M = 10.0
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
NEARBY_RADIUS_M = 5.0

CLASSES = ["build_up", "pressing", "transition", "scoring_opp"]

ROLE_MAP = {
    "Goalkeeper": "GK",
    "Center Back": "DEF", "Left Back": "DEF", "Right Back": "DEF",
    "Left Center Back": "DEF", "Right Center Back": "DEF",
    "Left Wing Back": "DEF", "Right Wing Back": "DEF",
    "Center Defensive Midfield": "MID", "Left Defensive Midfield": "MID",
    "Right Defensive Midfield": "MID", "Center Midfield": "MID",
    "Left Midfield": "MID", "Right Midfield": "MID",
    "Center Attacking Midfield": "MID", "Left Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Left Wing": "FWD", "Right Wing": "FWD", "Center Forward": "FWD",
    "Striker": "FWD", "Secondary Striker": "FWD",
}
ROLE_ORDER = ["GK", "DEF", "MID", "FWD"]


def role_bucket(position_name):
    return ROLE_MAP.get(position_name, "MID")


def role_one_hot(role):
    vec = [0, 0, 0, 0]
    if role in ROLE_ORDER:
        vec[ROLE_ORDER.index(role)] = 1
    return vec


def build_player_role_lookup(lineup_json):
    role_lookup = {}
    for team in lineup_json:
        for player in team.get("lineup", []):
            positions = player.get("positions", [])
            pos_name = positions[-1]["position"] if positions else None
            role_lookup[player["player_id"]] = role_bucket(pos_name)
    return role_lookup


def event_time_seconds(event):
    return event["minute"] * 60 + event["second"]


def label_window(events_in_window):
    types = [e.get("type", {}).get("name") for e in events_in_window]
    if "Shot" in types:
        return "scoring_opp"
    if sum(1 for t in types if t == "Pressure") >= 3:
        return "pressing"
    possession_teams = [e.get("possession_team", {}).get("id")
                         for e in events_in_window if e.get("possession_team")]
    if len(set(possession_teams)) > 1:
        return "transition"
    return "build_up"


def compute_360_context(freeze_frame, actor_xy):
    if not freeze_frame or actor_xy is None:
        return 0.0, 0.0, 0.0
    ax, ay = actor_xy
    opp_count, mate_count, pressure_score = 0.0, 0.0, 0.0
    for p in freeze_frame:
        loc = p.get("location")
        if loc is None or p.get("actor"):
            continue
        dist = float(np.linalg.norm([loc[0] - ax, loc[1] - ay]))
        if dist > NEARBY_RADIUS_M:
            continue
        if p.get("teammate"):
            mate_count += 1.0
        else:
            opp_count += 1.0
            pressure_score += 1.0 / (1.0 + dist)
    return opp_count, mate_count, pressure_score


def make_graph_for_window(events_in_window, role_lookup, three_sixty_lookup):
    """Returns (Data, player_id_map) or (None, None) if the window doesn't
    have enough players/edges to form a valid graph. player_id_map maps
    graph-local node index -> original StatsBomb player_id AND player name,
    so the API response can report real player names back to the user."""
    player_frames = {}
    ball_positions = []
    player_360_context = {}
    player_names = {}

    for e in events_in_window:
        loc = e.get("location")
        if loc is None:
            continue
        t = event_time_seconds(e)
        ball_positions.append((t, loc[0], loc[1]))

        player = e.get("player")
        if player is None:
            continue
        pid = player["id"]
        player_names[pid] = player.get("name", f"Player {pid}")
        player_frames.setdefault(pid, []).append((t, loc[0], loc[1]))

        event_uuid = e.get("id")
        freeze_frame = three_sixty_lookup.get(event_uuid)
        if freeze_frame:
            ctx = compute_360_context(freeze_frame, (loc[0], loc[1]))
            player_360_context.setdefault(pid, []).append(ctx)

    if len(player_frames) < MIN_PLAYERS_PER_WINDOW or not ball_positions:
        return None, None

    ball_xy = np.mean([[x, y] for _, x, y in ball_positions], axis=0)

    node_features = []
    node_ids = {}
    team_of_event = {}
    for e in events_in_window:
        p, tm = e.get("player"), e.get("team")
        if p and tm:
            team_of_event[p["id"]] = tm["id"]
    team_ids_seen = sorted(set(team_of_event.values()))

    for pid, frames in player_frames.items():
        frames_sorted = sorted(frames, key=lambda f: f[0])
        (t0, x0, y0), (t1, x1, y1) = frames_sorted[0], frames_sorted[-1]
        dt = max(t1 - t0, 1e-3)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        dist_to_ball = float(np.linalg.norm([x1 - ball_xy[0], y1 - ball_xy[1]]))

        team_id = team_of_event.get(pid)
        team_indicator = float(team_ids_seen.index(team_id)) if team_id is not None and team_ids_seen else 0.0

        role = role_lookup.get(pid, "MID")
        role_vec = role_one_hot(role)
        possession_flag = 1.0 if pid in team_of_event else 0.0

        contexts = player_360_context.get(pid, [])
        if contexts:
            opp_5m = float(np.mean([c[0] for c in contexts]))
            mate_5m = float(np.mean([c[1] for c in contexts]))
            pressure_score = float(np.mean([c[2] for c in contexts]))
        else:
            opp_5m, mate_5m, pressure_score = 0.0, 0.0, 0.0

        feat = [
            x1 / PITCH_LENGTH, y1 / PITCH_WIDTH, vx, vy, dist_to_ball,
            team_indicator, *role_vec, possession_flag,
            opp_5m, mate_5m, pressure_score,
        ]
        node_ids[pid] = len(node_features)
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index, edge_attr = [], []
    pids = list(node_ids.keys())
    positions = {pid: (node_features[node_ids[pid]][0] * PITCH_LENGTH,
                        node_features[node_ids[pid]][1] * PITCH_WIDTH) for pid in pids}

    for i, pid_a in enumerate(pids):
        for pid_b in pids[i + 1:]:
            xa, ya = positions[pid_a]
            xb, yb = positions[pid_b]
            dist = float(np.linalg.norm([xa - xb, ya - yb]))
            if dist <= PROXIMITY_THRESHOLD_M:
                w = 1.0 / (1.0 + dist)
                a, b = node_ids[pid_a], node_ids[pid_b]
                edge_index += [[a, b], [b, a]]
                edge_attr += [[w], [w]]

    for e in events_in_window:
        etype = e.get("type", {}).get("name")
        player = e.get("player")
        if player is None:
            continue
        pid_from = player["id"]
        if pid_from not in node_ids:
            continue
        if etype == "Pass":
            pass_info = e.get("pass", {})
            recipient = pass_info.get("recipient")
            if recipient and pass_info.get("outcome") is None:
                pid_to = recipient["id"]
                if pid_to in node_ids:
                    edge_index.append([node_ids[pid_from], node_ids[pid_to]])
                    edge_attr.append([1.0])
        elif etype == "Pressure":
            xa, ya = positions[pid_from]
            best_pid, best_dist = None, float("inf")
            for pid_b in pids:
                if pid_b == pid_from:
                    continue
                xb, yb = positions[pid_b]
                d = float(np.linalg.norm([xa - xb, ya - yb]))
                if d < best_dist:
                    best_dist, best_pid = d, pid_b
            if best_pid is not None:
                edge_index.append([node_ids[pid_from], node_ids[best_pid]])
                edge_attr.append([1.0 / (1.0 + best_dist)])

    if len(edge_index) == 0:
        return None, None

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    label = label_window(events_in_window)

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t,
                y=torch.tensor([CLASSES.index(label)], dtype=torch.long))

    id_map = {v: {"player_id": k, "player_name": player_names.get(k, f"Player {k}")}
              for k, v in node_ids.items()}
    return data, id_map


def build_graphs_from_uploaded_match(events_json, lineup_json, three_sixty_json=None):
    """
    Main entry point for the backend. Takes parsed JSON content (already
    loaded from the uploaded files, not file paths) and returns a list of
    (Data, id_map, window_start_seconds) tuples, one per valid 5-second
    window found in the match.
    """
    role_lookup = build_player_role_lookup(lineup_json)
    three_sixty_lookup = {}
    if three_sixty_json:
        three_sixty_lookup = {f["event_uuid"]: f.get("freeze_frame", []) for f in three_sixty_json}

    events_with_time = [e for e in events_json if "minute" in e and "second" in e]
    if not events_with_time:
        return []

    max_t = max(event_time_seconds(e) for e in events_with_time)
    n_windows = int(max_t // WINDOW_SECONDS) + 1
    buckets = [[] for _ in range(n_windows)]
    for e in events_with_time:
        idx = int(event_time_seconds(e) // WINDOW_SECONDS)
        buckets[idx].append(e)

    results = []
    for window_idx, window_events in enumerate(buckets):
        if len(window_events) < MIN_EVENTS_PER_WINDOW:
            continue
        data, id_map = make_graph_for_window(window_events, role_lookup, three_sixty_lookup)
        if data is not None:
            results.append((data, id_map, window_idx * WINDOW_SECONDS))

    return results
