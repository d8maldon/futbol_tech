"""Train an Expected Threat (xT) model from the extracted moves and shots.

Markov reward process on a 16x12 pitch grid, solved by value iteration:
  xT(c) = p_shot(c) * p_goal_given_shot(c)
        + p_move(c) * sum_c' T(c -> c') * xT(c')
Transition and shot/goal probabilities are estimated from the data.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
NX, NY = 16, 12  # pitch is 120 x 80 in StatsBomb coordinates


def cell(x, y):
    cx = np.clip((np.asarray(x) / 120.0 * NX).astype(int), 0, NX - 1)
    cy = np.clip((np.asarray(y) / 80.0 * NY).astype(int), 0, NY - 1)
    return cx * NY + cy


def main():
    moves = pd.read_csv(os.path.join(ROOT, "moves.csv"))
    shots = pd.read_csv(os.path.join(ROOT, "shots.csv")).dropna(subset=["x", "y"])
    # AUDIT-DRAFT #17: the "open play only" claim is overstated. This filter
    # drops shootout kicks and in-game penalties (pen == 0), but free-kick and
    # corner shots are STILL retained here -- so the xT grid is "non-penalty,
    # non-shootout", not strictly open play. analyze.py's optional --strict mode
    # restricts the momentum/threat analysis to true open play; the xT model
    # itself is left unchanged (committed model output must not move).
    shots = shots[(shots.period <= 4) & (shots.pen == 0)]
    ncells = NX * NY

    start = cell(moves.sx, moves.sy)
    end = cell(moves.ex, moves.ey)
    shot_c = cell(shots.x, shots.y)
    completed = moves.ok.values == 1

    # every move attempt counts; failed attempts are absorbed (turnover, value 0)
    move_counts = np.zeros(ncells)
    np.add.at(move_counts, start, 1)
    shot_counts = np.zeros(ncells)
    np.add.at(shot_counts, shot_c, 1)
    goal_counts = np.zeros(ncells)
    np.add.at(goal_counts, shot_c, shots.goal.values)

    total = move_counts + shot_counts
    total[total == 0] = 1
    p_shot = shot_counts / total
    p_move = move_counts / total
    p_goal = np.divide(goal_counts, shot_counts,
                       out=np.zeros(ncells), where=shot_counts > 0)

    T = np.zeros((ncells, ncells))
    np.add.at(T, (start[completed], end[completed]), 1)
    attempts = move_counts.copy()
    attempts[attempts == 0] = 1
    T /= attempts[:, None]  # rows sum to completion rate, not 1

    xt = np.zeros(ncells)
    for i in range(100):
        new = p_shot * p_goal + p_move * (T @ xt)
        delta = np.abs(new - xt).max()
        xt = new
        if delta < 1e-8:
            break

    grid = xt.reshape(NX, NY)
    np.save(os.path.join(ROOT, "xt_grid.npy"), grid)
    meta = {
        "grid": [NX, NY],
        "n_moves": int(len(moves)),
        "n_shots": int(len(shots)),
        "iterations": i + 1,
        "max_xt": float(grid.max()),
        "xt_at_center": float(grid[NX // 2, NY // 2]),
    }
    with open(os.path.join(ROOT, "xt_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
