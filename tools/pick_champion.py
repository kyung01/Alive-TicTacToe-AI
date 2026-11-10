"""
Pick the champion brain from a saved population by playing it, not by its training fitness.

    python tools/pick_champion.py evo_pop_01278.npz                 # screen the top 60, confirm the top 8
    python tools/pick_champion.py evo_pop_01278.npz --top 60 --screen 100 --confirm 1000 --out champion.npz

Training fitness is the share of peer games a genome survived, a noisy, relative number. Here every
candidate plays the perfect minimax player and a random legal mover, both seats, with the training timing
(the opponent answers after 1..30 ticks), and is ranked by

    score = ( draws vs perfect  +  wins vs random  +  half the draws vs random ) / 2

The top genomes by fitness are screened with --screen games per seat per opponent, the best --keep are
re-played with --confirm games, and the winner is written in the same format as best_evo_*.npz, so
battle.py, the dashboard and tools/export_brain3d.py all read it.
"""
import argparse
import os
import random
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
import battle  # noqa: E402

N_IN, H_MAX, N_OUT = 29, 82, 9
OUT_LO = N_IN + H_MAX
PERFECT = battle.perfect_cell


def random_cell(board):
    return random.choice([i for i, v in enumerate(board.tolist()) if v == 0])


def write_brain(d, gi, path, rank):
    """One genome of a saved population -> a best_evo-format brain file."""
    names = [str(x) for x in d["gene_names"]]
    alive = d["alive"][gi].copy(); alive[:N_IN] = 1; alive[OUT_LO:] = 1
    np.savez(path, W=(d["W0"][gi] * d["conn"][gi]).astype(np.float32), b=d["b"][gi].astype(np.float32),
             hidden=H_MAX, hmask=alive.astype(np.float32), threshold=float(d["scalars"][gi][names.index("thr")]),
             max_ticks=30, refractory=0, spiking=True, n_in=N_IN, continuous=True, mask_illegal=False,
             plastic=True, eta=1.0, lam=0.8, lives=1, plastic_lives=10 ** 9, wclip=5.0, game=int(d["generation"]),
             noise=np.exp(d["log_noise"][gi]).astype(np.float32), leak=np.clip(d["d0"][gi], 0, 1).astype(np.float32),
             genome=int(gi), pop_rank=int(rank), fitness=float(d["fitness"][gi]))


def play(path, n):
    """n games per seat vs each opponent -> (score, {opponent: {outcome: share}})."""
    brain = battle.load_brain(path)
    out_lo = brain["n_in"] + brain["hidden"]
    row = {}
    for name, opponent in (("perfect", PERFECT), ("random", random_cell)):
        battle.perfect_cell = opponent                  # play_vs_perfect asks this function for the opponent's move
        c = {"win": 0, "loss": 0, "draw": 0, "timeout": 0}
        for g in range(2 * n):
            me = 1 + g % 2
            board = torch.zeros(9, dtype=torch.int8, device=battle.DEV)
            _, r, _ = battle.play_vs_perfect(brain, board, battle.Side(brain), me, out_lo)
            if r["kind"] == "win":
                c["win" if r["winner"] == me else "loss"] += 1
            else:
                c["draw" if r["kind"] == "draw" else "timeout"] += 1
        row[name] = {k: v / (2 * n) for k, v in c.items()}
    battle.perfect_cell = PERFECT
    return (row["perfect"]["draw"] + row["random"]["win"] + 0.5 * row["random"]["draw"]) / 2, row


def fmt(row):
    p, r = row["perfect"], row["random"]
    return (f"vs perfect: draw {100 * p['draw']:.1f}% loss {100 * p['loss']:.1f}% timeout {100 * p['timeout']:.1f}% | "
            f"vs random: win {100 * r['win']:.1f}% loss {100 * r['loss']:.1f}% draw {100 * r['draw']:.1f}% timeout {100 * r['timeout']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("population", help="evo_pop_*.npz")
    ap.add_argument("--top", type=int, default=60, help="genomes (by training fitness) to screen")
    ap.add_argument("--screen", type=int, default=100, help="games per seat per opponent in the screen")
    ap.add_argument("--keep", type=int, default=8, help="best screened genomes to confirm")
    ap.add_argument("--confirm", type=int, default=1000, help="games per seat per opponent to confirm")
    ap.add_argument("--out", default=None, help="output file (default: best_evo_<generation>.npz)")
    a = ap.parse_args()
    d = np.load(a.population)
    gen = int(d["generation"])
    order = np.argsort(-d["fitness"])[:a.top]
    tmp = tempfile.mkdtemp(prefix="champion-")
    screened = []
    for rank, gi in enumerate(order):
        path = os.path.join(tmp, f"r{rank:03d}_g{gi}.npz")
        write_brain(d, gi, path, rank)
        s, row = play(path, a.screen)
        screened.append((s, path, gi))
        print(f"screen  rank {rank:3d} genome {gi:5d}  score {s:.3f}  {fmt(row)}", flush=True)
    screened.sort(key=lambda t: -t[0])
    best = None
    for s0, path, gi in screened[:a.keep]:
        s, row = play(path, a.confirm)
        print(f"confirm genome {gi:5d}  score {s:.3f}  {fmt(row)}", flush=True)
        if best is None or s > best[0]:
            best = (s, path, gi, row)
    out = a.out or f"best_evo_{gen:05d}.npz"
    with open(best[1], "rb") as src, open(out, "wb") as dst:
        dst.write(src.read())
    print(f"champion: generation {gen}, genome {best[2]}, score {best[0]:.3f} -> {out}\n  {fmt(best[3])}")


if __name__ == "__main__":
    main()
