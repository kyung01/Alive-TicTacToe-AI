"""
Long-life test: take one evolved brain and let it live continuously, learning
as it plays, for thousands of games. No death, no reset. Logs a rolling window
of outcomes so you can see whether it keeps improving, plateaus, or drifts.

    python long_life.py best_full.npz --games 5000 --opponent random --window 200
"""
import argparse
import time

import numpy as np
import torch

import brain_ttt_gpu as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--opponent", default="random", choices=["random", "minimax", "mixed"])
    ap.add_argument("--window", type=int, default=200, help="games per report line")
    ap.add_argument("--lives", type=int, default=64, help="independent parallel copies of the brain")
    ap.add_argument("--log", default="long_life.log")
    a = ap.parse_args()

    W, b, arena, eta, lam, mem = B.load(a.path)
    if not arena.plastic:
        raise SystemExit("this brain has no within-life learning (--plastic); nothing to watch")
    arena.lives = a.lives
    arena.plastic_lives = 10 ** 9           # immortal
    print(f"brain {a.path}: eta={eta.item():.4f} lam={lam.item():.3f} memory={mem is not None}  "
          f"{a.lives} copies x {a.games} games vs {a.opponent}", flush=True)

    out = open(a.log, "w", encoding="utf-8")
    out.write("game win draw loss illegal timeout ticks wmean wabs sat\n")
    t0 = time.time()
    done = 0
    carry = None
    while done < a.games:
        g = min(a.window, a.games - done)
        outcome, ticks, _, _ = arena.play_plastic(W, b, eta, lam, a.opponent, g, mem, carry)
        carry = arena.carry              # weights, trace, state and memory continue
        Wl = carry["W"]
        st = B.stats(outcome)
        done += g
        line = (f"{done:6d}  win={st['win']:5.1%} draw={st['draw']:5.1%} loss={st['loss']:5.1%} "
                f"illegal={st['illegal']:5.1%} timeout={st['timeout']:5.1%}  ticks={ticks.nanmean().item():4.1f}  "
                f"|W|={Wl.abs().mean().item():.3f} sat={arena.last_sat if arena.last_sat is not None else 0:.1%}  "
                f"[{time.time()-t0:4.0f}s]")
        print(line, flush=True)
        out.write(f"{done} {st['win']:.4f} {st['draw']:.4f} {st['loss']:.4f} {st['illegal']:.4f} "
                  f"{st['timeout']:.4f} {ticks.nanmean().item():.2f} {Wl.mean().item():.4f} "
                  f"{Wl.abs().mean().item():.4f} {arena.last_sat or 0:.4f}\n")
        out.flush()


if __name__ == "__main__":
    main()
