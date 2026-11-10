"""
One brain, many parallel games, averaged backprop. No evolution, no death.

The brain plays `--batch` games at once (each game has its own activation and
memory state) but all share ONE weight matrix. After every judged decision
(illegal move, legal move, win, loss, freeze) the backward waves from all games
are averaged and applied once, like a minibatch gradient step.

    python single.py --batch 4096 --games 5000 --lr 0.01
"""
import argparse
import glob
import os
import signal
import time

import numpy as np
import torch

import brain_ttt_gpu as B

DEV = B.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4096, help="parallel games sharing the one brain")
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=82)
    ap.add_argument("--opponent", default="random", choices=["random", "minimax", "mixed"])
    ap.add_argument("--memory", action="store_true", default=True)
    ap.add_argument("--no-memory", dest="memory", action="store_false")
    ap.add_argument("--reward-legal", type=float, default=0.5)
    ap.add_argument("--reward-loss", type=float, default=0.5)
    ap.add_argument("--reward-freeze", type=float, default=0.5)
    ap.add_argument("--bp-steps", type=int, default=5)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--mem-init", type=float, default=0.005)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--max-ticks", type=int, default=30)
    ap.add_argument("--refractory", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="best_single.npz")
    ap.add_argument("--init-from", default=None, help="continue from a saved brain (.npz)")
    ap.add_argument("--minutes", type=float, default=None, help="planned exit: save and stop after this many minutes")
    ap.add_argument("--learn-mem", action="store_true", help="backprop also through the memory path (write/multiply/read/base genes learn)")
    ap.add_argument("--credit-decay", type=float, default=0.9,
                    help="synaptic trace fade per tick; outcomes reward/punish synapses by their trace (0 = last decision only)")
    ap.add_argument("--reset-mem", action="store_true", default=True, help="zero the stored numbers at the start of every game (default)")
    ap.add_argument("--keep-mem", dest="reset_mem", action="store_false", help="carry stored numbers across games")
    ap.add_argument("--reset-state", action="store_true", default=True, help="zero all activations at the start of every game (default)")
    ap.add_argument("--keep-state", dest="reset_state", action="store_false", help="carry activations across games")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="std of the random kick added to every non-sensor node's input each tick (0 = deterministic brain)")
    ap.add_argument("--rule", choices=["bp", "hebb"], default="bp",
                    help="bp: backward wave from the leaning output each tick; hebb: reward-modulated Hebbian (co-activation x outcome)")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false", help="hebb: use raw outcome instead of outcome minus running mean")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="multiply weights by (1 - this) after each update; 0 = off")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    arena = B.Arena(a.hidden, a.threshold, a.max_ticks, a.refractory, False, a.seed,
                    plastic=True, lives=a.batch, plastic_lives=10 ** 9, memory=a.memory, feel=False,
                    noise=a.noise)
    N = arena.N
    print(f"device={DEV} single brain  nodes={N}  batch={a.batch} games  lr={a.lr}  opponent={a.opponent}  "
          f"memory={a.memory} learn_mem={a.learn_mem} reset_mem={a.reset_mem} reset_state={a.reset_state} rule={a.rule} noise={a.noise} credit_decay={a.credit_decay}  rewards legal={a.reward_legal} loss={a.reward_loss} freeze={a.reward_freeze}",
          flush=True)
    W0 = torch.randn((1, N, N), device=DEV) * a.init_std
    b = torch.randn((1, N), device=DEV) * 0.1
    eta = torch.ones(1, device=DEV)
    lam = torch.full((1,), 0.8, device=DEV)
    mem = B.mem_init(1, N, a.mem_init) if a.memory else None
    if a.init_from:
        if a.init_from == "latest":
            base, ext = os.path.splitext(a.out)
            cands = sorted(glob.glob(f"{base}_*{ext}"))
            if not cands:
                raise SystemExit("no saved brain found for --init-from latest")
            a.init_from = cands[-1]
        d = np.load(a.init_from)
        W0 = torch.tensor(d["W"], device=DEV)[None]
        b = torch.tensor(d["b"], device=DEV)[None]
        if mem is not None and "mem_r" in d:
            mem = {k: torch.tensor(d["mem_" + k], device=DEV)[None] for k in ("r", "d0", "M", "G")}
        step0 = int(d["game"]) if "game" in d else 0          # cumulative training step across restarts
        print(f"continuing from {a.init_from} (step {step0})", flush=True)
    plast = torch.full((1, 1, 1), a.lr, device=DEV)
    if not a.init_from:
        step0 = 0

    carry = None
    counts = torch.zeros(5, device=DEV)
    t0 = time.time()
    # graceful early exit: Ctrl+C / SIGTERM / SIGBREAK set a flag, the loop saves and leaves
    want_exit = {"flag": False, "why": ""}
    def _sig(signum, frame):
        want_exit["flag"] = True
        want_exit["why"] = f"signal {signum}"
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _sig)
    lr_now = a.lr
    for g in range(a.games):
        # live learning-rate control: a file named LR holding a number
        if os.path.exists("LR"):
            try:
                new_lr = float(open("LR").read().strip())
                if new_lr != lr_now:
                    print(f"learning rate {lr_now:g} -> {new_lr:g} (LR file) at game {g+1}", flush=True)
                    lr_now = new_lr
                    plast.fill_(lr_now)
            except ValueError:
                pass
            os.remove("LR")
        outcome, ticks, _, _ = arena.play_plastic(W0, b, eta, lam, a.opponent, 1, mem, carry, start_game=g,
                                                  reward_legal=a.reward_legal, reward_loss=a.reward_loss,
                                                  cov=False, plast=plast, bp=(a.rule == "bp"), hebb=(a.rule == "hebb"), baseline=a.baseline, bp_steps=a.bp_steps,
                                                  reward_freeze=a.reward_freeze, shared=True,
                                                  learn_mem=a.learn_mem, reset_mem=a.reset_mem, reset_state=a.reset_state,
                                                  credit_decay=a.credit_decay)
        carry = arena.carry
        if a.weight_decay > 0:
            carry["W"].mul_(1.0 - a.weight_decay)
        counts += torch.bincount(outcome[0, :, 0].long(), minlength=5).float()
        if (g + 1) % a.log_every == 0 or g == 0:
            tot = counts.sum().item()
            st = {k: counts[i].item() / tot for i, k in enumerate(B.OUTCOMES)}
            W = carry["W"]
            print(f"game {g+1:5d}  " + " ".join(f"{k}={v:5.1%}" for k, v in st.items())
                  + f"  step={step0 + g + 1}"
                  + f"  lr={lr_now:g}"
                  + (f"  |M|={mem['M'].abs().mean().item():.4f} |G|={mem['G'].abs().mean().item():.4f} |r|={mem['r'].abs().mean().item():.3f}" if (a.learn_mem and mem is not None) else "")
                  + f"  ticks={ticks.nanmean().item():4.1f}  |W|={W.abs().mean().item():.3f} "
                  f"Wmax={W.abs().max().item():.2f}"
                  + (f"  sat={arena.last_sat:.1%}" if arena.last_sat is not None else "")
                  + f"  [{time.time()-t0:5.0f}s]", flush=True)
            counts.zero_()
        # exits: STOP file, signal, or planned time limit -> save and leave. SAVE file -> save and continue
        if os.path.exists("STOP"):
            want_exit["flag"] = True; want_exit["why"] = "STOP file"
        if a.minutes is not None and (time.time() - t0) / 60 >= a.minutes:
            want_exit["flag"] = True; want_exit["why"] = f"{a.minutes:g} minute limit"
        stop_req = want_exit["flag"]
        save_req = os.path.exists("SAVE")
        if (g + 1) % a.save_every == 0 or g == a.games - 1 or stop_req or save_req:
            extra = {}
            if mem is not None:
                for k, v in mem.items():
                    extra["mem_" + k] = v[0].cpu().numpy()
            # numbered saves (<out>_<step>.npz); keep the newest three. A new file is
            # never written over one the dashboard may be reading, so no lock errors.
            base, ext = os.path.splitext(a.out)
            step_now = step0 + g + 1
            fname = f"{base}_{step_now:07d}{ext}"
            np.savez(fname, W=carry["W"][0, 0].cpu().numpy(), b=b[0].cpu().numpy(),
                     hidden=a.hidden, threshold=a.threshold, max_ticks=a.max_ticks,
                     refractory=a.refractory, mask_illegal=False, plastic=True,
                     eta=1.0, lam=0.8, lives=1, plastic_lives=10 ** 9, wclip=5.0, game=step_now,
                     noise=a.noise, **extra)
            for old in sorted(glob.glob(f"{base}_*{ext}"))[:-3]:
                try:
                    os.remove(old)
                except OSError:
                    pass
            if save_req:
                os.remove("SAVE")
                print(f"saved brain to {a.out} (SAVE request) at game {g+1}", flush=True)
            if stop_req:
                if os.path.exists("STOP"):
                    os.remove("STOP")
                print(f"saved brain to {a.out} and exiting ({want_exit['why']}) at game {g+1}", flush=True)
                return
    print(f"saved brain to {a.out}")


if __name__ == "__main__":
    main()
