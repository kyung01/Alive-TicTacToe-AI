"""
Continuous evolution: no generations.

Every brain in the population lives and learns continuously (plastic weights,
memory, pleasure/pain), game after game, never reset. A brain has a number of
lives; each illegal move or freeze costs one; at zero it is terminated and its
slot is filled at once by a mutated copy of a living brain chosen by age
(games survived). The copy inherits the parent's *current* learned weights,
its memory genes and learning rule, with mutation; it starts with fresh memory
and a clean record.

    python continuous.py --pop 10000 --games 2000 --lives 3 --out best_cont.npz
"""
import argparse
import os
import time

import numpy as np
import torch

import brain_ttt_gpu as B

DEV = B.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=10000)
    ap.add_argument("--games", type=int, default=2000, help="games each slot plays (total run length)")
    ap.add_argument("--lives", type=int, default=3, help="mistakes a brain survives before termination")
    ap.add_argument("--hidden", type=int, default=82)
    ap.add_argument("--opponent", default="random", choices=["random", "minimax", "mixed"])
    ap.add_argument("--memory", action="store_true", default=True)
    ap.add_argument("--no-memory", dest="memory", action="store_false")
    ap.add_argument("--tournament", type=int, default=8, help="candidates compared by age to pick a parent")
    ap.add_argument("--mut-rate", type=float, default=0.01)
    ap.add_argument("--mut-std", type=float, default=0.1)
    ap.add_argument("--init-std", type=float, default=0.15)
    ap.add_argument("--mem-init", type=float, default=0.005)
    ap.add_argument("--eta0", type=float, default=0.01)
    ap.add_argument("--eta-fixed", type=float, default=None,
                    help="make within-life learning mandatory at this rate (not evolvable)")
    ap.add_argument("--reward-legal", type=float, default=0.3, help="pleasure after a legal move (0 = none)")
    ap.add_argument("--reward-loss", type=float, default=1.0, help="pain after losing a game")
    ap.add_argument("--cov", action="store_true", help="covariance Hebbian rule (activity minus running mean)")
    ap.add_argument("--newgame-node", action="store_true", help="add a sensor that fires at the start of each game (off by default)")
    ap.add_argument("--bp", action="store_true",
                    help="backprop-through-time learning: reward flows back from the fired output along the path that caused it")
    ap.add_argument("--bp-steps", type=int, default=5, help="how many ticks back the reward wave travels")
    ap.add_argument("--reward-freeze", type=float, default=0.5,
                    help="positive wave on the nearest-to-firing output after a freeze (bp mode)")
    ap.add_argument("--plast-positive", action="store_true", help="start learning-rate genes positive")
    ap.add_argument("--plast-global", action="store_true", help="one evolvable learning rate per brain instead of per synapse")
    ap.add_argument("--feel", action="store_true", help="add pleasure/pain sensor nodes (off by default)")
    ap.add_argument("--plast-init", type=float, default=0.0005,
                    help="init std of per-synapse learning-rate genes (the only learning-rate setting)")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--max-ticks", type=int, default=100)
    ap.add_argument("--refractory", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="best_cont.npz")
    ap.add_argument("--seed-from", default=None,
                    help="npz of one brain: the whole population starts as mutated copies of it")
    ap.add_argument("--checkpoint", default="cont_state.pt", help="full population state, saved every --save-every games")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    arena = B.Arena(a.hidden, a.threshold, a.max_ticks, a.refractory, False, a.seed,
                    plastic=True, lives=1, plastic_lives=10 ** 9, memory=a.memory, newgame=a.newgame_node,
                    feel=a.feel)
    P, N = a.pop, arena.N
    print(f"device={DEV} continuous  nodes={N}  pop={P}  lives={a.lives}  opponent={a.opponent}  "
          f"memory={a.memory}  eta_fixed={a.eta_fixed}  reward_legal={a.reward_legal} reward_loss={a.reward_loss} cov={a.cov}"
          f"  bp={a.bp} bp_steps={a.bp_steps} feel={a.feel}",
          flush=True)

    # genomes
    W0 = torch.randn((P, N, N), device=DEV) * a.init_std
    b = torch.randn((P, N), device=DEV) * 0.1
    log_eta = torch.randn(P, device=DEV) + np.log(a.eta0)
    lam_logit = torch.randn(P, device=DEV) + 2.0
    mem = B.mem_init(P, N, a.mem_init) if a.memory else None
    if a.plast_global:
        # one learning-rate gene per brain (log-normal around plast_init), broadcast over all synapses
        plast = (a.plast_init * torch.exp(torch.randn(P, device=DEV) * 0.3)).view(P, 1, 1)
    else:
        plast = torch.randn((P, N, N), device=DEV) * a.plast_init     # per-synapse plasticity genes
        if a.plast_positive:
            plast = plast.abs()                                         # start in the gradient direction; may still evolve negative
    # life records
    age = torch.zeros(P, device=DEV)
    wins = torch.zeros(P, device=DEV)
    mistakes = torch.zeros(P, dtype=torch.int32, device=DEV)
    births = 0
    carry = None
    start = 0
    if a.seed_from:
        d = np.load(a.seed_from)
        Ws = torch.tensor(d["W"], device=DEV)
        bs = torch.tensor(d["b"], device=DEV)
        if Ws.shape[0] != N:
            # seed brain comes from a build with fewer input nodes: insert zero rows/cols
            # for the new inputs (right after the old inputs) so its behaviour is unchanged
            old_n = Ws.shape[0]
            k = N - old_n
            old_in = N_IN_OLD = old_n - a.hidden - B.N_OUT
            Wn = torch.zeros((N, N), device=DEV)
            idx = torch.cat([torch.arange(old_in), torch.arange(old_in + k, N)]).to(DEV)
            Wn[idx[:, None], idx[None, :]] = Ws
            bn = torch.zeros(N, device=DEV); bn[idx] = bs
            Ws, bs = Wn, bn
            print(f"seed brain padded from {old_n} to {N} nodes", flush=True)
        mutm = torch.rand((P, N, N), device=DEV) < a.mut_rate
        W0 = Ws[None].expand(P, N, N) + mutm * torch.randn((P, N, N), device=DEV) * a.mut_std
        b = bs[None].expand(P, N).clone()
        log_eta = torch.full((P,), float(np.log(max(float(d["eta"]), 1e-6))), device=DEV)             + torch.randn(P, device=DEV) * 0.3
        lam_logit = torch.logit(torch.full((P,), float(d["lam"]), device=DEV).clamp(1e-4, 1 - 1e-4))             + torch.randn(P, device=DEV) * 0.3
        if mem is not None and "mem_r" in d:
            for k in mem:
                v = torch.tensor(d["mem_" + k], device=DEV)
                if v.shape[0] != N:                    # pad for the added input nodes
                    old_n = v.shape[0]
                    old_in = old_n - a.hidden - B.N_OUT
                    idx = torch.cat([torch.arange(old_in), torch.arange(old_in + (N - old_n), N)]).to(DEV)
                    if v.dim() == 1:
                        vn = torch.zeros(N, device=DEV)
                        if k == "d0":
                            vn.fill_(1.0)
                        vn[idx] = v
                    else:
                        vn = torch.zeros((N, N), device=DEV)
                        vn[idx[:, None], idx[None, :]] = v
                    v = vn
                v = v[None].expand_as(mem[k])
                mm = torch.rand(mem[k].shape, device=DEV) < a.mut_rate
                mem[k] = v + mm * torch.randn(mem[k].shape, device=DEV) * a.mut_std
        print(f"seeded population from {a.seed_from} (age {float(d['age']) if 'age' in d else 0:.0f})", flush=True)
    if a.resume:
        ck = torch.load(a.resume, map_location=DEV)
        carry = ck["carry"]; b = ck["b"]; log_eta = ck["log_eta"]; lam_logit = ck["lam_logit"]
        if "e" not in carry:
            carry["e"] = torch.zeros_like(carry["W"])
        mem = ck["mem"]; age = ck["age"]; wins = ck["wins"]; mistakes = ck["mistakes"]
        if "plast" in ck and ck["plast"].shape == plast.shape:
            plast = ck["plast"]
        else:
            print("checkpoint has no plasticity genes; starting them fresh", flush=True)
        births = ck["births"]; start = ck["game"]
        print(f"resumed from {a.resume} at game {start}", flush=True)

    def save_checkpoint(g):
        # rotate: keep the last 3 checkpoints  (<name>.pt, <name>.1.pt, <name>.2.pt)
        import os
        base, ext = os.path.splitext(a.checkpoint)
        for i in (2, 1):
            src_ = f"{base}.{i-1}{ext}" if i > 1 else a.checkpoint
            dst_ = f"{base}.{i}{ext}"
            if os.path.exists(src_):
                os.replace(src_, dst_)
        # the eligibility trace is transient; leave it out (saves 1/4 of the file)
        slim = {k: v for k, v in carry.items() if k != "e"}
        torch.save({"carry": slim, "b": b, "log_eta": log_eta, "lam_logit": lam_logit, "mem": mem, "plast": plast,
                    "age": age, "wins": wins, "mistakes": mistakes, "births": births, "game": g},
                   a.checkpoint)
    win_hist = []
    t0 = time.time()
    counts_window = torch.zeros(5, device=DEV)
    deaths_window = 0
    oldest_ever = 0.0

    if a.eta_fixed is not None:
        log_eta = torch.full((P,), float(np.log(a.eta_fixed)), device=DEV)
    for g in range(start, a.games):
        eta, lam = torch.exp(log_eta), torch.sigmoid(lam_logit)
        outcome, ticks, _, _ = arena.play_plastic(W0, b, eta, lam, a.opponent, 1, mem, carry, start_game=g,
                                                  reward_legal=a.reward_legal, reward_loss=a.reward_loss, cov=a.cov,
                                                  plast=plast, bp=a.bp, bp_steps=a.bp_steps,
                                                  reward_freeze=a.reward_freeze)
        carry = arena.carry
        o = outcome[:, 0, 0].long()
        counts_window += torch.bincount(o, minlength=5).float()
        age += 1
        wins += (o == B.WIN).float()
        bad = (o == B.ILLEGAL) | (o == B.TIMEOUT)
        mistakes += bad.int()
        dead = mistakes >= a.lives
        nd = int(dead.sum().item())
        if nd:
            deaths_window += nd
            births += nd
            oldest_ever = max(oldest_ever, age[dead].max().item())
            di = torch.nonzero(dead, as_tuple=True)[0]
            alive = torch.nonzero(~dead, as_tuple=True)[0]
            if len(alive) == 0:          # everyone died at once: restart from random
                alive = torch.arange(P, device=DEV)
            # tournament by age among the living
            cand = alive[torch.randint(len(alive), (nd, a.tournament), device=DEV)]
            best = cand.gather(1, age[cand].argmax(1, keepdim=True)).squeeze(1)
            # inherit learned weights + genes, mutated
            def mut(x, shape_tail):
                m = torch.rand((nd,) + shape_tail, device=DEV) < a.mut_rate
                return x + m * torch.randn((nd,) + shape_tail, device=DEV) * a.mut_std
            carry["W"][di] = mut(carry["W"][best], (1, N, N))
            if a.plast_global:
                plast[di] = plast[best] * torch.exp(torch.randn((nd, 1, 1), device=DEV) * 0.3)
            else:
                pm = torch.rand((nd, N, N), device=DEV) < a.mut_rate
                plast[di] = plast[best] + pm * torch.randn((nd, N, N), device=DEV) * a.plast_init
            if carry["e"].dim() == 4:          # trace exists only in the Hebbian modes
                carry["e"][di] = 0
            carry["state"][di] = 0
            if carry["m"] is not None:
                carry["m"][di] = 0
            b[di] = mut(b[best], (N,))
            if a.eta_fixed is None:
                log_eta[di] = log_eta[best] + torch.randn(nd, device=DEV) * 0.3
            lam_logit[di] = lam_logit[best] + torch.randn(nd, device=DEV) * 0.3
            if mem is not None:
                for k in mem:
                    mem[k][di] = mut(mem[k][best], tuple(mem[k].shape[1:]))
            age[di] = 0
            wins[di] = 0
            mistakes[di] = 0

        if (g + 1) % a.log_every == 0 or g == 0:
            tot = counts_window.sum().item()
            st = {k: counts_window[i].item() / tot for i, k in enumerate(B.OUTCOMES)}
            eta, lam = torch.exp(log_eta), torch.sigmoid(lam_logit)
            top = age.argmax()
            line = (f"game {g+1:5d}  age mean={age.mean().item():6.1f} max={age.max().item():5.0f} "
                    f"ever={oldest_ever:5.0f}  deaths={deaths_window:5d}  births={births:7d}  "
                    + " ".join(f"{k}={v:5.1%}" for k, v in st.items())
                    + f"  ticks={ticks.nanmean().item():4.1f}  eta={eta.mean().item():.4f} lam={lam.mean().item():.3f}"
                    + (f"  sat={arena.last_sat:.1%}" if arena.last_sat is not None else "")
                    + f"  |W|={carry['W'].abs().mean().item():.3f} |A|={plast.abs().mean().item():.5f}"
                    f" A+={(plast > 0).float().mean().item():.0%}  [{time.time()-t0:5.0f}s  "
                    f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB]")
            print(line, flush=True)
            counts_window.zero_()
            deaths_window = 0
        stop_req = os.path.exists("STOP")
        save_req = os.path.exists("SAVE")
        if (g + 1) % a.save_every == 0 or g == a.games - 1 or stop_req or save_req:
            top = int(age.argmax().item())
            extra = {}
            if mem is not None:
                for k, v in mem.items():
                    extra["mem_" + k] = v[top].cpu().numpy()
            extra["plast"] = plast[top].cpu().numpy()
            np.savez(a.out, W=carry["W"][top, 0].cpu().numpy(), b=b[top].cpu().numpy(),
                     hidden=a.hidden, threshold=a.threshold, max_ticks=a.max_ticks,
                     refractory=a.refractory, mask_illegal=False, plastic=True,
                     eta=float(torch.exp(log_eta[top])), lam=float(torch.sigmoid(lam_logit[top])),
                     lives=1, plastic_lives=a.lives, wclip=5.0, age=float(age[top]), **extra)
            save_checkpoint(g + 1)
            if save_req:
                os.remove("SAVE")
                print(f"checkpoint saved (SAVE request) at game {g+1}", flush=True)
            if stop_req:
                os.remove("STOP")
                print(f"checkpoint saved, exiting (STOP request) at game {g+1}", flush=True)
                return
    print(f"saved oldest brain to {a.out}")


if __name__ == "__main__":
    main()
