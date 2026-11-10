"""
Evolution only. Every aspect of the brain is a gene; nothing inside a life is written by hand, and
nothing inside a life changes the brain: a brain is its genome from its first tick to its last.

A genome (one per lineage) holds:
  structure   alive (each hidden neuron's own on/off, inherited with the neuron, flipped by mutation),
              thr (firing threshold: an output node spikes the tick it crosses thr from below; it must
              fall back under thr before it can spike again)
  birth       W0 (the weights), conn (each connection's own on/off: a synapse exists or it does not),
              b (bias per node), d0 (leak per node: how much of its own value a node keeps from tick to
              tick, the brain's memory)
  noise       per-node std of the random kick added to the input sum every tick
  breeding    log_mw, the mutation step for the weights
There is no plasticity: the weights do not change during a life. Memory lives in the node values,
which persist across ticks within a game and are zeroed when a game starts.

Selection is the world's: see --select. Each genome drives K identical brains (they differ only in the
noise they happen to get); a life is --life games (default 1); fitness = mean score over the last third
of life (the whole life when life is 1). The top --elite fraction of genomes carries over; the rest are
their mutated children.

    python evo_rule.py --rules 256 --brains 512 --life 1 --select survival --opponent random
Controls: STOP file (save + exit).
"""
import argparse, glob, math, os, signal, time

import numpy as np
import torch

import brain_ttt_gpu as B
import perfect as PF

DEV = B.DEV
# sensors ("vision"): for every cell one node each for empty / X / O, plus two nodes saying which
# seat I am. The brain reads them every tick of the whole game and may fire at any tick.
N_CELL_IN, N_SEAT_IN = 27, 2
N_IN = N_CELL_IN + N_SEAT_IN
N_OUT = 9
H_MAX = 82
N = N_IN + H_MAX + N_OUT
OUT_LO = N_IN + H_MAX


def encode(boards, seat):
    """boards (P, 9) int8 with 0 empty / 1 X / 2 O; seat (P,) int8 -> (P, 29) sensor values."""
    s = torch.stack([seat == 1, seat == 2], 1)
    return torch.cat([boards == 0, boards == 1, boards == 2, s], 1).float()
WIN, LOSS, ILLEGAL, DRAW, FREEZE, OPPDIED = 1, 2, 3, 0, 4, 5
# OPPDIED: the opponent made an illegal move or froze; the survivor gets a draw's score, not a win
# the world's scores (selection). Set by the user 2026-11-10: a legal move pays +1, a win on the board
# +100, an illegal move costs -1 (so firing twice on the same cell is not free), everything else 0.
EXT = {WIN: 100.0, LOSS: 0.0, ILLEGAL: -1.0, DRAW: 0.0, FREEZE: 0.0, OPPDIED: 0.0}
NCODES = 6

# scalar genes: name -> (init mean, init std, mutation step, lo, hi)
SCALARS = {
    "thr": (0.6, 0.2, 0.05, 0.05, 0.99),
    "log_mw": (math.log(0.01), 0.3, 0.3, math.log(1e-4), math.log(0.3)),   # mutation step for W0
}
SNAMES = list(SCALARS)
# vector genes (per node) and matrix genes
VECTORS = {"log_noise": (math.log(0.3), 0.6), "b": (0.0, 0.15), "d0": (0.5, 0.35)}   # d0 = per-node leak in [0,1]
ALIVE_P0 = 0.7          # a hidden neuron starts switched on with this probability
ALIVE_FLIP = 0.01       # per-neuron chance that a mutation flips it on or off (times the mutation scale)
CONN_P0 = 0.5           # a connection starts present with this probability
CONN_FLIP = 0.005       # per-connection chance that a mutation makes or breaks it (times the mutation scale)
MATRICES = {"W0": 0.1}
WCLIP = 5.0             # a synapse cannot be stronger than this (the bound that used to sit in the plasticity update)


def init_pop(R, gen):
    rn = lambda *s: torch.randn(*s, device=DEV, generator=gen)
    pop = {}
    sc = torch.zeros((R, len(SNAMES)), device=DEV)
    for i, k in enumerate(SNAMES):
        m, s, _, lo, hi = SCALARS[k]
        sc[:, i] = (m + rn(R) * s).clamp(lo, hi)
    pop["scalars"] = sc
    for k, (m, s) in VECTORS.items():
        pop[k] = m + rn(R, N) * s
    pop["d0"] = torch.rand((R, N), device=DEV, generator=gen)      # every node's leak: anywhere in [0, 1]
    # every neuron has its own on/off gene; it is inherited with the neuron and flipped by mutation.
    # Sensors and outputs are always on; only the hidden ones can be switched off.
    alive = (torch.rand((R, N), device=DEV, generator=gen) < ALIVE_P0).float()
    alive[:, :N_IN] = 1.0; alive[:, OUT_LO:] = 1.0
    pop["alive"] = alive
    # every connection has its own on/off too: a synapse is there or it is not, and mutation makes
    # and breaks them. W0 is the strength a connection has when it exists.
    pop["conn"] = (torch.rand((R, N, N), device=DEV, generator=gen) < CONN_P0).float()
    for k, s in MATRICES.items():
        pop[k] = rn(R, N, N) * s
    return pop


STRONG_FRAC = 0.10      # share of children that are "freaks": every gene mutated much harder
STRONG_MULT = 20.0      # how much harder (on top of --mutation)


def mutate(pop, idx, gen, scale=1.0):
    """Children = copies of parents idx, each gene nudged. 10% of children are freaks: every gene
    mutated 20x harder, so part of each generation lands well outside the parents' cluster.
    `scale` multiplies every mutation step (--mutation)."""
    rn = lambda *s: torch.randn(*s, device=DEV, generator=gen)
    ch = {k: v[idx].clone() for k, v in pop.items()}
    n = len(idx)
    strong = (torch.rand((n,), device=DEV, generator=gen) < STRONG_FRAC)
    boost = torch.where(strong, torch.full((n,), STRONG_MULT, device=DEV), torch.ones((n,), device=DEV)) * scale
    sc = ch["scalars"]
    step = torch.tensor([SCALARS[k][2] for k in SNAMES], device=DEV)
    hit = (torch.rand((n, len(SNAMES)), device=DEV, generator=gen) < 0.5).float()
    sc += hit * rn(n, len(SNAMES)) * step * boost[:, None]
    lo = torch.tensor([SCALARS[k][3] for k in SNAMES], device=DEV, dtype=torch.float32)
    hi = torch.tensor([SCALARS[k][4] for k in SNAMES], device=DEV, dtype=torch.float32)
    sc.clamp_(lo, hi)
    mw = sc[:, SNAMES.index("log_mw")].exp() * boost
    for k in VECTORS:
        ch[k] += rn(n, N) * 0.02 * boost[:, None]
    # a neuron switches on or off with a small chance per mutation
    flip = torch.rand((n, N), device=DEV, generator=gen) < (ALIVE_FLIP * boost[:, None]).clamp(max=0.5)
    ch["alive"] = torch.where(flip, 1.0 - ch["alive"], ch["alive"])
    ch["alive"][:, :N_IN] = 1.0; ch["alive"][:, OUT_LO:] = 1.0
    cflip = torch.rand((n, N, N), device=DEV, generator=gen) < (CONN_FLIP * boost[:, None, None]).clamp(max=0.5)
    ch["conn"] = torch.where(cflip, 1.0 - ch["conn"], ch["conn"])
    ch["d0"].clamp_(0.0, 1.0)                 # a node can keep, fade or forget, never amplify itself
    ch["log_noise"].clamp_(math.log(1e-3), math.log(3.0))   # noise std in [0.001, 3]: a physical bound, like the weight clamp
    ch["b"].clamp_(-5.0, 5.0)
    for k in MATRICES:
        ch[k] += rn(n, N, N) * mw[:, None, None]
        ch[k].clamp_(-WCLIP, WCLIP)
    return ch


def breed(pop, pa, pb, gen):
    """Children from parent pairs (pa, pb): each scalar gene from either parent; each NODE is
    inherited whole (its bias, noise, memory genes and all of its incoming weights come from the
    same parent), so crossover swaps neurons, not random synapses."""
    n = len(pa)
    A, Bp = sub_pop(pop, pa), sub_pop(pop, pb)
    ch = {}
    pick = torch.rand((n, len(SNAMES)), device=DEV, generator=gen) < 0.5
    ch["scalars"] = torch.where(pick, A["scalars"], Bp["scalars"])
    node = torch.rand((n, N), device=DEV, generator=gen) < 0.5            # which parent gives node j
    for k in list(VECTORS) + ["alive"]:
        ch[k] = torch.where(node, A[k], Bp[k])
    for k in list(MATRICES) + ["conn"]:
        ch[k] = torch.where(node[:, None, :], A[k], Bp[k])               # column j = inputs to node j
    return ch


def cat_pop(a, b):
    return {k: torch.cat([a[k], b[k]], 0) for k in a}


def sub_pop(pop, idx):
    return {k: v[idx] for k, v in pop.items()}


def migrate_nodes(pop, gen):
    """Carry a population saved with a different sensor count over to the current node layout:
    hidden and output nodes keep their genes and the weights among them; the sensor nodes and
    the weights leading out of them are fresh, since the sensors now mean something else."""
    n_old = pop["W0"].shape[-1]
    old_in = n_old - H_MAX - N_OUT
    R = pop["scalars"].shape[0]
    fresh = init_pop(R, gen)
    for k in list(VECTORS) + ["alive"]:
        v = fresh[k].clone(); v[:, N_IN:] = pop[k][:, old_in:]; pop[k] = v
    for k in list(MATRICES) + ["conn"]:
        m = fresh[k].clone(); m[:, N_IN:, N_IN:] = pop[k][:, old_in:, old_in:]; pop[k] = m
    print(f"migrated genomes from {n_old} to {N} nodes ({old_in} -> {N_IN} sensors); sensor weights are fresh", flush=True)
    return pop


def fmt_genes(pop, i):
    sc = pop["scalars"][i]
    g = {k: float(sc[j]) for j, k in enumerate(SNAMES)}
    return (f"thr={g['thr']:.2f} hid={int(pop['alive'][i, N_IN:OUT_LO].sum())} conn={100*float(pop['conn'][i].mean()):.0f}% mw={math.exp(g['log_mw']):.3f} "
            f"noise={pop['log_noise'][i].exp().mean():.2f} |W0|={pop['W0'][i].abs().mean():.3f} leak={pop['d0'][i].mean():.2f}")


class Life:
    """R genomes x K brains, one game each at a time, all in lock-step on the GPU."""

    def __init__(self, a, pop):
        self.a = a
        # perfect-player lookup (board code -> best cell) and powers of 3 for coding boards
        self.ptab = torch.tensor(PF.perfect_table(), dtype=torch.long, device=DEV)
        self.pow3 = (3 ** torch.arange(9, device=DEV)).to(torch.int64)
        self.R, self.K = pop["scalars"].shape[0], a.brains
        self.P = self.R * self.K
        self.W = torch.zeros((self.P, N, N), device=DEV)
        self.board = torch.zeros((self.P, 9), dtype=torch.int8, device=DEV)
        self.set_pop(pop)

    def set_pop(self, pop):
        K = self.K
        sc = pop["scalars"].repeat_interleave(K, 0)
        g = {k: sc[:, j] for j, k in enumerate(SNAMES)}
        self.g = g
        self.hmask = pop["alive"].repeat_interleave(K, 0).clone()   # which neurons exist in this brain
        self.hmask[:, :N_IN] = 1.0; self.hmask[:, OUT_LO:] = 1.0
        self.noise = pop["log_noise"].exp().repeat_interleave(K, 0)      # (P, N)
        self.b = pop["b"].repeat_interleave(K, 0)
        self.d0 = pop["d0"].repeat_interleave(K, 0).clamp(0.0, 1.0)     # how much of itself each node keeps per tick
        self.W0 = pop["W0"] * pop["conn"]                                # a connection that is off carries nothing

    def birth(self):
        # every brain starts from its genome's W0 with a small individual jitter (same lineage, not clones)
        self.W.copy_(self.W0.repeat_interleave(self.K, 0))       # every brain is exactly its genome
        self.state = torch.zeros((self.P, N), device=DEV)

    @torch.no_grad()
    def opp_cells(self, boards, idx, kind, perfect_mask):
        """Opponent's move for boards[idx] (as player 1 or 2 on those boards)."""
        empties = boards == 0
        rnd = (torch.rand(boards.shape, device=DEV) * empties).argmax(-1)
        if kind == "random":
            return rnd
        codes = (boards.long() * self.pow3).sum(-1)
        pf = self.ptab[codes]
        if kind == "perfect":
            return pf
        return torch.where(perfect_mask[idx], pf, rnd)      # mixed: per brain, fixed for the game

    @torch.no_grad()
    def open_test(self, idx):
        """The user's tie-break when X never opened: the O brain is re-seated as X on an empty board
        with its memory cleared and gets max_ticks ticks to make any move. Returns a bool per brain."""
        n = len(idx)
        passed = torch.zeros((n,), dtype=torch.bool, device=DEV)
        if n == 0:
            return passed
        a = self.a
        x = torch.zeros((n, N), device=DEV)
        Wi, d0, b, noise, hm = self.W[idx], self.d0[idx], self.b[idx], self.noise[idx], self.hmask[idx]
        thr = self.g["thr"][idx]
        board = torch.zeros((n, 9), dtype=torch.int8, device=DEV)
        seat = torch.ones((n,), dtype=torch.int8, device=DEV)
        over_prev = torch.zeros((n, N_OUT), dtype=torch.bool, device=DEV)
        for t in range(a.max_ticks):
            cur = d0 * x + torch.bmm(x[:, None, :], Wi).squeeze(1) + b
            cur = cur + noise * torch.randn_like(cur)
            new = torch.tanh(cur) * hm
            new[:, :N_IN] = encode(board, seat)
            over = new[:, OUT_LO:] > thr[:, None]
            rising = over & ~over_prev
            over_prev = over
            passed |= rising.any(-1)                     # the board is empty: any spike is a legal move
            x = new
            if bool(passed.all()):
                break
        return passed

    def play_game(self, gi, peer=False, kind="random", peer_frac=None):
        """One game for every brain, as one continuous stream of ticks: the sensors show the board
        every tick, the brain may fire at any tick, and the game answers like a board would.
        A fire on a taken cell, or when it is not the brain's turn, is illegal: the world ignores it
        and the game keeps ticking. Sitting on one's own turn for more than max_ticks ticks is a
        freeze (timeout), the only way a turn ends without a move.
        peer=False: each brain vs a scripted mover (`kind`; seat alternates with gi) that answers
        after a random 1..opp_delay ticks. peer=True: brains are paired at random and play each
        other; a win counts only when earned on the board (an opponent's illegal move or freeze
        gives the survivor a draw's score, OPPDIED). peer_frac (0..1) mixes the two in one game:
        that share of the brains, drawn at random, play a peer, the rest play the scripted mover.
        Returns (outcome, external score, is_peer) per brain."""
        a, P, R, K = self.a, self.P, self.R, self.K
        W, g = self.W, self.g
        self.state.zero_()
        nticks = torch.zeros((P,), device=DEV)
        if peer_frac is None:
            peer_frac = 1.0 if peer else 0.0
        # who plays whom this game: a random share of the brains are paired with each other, the
        # rest each get their own board against the scripted mover
        perm = torch.randperm(P, device=DEV)
        nP = (int(round(P * peer_frac)) // 2) * 2
        xs, os_, solo = perm[:nP // 2], perm[nP // 2:nP], perm[nP:]
        nS = len(solo)
        G = nP // 2 + nS
        # solo brains: a random seat each, so half play X and half play O in every game (seating by the
        # game index broke with one-game lives: gi was always 0 and every brain played X for 500 generations)
        solo_seat = torch.randint(1, 3, (nS,), device=DEV, dtype=torch.int8)
        game_of = torch.empty((P,), dtype=torch.long, device=DEV)
        seat = torch.empty((P,), dtype=torch.int8, device=DEV)
        partner = torch.full((P,), -1, dtype=torch.long, device=DEV)     # -1: the opponent is scripted
        game_of[xs] = torch.arange(nP // 2, device=DEV); seat[xs] = 1; partner[xs] = os_
        game_of[os_] = torch.arange(nP // 2, device=DEV); seat[os_] = 2; partner[os_] = xs
        game_of[solo] = nP // 2 + torch.arange(nS, device=DEV); seat[solo] = solo_seat
        is_peer = partner >= 0
        boardG = torch.zeros((G, 9), dtype=torch.int8, device=DEV)
        perfect_mask = (torch.rand((P,), device=DEV) < 0.5) if kind == "mixed" else None
        # the scripted opponent takes 1..opp_delay ticks to answer, like a real board game
        opp_wait = torch.randint(1, a.opp_delay + 1, (nS,), device=DEV).float()
        turn = torch.ones((G,), dtype=torch.int8, device=DEV)          # whose turn, per game (X opens)
        done = torch.zeros((P,), dtype=torch.bool, device=DEV)
        outcome = torch.zeros((P,), dtype=torch.int8, device=DEV)
        nmoves = torch.zeros((P,), device=DEV)                          # legal moves made this game
        nbad = torch.zeros((P,), device=DEV)                            # illegal fires this game (ignored by the world)
        waited = torch.zeros((P,), device=DEV)                          # ticks spent alive on the opponent's turn
        over_prev = torch.zeros((P, N_OUT), dtype=torch.bool, device=DEV)   # outputs above threshold last tick
        clock = torch.zeros((P,), device=DEV)                           # ticks sat on my own turn
        thr = g["thr"]

        def finish(idx, code, partner_code):
            outcome[idx] = code; done[idx] = True
            pidx = partner[idx]; pidx = pidx[pidx >= 0]
            outcome[pidx] = partner_code; done[pidx] = True

        def judge(idx, by_brain):
            """idx: brains whose game just received a move; by_brain: it was their own move
            (otherwise the scripted opponent's)."""
            if len(idx) == 0:
                return
            bd = boardG[game_of[idx]]
            mover = seat[idx] if by_brain else (3 - seat[idx])
            w = B.winner(bd) == mover
            full = (bd != 0).all(-1)
            if by_brain:
                finish(idx[w], WIN, LOSS)
            else:
                finish(idx[w], LOSS, WIN)
            finish(idx[full & ~w], DRAW, DRAW)
            # one cell left and filling it cannot win for the player to move: the game is a draw now,
            # so X does not collect a fifth legal move for a cell that changes nothing (seat fairness)
            eight = ((bd != 0).sum(-1) == 8) & ~w
            if eight.any():
                e_idx = idx[eight]
                bd8 = bd[eight].clone()
                empty = (bd8 == 0).to(torch.int8).argmax(-1)
                bd8[torch.arange(len(e_idx), device=DEV), empty] = (3 - mover[eight]).to(bd8.dtype)
                nowin = B.winner(bd8) == 0
                finish(e_idx[nowin], DRAW, DRAW)

        for t in range(a.game_ticks):
            active = ~done
            if not active.any():
                break
            waited += (active & (turn[game_of] != seat)).float()       # patience: alive while the other side thinks
            pre = self.state
            idx = torch.nonzero(active, as_tuple=True)[0]
            # once most brains are finished, compute only the ones still playing: gathering their
            # rows of W costs less than streaming the whole tensor every tick
            sparse = len(idx) < 0.45 * P
            sl = idx if sparse else slice(None)
            pre_s = pre[sl]
            # one persistent value per node: keep a fraction d0 of itself (leak), add what the other
            # nodes push in, plus bias and noise; squash. d0 = 0 is a plain recomputed node, d0 near 1
            # holds its value across ticks.
            cur = self.d0[sl] * pre_s + torch.bmm(pre_s[:, None, :], W[sl]).squeeze(1) + self.b[sl]
            xi = self.noise[sl] * torch.randn_like(cur)
            cur = cur + xi
            new_s = torch.tanh(cur) * self.hmask[sl]
            new_s[:, :N_IN] = encode(boardG[game_of[sl]], seat[sl])
            if sparse:
                new = pre.clone(); new[idx] = new_s      # finished brains keep their last state
            else:
                new = new_s
            nticks += active.float()
            self.state = new
            # the brain acts: an output node spikes the tick it crosses its threshold from below. It
            # cannot spike again until it has fallen back under the threshold; nothing else gates it.
            over = new[:, OUT_LO:] > thr[:, None]
            rising = over & ~over_prev
            over_prev = over
            fire = active & rising.any(-1)
            if fire.any():
                idx = torch.nonzero(fire, as_tuple=True)[0]
                # the cell that crossed is the move; if several crossed this tick, the highest one
                cells = torch.where(rising[idx], new[idx, OUT_LO:], torch.full_like(new[idx, OUT_LO:], -2.0)).argmax(-1)
                myturn = turn[game_of[idx]] == seat[idx]
                occ = boardG[game_of[idx], cells] != 0
                bad = occ | ~myturn                              # taken cell, or not my turn: illegal
                # an illegal fire does nothing: the board ignores it, the game keeps ticking, and the
                # brain's own clock keeps running. Only a legal move ends a turn.
                nbad[idx[bad]] += 1
                okm = ~bad & ~done[idx]
                ok = idx[okm]
                if len(ok):
                    boardG[game_of[ok], cells[okm]] = seat[ok]
                    nmoves[ok] += 1
                    judge(ok, True)
                    turn[game_of[ok]] = 3 - seat[ok]
            # freeze: sitting on my own turn for too long
            myturn = (turn[game_of] == seat) & ~done
            clock = torch.where(myturn, clock + 1, torch.zeros_like(clock))
            froze = myturn & (clock > a.max_ticks)
            if froze.any():
                finish(torch.nonzero(froze, as_tuple=True)[0], FREEZE, OPPDIED)
            # the scripted opponent answers once its delay has run out
            if nS:
                gsolo = game_of[solo]
                oppturn = ~done[solo] & (turn[gsolo] != seat[solo])
                opp_wait -= oppturn.float()
                mv = oppturn & (opp_wait <= 0)
                if mv.any():
                    k = torch.nonzero(mv, as_tuple=True)[0]
                    idx, gid = solo[k], gsolo[k]
                    cells = self.opp_cells(boardG[gid], idx, kind, perfect_mask)
                    boardG[gid, cells] = 3 - seat[idx]
                    judge(idx, False)
                    turn[gid] = seat[idx]
                    opp_wait[k] = torch.randint(1, a.opp_delay + 1, (len(k),), device=DEV).float()
        if not done.all():                                       # hit the hard cap: counts as a freeze
            finish(torch.nonzero(~done, as_tuple=True)[0], FREEZE, OPPDIED)
        self.last_ticks = float(nticks.mean())                   # for the log: game length in ticks
        self.last_bad = float(nbad.mean())                       # for the log: illegal fires per brain per game

        if a.select in ("survival", "pair"):
            # tournament scoring, reward only: a win on the board (--reward-win), each legal move
            # (--reward-legal), a little for an illegal attempt (--reward-attempt), a little for every
            # tick spent waiting on the opponent's turn (--reward-wait). A draw, a loss, a freeze or
            # the opponent dying first earn nothing by themselves.
            ext = a.reward_win * (outcome == WIN).float()
            ext = ext + a.reward_legal * nmoves
            ext = ext + a.reward_attempt * nbad                         # per illegal fire (negative: a small cost each)
            ext = ext + a.reward_wait * waited
        else:
            ext = torch.tensor([EXT[k] for k in range(NCODES)], device=DEV)[outcome.long()]
            ext = ext + a.reward_legal * nmoves             # the world also pays for every legal move
        if a.select == "pair":
            # head-to-head: within each pair the brain with the higher score survives (1), the other does
            # not (0), a tie splits (0.5). If X never opened (froze without a move), O is re-seated as X
            # with a cleared memory and must make a move within max_ticks: pass and O is the winner, fail
            # and both are thrown out. Solo games (exams) keep the plain score.
            has = partner >= 0
            if has.any():
                pi = partner.clamp_min(0)
                pe = ext[pi]
                res = torch.where(has, (ext > pe).float() + 0.5 * (ext == pe).float(), ext)
                # terminal events decide the pair outright: the brain that timed out loses it, a win on
                # the board takes it, a loss gives it away. Only a draw falls back to the score.
                one, zero = torch.ones_like(res), torch.zeros_like(res)
                res = torch.where(has & ((outcome == FREEZE) | (outcome == LOSS)), zero, res)
                res = torch.where(has & ((outcome == OPPDIED) | (outcome == WIN)), one, res)
                # the user's first-move rule: if X never opened at all, O has not shown it can move either.
                # O is re-seated as X with a cleared memory and gets one turn: a move wins the pair, none
                # and both are thrown out. Any later timeout is simply a loss for the brain whose clock ran out.
                xfail_o = has & (seat == 2) & (outcome[pi] == FREEZE) & (nmoves[pi] == 0)
                if xfail_o.any():
                    oi = torch.nonzero(xfail_o, as_tuple=True)[0]
                    res[oi] = self.open_test(oi).float()
                    res[partner[oi]] = 0.0
                ext = res
        return outcome, ext, is_peer


def main():
    ap = argparse.ArgumentParser()
    # defaults = the final settings (run 5): `python evo_rule.py` reproduces that run
    ap.add_argument("--rules", type=int, default=3000, help="population of genomes")
    ap.add_argument("--brains", type=int, default=512, help="brains per genome")
    ap.add_argument("--chunk", type=int, default=300, help="genomes per GPU pass; the population is evaluated in R/chunk passes per generation")
    ap.add_argument("--life", type=int, default=1, help="games per life (no plasticity: one is enough to judge a brain)")
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--max-ticks", type=int, default=30, help="ticks a brain may sit on its own turn before it freezes")
    ap.add_argument("--opp-delay", type=int, default=30, help="the scripted opponent answers after 1..this many ticks (random)")
    ap.add_argument("--game-ticks", type=int, default=400, help="hard cap on ticks per game")
    ap.add_argument("--wclip", type=float, default=5.0)
    ap.add_argument("--reward-legal", type=float, default=0.0, help="score credit per legal move (pair mode: 0, the board and the clock decide)")
    ap.add_argument("--elite", type=float, default=0.10, help="top fraction that survives and breeds")
    ap.add_argument("--mutation", type=float, default=2.5, help="multiplier on every mutation step (scalar, per-node and weight genes)")
    ap.add_argument("--reward-attempt", type=float, default=-0.001, help="score per illegal fire (the world ignores the fire; this is its cost)")
    ap.add_argument("--reward-win", type=float, default=0.0, help="score credit for a win on the board (pair mode: the win already decides the game, so 0)")
    ap.add_argument("--reward-wait", type=float, default=0.0, help="survival mode: credit per tick spent alive while it is the opponent's turn")
    ap.add_argument("--select", choices=["score", "survival", "pair"], default="pair",
                    help="score: fitness = the world's prices; survival: the reward-only tournament score; "
                         "pair: head-to-head, the higher survival score in each pair wins (1/0, tie 0.5); "
                         "if X never opened, O is re-seated as X and must move or both lose")
    ap.add_argument("--seed", type=int, default=0, help="seed for breeding and mutation (the world)")
    ap.add_argument("--noise-seed", type=int, default=None, help="pin the brains' noise stream for an exact replay; default: fresh entropy")
    ap.add_argument("--resume", default=None, help="population file or 'latest'")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--opponent", choices=["random", "perfect", "mixed", "peer", "both"], default="peer",
                    help="random: a random legal mover; perfect: the minimax player (never loses, takes the center first); "
                         "mixed: each brain faces perfect or random, 50/50 per game; peer: brains play each other; "
                         "both: games alternate between a peer and the perfect player")
    ap.add_argument("--both-kind", choices=["random", "perfect"], default="random", help="both mode: the scripted opponent the other half of the brains face")
    ap.add_argument("--exam", type=int, default=6, help="exam games vs the random mover (peer mode) and vs the perfect player, for the log only")
    ap.add_argument("--exam-every", type=int, default=5, help="run the exams every this many generations")
    a = ap.parse_args()

    gen_rng = torch.Generator(device=DEV); gen_rng.manual_seed(a.seed)      # breeding lottery: the world's, repeatable
    # the brains' noise is never repeatable: a fresh entropy seed every start, so no two runs, and no
    # two resumes, ever hand the same kicks to a node. --noise-seed pins it only for a deliberate replay.
    noise_seed = a.noise_seed if a.noise_seed is not None else int.from_bytes(os.urandom(8), "little") % (2 ** 62)
    torch.manual_seed(noise_seed)
    gen0 = 0
    pop = None
    if a.resume:
        path = a.resume
        if path == "latest":
            c = sorted(glob.glob("evo_pop_*.npz"))
            path = c[-1] if c else None
        if path and os.path.exists(path):
            d = np.load(path)
            want = ["scalars"] + list(VECTORS) + ["alive"] + list(MATRICES) + ["conn"]
            pop = {k: torch.tensor(d[k], device=DEV) for k in want if k in d.files}
            if len(pop) < len(want):                         # genes new since this save start fresh
                fresh = init_pop(pop["scalars"].shape[0], gen_rng)
                for k in want:
                    pop.setdefault(k, fresh[k])
                print("seeded from an older genome layout; dropped:", [k for k in d.files if k not in want and k not in ("generation", "fitness", "gene_names")], flush=True)
            saved_names = list(d["gene_names"]) if "gene_names" in d else list(SNAMES)
            if saved_names != SNAMES:                        # genes added since this save: fill by NAME
                fresh = init_pop(pop["scalars"].shape[0], gen_rng)["scalars"]
                cols = [pop["scalars"][:, saved_names.index(n)] if n in saved_names else fresh[:, i]
                        for i, n in enumerate(SNAMES)]
                pop["scalars"] = torch.stack(cols, 1)
            if pop["W0"].shape[-1] != N:                     # sensor layout changed: keep hidden/output nodes
                pop = migrate_nodes(pop, gen_rng)
            gen0 = int(d["generation"]) + 1
            print(f"resuming from {path} at generation {gen0}", flush=True)
    if pop is None:
        pop = init_pop(a.rules, gen_rng)
    R, K = pop["scalars"].shape[0], a.brains
    CH = a.chunk if (a.chunk > 0 and R % a.chunk == 0) else R        # genomes per GPU pass
    nch = R // CH
    life = Life(a, sub_pop(pop, torch.arange(CH, device=DEV)))       # buffers sized for one chunk
    print(f"device={DEV} evolution-only, no plasticity  genomes={R} brains/genome={K} total brains={R*K} in {nch} GPU pass(es) of {CH*K}  life={a.life} games  opponent={a.opponent}  select={a.select}{" (both: peer+" + a.both_kind + ")" if a.opponent == "both" else ""} win={a.reward_win:g} legal={a.reward_legal:g} attempt={a.reward_attempt:g} wait/tick={a.reward_wait:g} elite={a.elite} mutation x{a.mutation:g}  noise seed={noise_seed}  "
          f"continuous game: {a.max_ticks} ticks per turn, opponent answers in 1..{a.opp_delay} ticks  "
          f"nodes<={N}  scalar genes={SNAMES} vector genes={list(VECTORS) + ['alive']} matrix genes={list(MATRICES) + ['conn']}", flush=True)

    stop = {"now": False}
    def on_sig(*_): stop["now"] = True
    signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)

    def save(gen, fit, order, brain_fit, lb):
        fname = f"evo_pop_{gen:05d}.npz"
        np.savez(fname, generation=gen, fitness=fit.cpu().numpy(), gene_names=np.array(SNAMES),
                 **{k: v.cpu().numpy() for k, v in pop.items()})
        for old in sorted(glob.glob("evo_pop_*.npz"))[:-3]:
            try: os.remove(old)
            except OSError: pass
        best_rule = int(order[0])
        bb = lb * K + int(brain_fit[best_rule].argmax())            # brain index inside the chunk `life` holds
        gsc = {k: float(life.g[k][bb]) for k in SNAMES}
        np.savez(f"best_evo_{gen:05d}.npz", W=life.W[bb].cpu().numpy(), b=life.b[bb].cpu().numpy(),
                 hidden=H_MAX, hmask=life.hmask[bb].cpu().numpy(), threshold=gsc["thr"],
                 max_ticks=a.max_ticks, refractory=0, spiking=True, n_in=N_IN, continuous=True,
                 mask_illegal=False, plastic=True, eta=1.0, lam=0.8, lives=1, plastic_lives=10 ** 9,
                 wclip=a.wclip, game=gen, noise=life.noise[bb].cpu().numpy(),
                 leak=life.d0[bb].cpu().numpy())
        for old in sorted(glob.glob("best_evo_*.npz"))[:-3]:
            try: os.remove(old)
            except OSError: pass
        print(f"saved {fname} and best_evo_{gen:05d}.npz", flush=True)

    t0 = time.time()
    for gen in range(gen0, a.generations):
        tail = a.life                                # no plasticity: every game of the life counts
        # lineages are dealt into chunks afresh every generation, so a brain meets peers from the
        # whole population over time, not just its chunk-mates
        perm = torch.randperm(R, device=DEV, generator=gen_rng)
        brain_fit = torch.zeros((R, K), device=DEV)
        counts = torch.zeros((R, NCODES), device=DEV)
        counts_pf = torch.zeros((R, NCODES), device=DEV)        # "both": the perfect-player games alone
        pop_oc = torch.zeros((NCODES,), device=DEV)              # the whole population's outcomes, for the log
        pop_ticks = pop_bad = 0.0
        for ci in range(nch):
            cidx = perm[ci * CH:(ci + 1) * CH]
            life.set_pop(sub_pop(pop, cidx)); life.birth()
            tot = torch.zeros((life.P,), device=DEV)
            for gi in range(a.life):
                if a.opponent == "both":
                    # a brain must handle both worlds: every game, half the brains (at random) play a
                    # peer and the other half play the scripted opponent
                    outcome, ext, is_peer = life.play_game(gi, kind=a.both_kind, peer_frac=0.5)
                else:
                    outcome, ext, is_peer = life.play_game(gi, peer=(a.opponent == "peer"), kind=a.opponent)
                if gi >= a.life - tail:
                    tot += ext
                    oc = outcome.view(CH, K)
                    pf = (~is_peer).view(CH, K)
                    for k in range(NCODES):
                        counts[cidx, k] += (oc == k).float().sum(1)
                        if a.opponent == "both":
                            counts_pf[cidx, k] += ((oc == k) & pf).float().sum(1)
                if gi == a.life - 1:
                    for k in range(NCODES):
                        pop_oc[k] += (outcome == k).float().sum()
                    pop_ticks += life.last_ticks / nch; pop_bad += life.last_bad / nch
                if stop["now"] or os.path.exists("STOP"):
                    break
            brain_fit[cidx] = (tot / tail).view(CH, K)
            if stop["now"] or os.path.exists("STOP"):
                break
        po = pop_oc / pop_oc.sum().clamp_min(1)
        print(f"gen {gen:4d} game {a.life:3d}  all brains: win={100*po[WIN]:5.1f}% "
              f"loss={100*po[LOSS]:5.1f}% illegal={100*po[ILLEGAL]:5.1f}% "
              f"freeze={100*po[FREEZE]:5.1f}%  ticks/game={pop_ticks:4.0f}  illegal fires/brain={pop_bad:4.1f}  [{time.time()-t0:5.0f}s]", flush=True)
        fit = brain_fit.mean(1)
        # ties (common under survival selection) are broken at random, not by index
        order = torch.argsort(fit + torch.rand_like(fit) * 1e-6, descending=True)
        bi = int(order[0])
        c = counts[bi] / counts[bi].sum().clamp_min(1)
        # the chunk that holds the best genome goes back on the GPU for the exams and the save
        pos = int((perm == bi).nonzero()[0, 0]); bci = pos // CH; lb = pos % CH
        cidx = perm[bci * CH:(bci + 1) * CH]
        life.set_pop(sub_pop(pop, cidx)); life.birth()
        exam = ""
        do_exam = a.exam > 0 and gen % a.exam_every == 0
        if a.opponent == "peer" and do_exam:
            # exam: a few games against the random mover so the log stays comparable across runs
            ec = torch.zeros((CH, NCODES), device=DEV)
            for gi in range(a.exam):
                oc, _, _ = life.play_game(gi, peer=False)
                oc = oc.view(CH, K)
                for k in range(NCODES):
                    ec[:, k] += (oc == k).float().sum(1)
            e = ec[lb] / ec[lb].sum().clamp_min(1)
            exam = (f"| exam vs random: win={100*e[WIN]:5.1f}% loss={100*e[LOSS]:5.1f}% "
                    f"illegal={100*e[ILLEGAL]:5.1f}% freeze={100*e[FREEZE]:5.1f}% ")
        pf = ""
        if a.opponent == "both" and counts_pf.sum() > 0:
            # both mode: how the best genome did against the scripted half of its games
            e = counts_pf[bi] / counts_pf[bi].sum().clamp_min(1)
            pf = f"| vs {a.both_kind}: win={100*e[WIN]:5.1f}% loss={100*e[LOSS]:5.1f}% illegal={100*e[ILLEGAL]:5.1f}% draw={100*e[DRAW]:5.1f}% freeze={100*e[FREEZE]:5.1f}% "
        if a.opponent in ("peer", "random", "both") and do_exam:
            # exam against the perfect player: a fixed yardstick for the log, never a teacher
            counts_ex = torch.zeros((CH, NCODES), device=DEV)
            for gi in range(a.exam):
                oc, _, _ = life.play_game(gi, peer=False, kind="perfect")
                oc = oc.view(CH, K)
                for k in range(NCODES):
                    counts_ex[:, k] += (oc == k).float().sum(1)
            e = counts_ex[lb] / counts_ex[lb].sum().clamp_min(1)
            pf += f"| vs perfect: loss={100*e[LOSS]:5.1f}% illegal={100*e[ILLEGAL]:5.1f}% draw={100*e[DRAW]:5.1f}% freeze={100*e[FREEZE]:5.1f}% "
        kind = a.opponent
        print(f"GEN {gen:4d}  best={fit[bi]:+.3f} mean={fit.mean():+.3f} median={fit.median():+.3f}  "
              f"best genome (last {tail} {kind} games): win={100*c[WIN]:5.1f}% loss={100*c[LOSS]:5.1f}% "
              f"illegal={100*c[ILLEGAL]:5.1f}% draw={100*c[DRAW]:5.1f}% freeze={100*c[FREEZE]:5.1f}% {exam}"
              f"| oppdied={100*c[OPPDIED]:5.1f}% {pf} "
              f"{fmt_genes(pop, bi)}  [{time.time()-t0:5.0f}s]", flush=True)
        save(gen, fit, order, brain_fit, lb)
        if stop["now"] or os.path.exists("STOP"):
            try: os.remove("STOP")
            except OSError: pass
            print("exiting (STOP)", flush=True)
            return
        # breeding: the top `elite` fraction survives unchanged and is the breeding pool;
        # every other slot is a child of two pool members (neuron-wise crossover), then mutated
        n_el = max(2, int(R * a.elite))
        pool = order[:n_el]
        elite = sub_pop(pop, pool)
        pa = pool[torch.randint(0, n_el, (R - n_el,), device=DEV, generator=gen_rng)]
        pb = pool[torch.randint(0, n_el, (R - n_el,), device=DEV, generator=gen_rng)]
        children = breed(pop, pa, pb, gen_rng)
        children = mutate(children, torch.arange(R - n_el, device=DEV), gen_rng, scale=a.mutation)
        pop = cat_pop(elite, children)


if __name__ == "__main__":
    main()
