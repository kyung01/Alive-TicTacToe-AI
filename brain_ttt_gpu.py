"""
GPU version of brain_ttt.py: same fully-connected recurrent brain, same rules,
same evolution, but the whole population plays all its games at once on the GPU.

    python brain_ttt_gpu.py train --pop 4000 --games 100 --gens 2000 --kill-mistakes
    python brain_ttt_gpu.py eval  best_gpu.npz
    python brain_ttt_gpu.py play  best_gpu.npz      (uses the CPU player from brain_ttt.py)
"""
import argparse
import time

import numpy as np
import torch

from brain_ttt import (N_IN, N_OUT, WIN, DRAW, LOSS, ILLEGAL, TIMEOUT, OUTCOMES,
                       build_minimax_table, play_human)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = torch.tensor([(0, 1, 2), (3, 4, 5), (6, 7, 8),
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),
                      (0, 4, 8), (2, 4, 6)], device=DEV)
SCORE = torch.tensor([1.0, 0.5, -1.0, -1.5, -1.5], device=DEV)
MEM_CLIP = 1.0
BP_DERIV_FLOOR = 0.1
BP_GRAD_CLIP = 1.0          # per-element clip on averaged gradients (memory path can explode)


def winner(boards):
    cells = boards[..., LINES]                                  # (...,8,3)
    same = (cells[..., 0] == cells[..., 1]) & (cells[..., 1] == cells[..., 2])
    won = same & (cells[..., 0] != 0)
    return (cells[..., 0] * won).amax(-1)


def encode(boards, player):
    mine = boards == player[..., None]
    theirs = (boards != 0) & ~mine
    return torch.cat([mine, theirs], -1).float()


class Arena:
    def __init__(self, hidden, threshold, max_ticks, refractory, mask_illegal, seed=0,
                 plastic=False, lives=4, plastic_lives=3, wclip=5.0, memory=False, newgame=False,
                 feel=True, noise=0.0):
        self.H = hidden
        self.noise = noise                               # jitter added to every non-sensor node's input each tick
        self.memory = memory
        self.plastic = plastic
        self.newgame = newgame                           # optional "new game" sensor (off: the brain must infer it)
        self.feel = feel                                 # pleasure/pain sensor nodes (off: outcome only reaches the learning rule)
        extra = (2 if feel else 0) + (1 if newgame else 0)
        self.n_in = N_IN + extra if plastic else N_IN
        self.ng_idx = N_IN + (2 if feel else 0)
        self.N = self.n_in + hidden + N_OUT
        self.lives = lives                  # parallel lives per brain (plastic mode)
        self.plastic_lives = plastic_lives  # mistakes allowed before death
        self.wclip = wclip
        self.threshold = threshold
        self.max_ticks = max_ticks
        self.refractory = refractory
        self.mask_illegal = mask_illegal
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self._mm = None

    # perfect-play lookup as a dense table indexed by base-3 board code
    def minimax_table(self):
        if self._mm is None:
            tab = build_minimax_table()
            best = torch.zeros((3 ** 9, 9), dtype=torch.bool)
            for state, (_, moves) in tab.items():
                code = sum(v * 3 ** i for i, v in enumerate(state))
                for m in moves:
                    best[code, m] = True
            self._mm = best.to(DEV)
            self._pow3 = (3 ** torch.arange(9, device=DEV)).to(torch.int64)
        return self._mm

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=DEV)

    def opponent_moves(self, boards, kind):
        K = boards.shape[0]
        empties = boards == 0
        if kind == "random":
            allowed = empties
        else:
            best = self.minimax_table()
            codes = (boards.long() * self._pow3).sum(-1)
            allowed = best[codes]
            if kind == "mixed":
                use_rand = self.rand(K) < 0.5
                allowed = torch.where(use_rand[:, None], empties, allowed)
        return (self.rand(K, 9) * allowed).argmax(-1)

    @staticmethod
    def mem_read(mem, m):
        """extra input current r_i * m_i (r: per-node read gain)."""
        return mem["r"].unsqueeze(1) * m

    @staticmethod
    def mem_write(mem, m, x):
        """Fully connected memory.
        m_i = tanh( (d0_i + sum_j G_ji x_j) * m_i + sum_j M_ji x_j )   squashed like an activation"""
        factor = mem["d0"].unsqueeze(1) + x @ mem["G"]
        return torch.tanh(factor * m + x @ mem["M"])

    @torch.no_grad()
    def play(self, W, b, opponent, games, mem=None):
        P, N, G = W.shape[0], self.N, games
        board = torch.zeros((P, G, 9), dtype=torch.int8, device=DEV)
        state = torch.zeros((P, G, N), device=DEV)
        net_player = torch.where(torch.arange(G, device=DEV) % 2 == 0, 1, 2) \
            .to(torch.int8).expand(P, G).clone()
        turn = torch.ones((P, G), dtype=torch.int8, device=DEV)
        done = torch.zeros((P, G), dtype=torch.bool, device=DEV)
        outcome = torch.full((P, G), DRAW, dtype=torch.int8, device=DEV)
        ticks = torch.zeros((P, G), device=DEV)
        nmoves = torch.zeros((P, G), device=DEV)        # legal moves made by the net
        mstate = torch.zeros((P, G, N), device=DEV) if mem is not None else None
        bb = b[:, None, :]
        out_lo = N_IN + self.H
        if opponent != "random":
            self.minimax_table()

        def settle(pi, gi, cells):
            board[pi, gi, cells] = turn[pi, gi]
            w = winner(board[pi, gi])
            full = (board[pi, gi] != 0).all(-1)
            mover_is_net = turn[pi, gi] == net_player[pi, gi]
            won = w != 0
            outcome[pi[won & mover_is_net], gi[won & mover_is_net]] = WIN
            outcome[pi[won & ~mover_is_net], gi[won & ~mover_is_net]] = LOSS
            fin = won | full
            done[pi[fin], gi[fin]] = True
            turn[pi, gi] = 3 - turn[pi, gi]

        for _round in range(9):
            m = ~done & (turn != net_player)
            if m.any():
                pi, gi = torch.nonzero(m, as_tuple=True)
                settle(pi, gi, self.opponent_moves(board[pi, gi], opponent))

            m = ~done & (turn == net_player)
            if not m.any():
                break
            fired = torch.zeros((P, G), dtype=torch.bool, device=DEV)
            inp = encode(board, net_player)
            for t in range(self.max_ticks):
                active = m & ~fired
                if not active.any():
                    break
                if mem is None:
                    state = torch.tanh(torch.baddbmm(bb, state, W))
                else:
                    state = torch.tanh(torch.baddbmm(bb, state, W) + self.mem_read(mem, mstate))
                state[..., :N_IN] = inp
                if mem is not None:
                    mstate = self.mem_write(mem, mstate, state)
                if t < self.refractory:
                    continue
                out = state[..., out_lo:]
                if self.mask_illegal:
                    out = torch.where(board == 0, out, torch.full_like(out, -9.0))
                fire = (out > self.threshold).any(-1) & active
                if fire.any():
                    pi, gi = torch.nonzero(fire, as_tuple=True)
                    cells = out[pi, gi].argmax(-1)
                    fired[pi, gi] = True
                    ticks[pi, gi] = t
                    occupied = board[pi, gi, cells] != 0
                    outcome[pi[occupied], gi[occupied]] = ILLEGAL
                    done[pi[occupied], gi[occupied]] = True
                    ok = ~occupied
                    if ok.any():
                        settle(pi[ok], gi[ok], cells[ok])
                        nmoves[pi[ok], gi[ok]] += 1
                    inp = encode(board, net_player)
            froze = m & ~fired
            outcome[froze] = TIMEOUT
            ticks[froze] = self.max_ticks
            done[froze] = True
            if done.all():
                break
        self.last_sat = (mstate.abs() > 0.99).float().mean().item() if mem is not None else None
        return outcome, ticks, nmoves


    @torch.no_grad()
    def play_plastic(self, W0, b, eta, lam, opponent, games, mem=None, carry=None, start_game=0,
                     reward_legal=0.3, reward_loss=1.0, cov=False, plast=None,
                     bp=False, bp_steps=5, reward_freeze=0.5, shared=False, learn_mem=False,
                     reset_mem=False, credit_decay=0.0, reset_state=False,
                     hebb=False, baseline=True):
        """Within-lifetime learning. Each brain lives L parallel lives; in each
        life it plays `games` games in sequence with one weight matrix that it
        rewires as it goes:  e = lam*e + pre*post ;  W += eta*(pleasure-pain)*e.
        Pleasure fires after a legal move (0.3) or a win (1); pain after an
        illegal move, a freeze or a loss (1). Both decay by half each tick.
        An illegal move / freeze ends the game and costs one life-mistake;
        after `plastic_lives` mistakes the brain is dead for the rest of the life.
        Returns outcome (P,L,G), ticks (P,L,G), nmoves (P,L,G), died_at (P,L)."""
        P, N, L, G = W0.shape[0], self.N, self.lives, games
        out_lo = self.n_in + self.H
        if carry is None:
            # shared=True: ONE weight matrix per brain used by all L parallel games
            # (like a minibatch); updates are averaged over the games
            W = W0[:, None].clone() if shared else W0[:, None].expand(P, L, N, N).clone()
            e = torch.zeros_like(W) if not (bp or hebb) else torch.zeros(1, device=DEV)   # trace unused in bp/hebb mode
            xavg = torch.zeros((P, L, N), device=DEV)
        else:                       # continue a life from where the last call left it
            W, e = carry["W"], carry["e"]
            xavg = carry.get("xavg", torch.zeros((P, L, N), device=DEV))
        eta4 = eta[:, None, None, None]
        lam4 = lam[:, None, None, None]
        # per-synapse plasticity genes (P,N,N): each connection has its own learning rate
        # (sign included), so evolution decides what is plastic and what is frozen
        plast4 = plast[:, None] if plast is not None else eta4   # genes ARE the learning rates
        state = torch.zeros((P, L, N), device=DEV) if carry is None else carry["state"]
        mstate = (torch.zeros((P, L, N), device=DEV) if mem is not None else None)             if carry is None else carry["m"]
        pleasure = torch.zeros((P, L), device=DEV)
        pain = torch.zeros((P, L), device=DEV)
        newgame = torch.zeros((P, L), device=DEV)     # fires at the start of each game
        # backprop-through-time mode: keep the last bp_steps+1 activations and the
        # output each brain fired last, so a reward can flow back along the path
        # (output <- hidden <- sensors over the last ticks) that produced the action
        K = bp_steps
        hist = torch.zeros((K + 1, P, L, N), device=DEV) if bp else None
        # eligibility traces (bp + credit_decay > 0): one per synapse per game, fading per decision
        use_trace = bool((bp or hebb) and credit_decay > 0)
        TW = torch.zeros((P, L, N, N), device=DEV) if use_trace else None
        TM = TG = Tr = Td0 = None
        if use_trace and learn_mem and shared and mem is not None:
            TM = torch.zeros((P, L, N, N), device=DEV)
            if bp:                      # hebb learns W and the write matrix M only
                TG = torch.zeros((P, L, N, N), device=DEV)
                Tr = torch.zeros((P, L, N), device=DEV)
                Td0 = torch.zeros((P, L, N), device=DEV)
        # reward-modulated Hebbian rule: reward minus what the brain usually gets
        # (reward prediction error); carried across games
        rbar = carry.get("rbar", 0.0) if carry is not None else 0.0
        R = torch.zeros((P, L), device=DEV)          # hebb: each life's outcome this game
        # memory history (m after each tick) so the wave can flow through the memory path
        learn_mem = bool(learn_mem and bp and shared and mem is not None)
        mhist = torch.zeros((K + 2, P, L, N), device=DEV) if learn_mem else None
        fired_cell = torch.full((P, L), -1, dtype=torch.long, device=DEV)
        n_in_ = self.n_in
        mistakes = torch.zeros((P, L), dtype=torch.int32, device=DEV)
        dead = torch.zeros((P, L), dtype=torch.bool, device=DEV)
        died_at = torch.full((P, L), float(G), device=DEV)
        outcome = torch.full((P, L, G), DRAW, dtype=torch.int8, device=DEV)
        ticks = torch.zeros((P, L, G), device=DEV)
        nmoves = torch.zeros((P, L, G), device=DEV)
        bb = b[:, None, :]
        if opponent != "random":
            self.minimax_table()
        board = torch.zeros((P, L, 9), dtype=torch.int8, device=DEV)
        net_player = torch.ones((P, L), dtype=torch.int8, device=DEV)

        pre_out = None                      # outputs before tanh: the brain's real preference
        def tick_learn():
            nonlocal state, mstate, xavg, pre_out
            pre = state
            cur = torch.matmul(state[..., None, :], W).squeeze(-2) + bb
            if mem is not None:
                cur = cur + self.mem_read(mem, mstate)
            # pleasure / pain are ordinary nodes that ALSO receive the external signal:
            # the brain can generate (imagine) and remember them on its own
            if self.feel:
                cur[..., N_IN] += 3.0 * pleasure
                cur[..., N_IN + 1] += 3.0 * pain
            if self.noise > 0:
                # neural noise: every tick each node's input sum gets a fresh random kick, so the
                # same board can lead to different moves (the brain's own behaviour varies)
                cur = cur + self.noise * torch.randn_like(cur)
            pre_out = cur[..., out_lo:]
            new = torch.tanh(cur)
            new[..., :N_IN] = encode(board, net_player)
            if self.newgame:
                new[..., self.ng_idx] = newgame
            state = new
            if mem is not None:
                mstate = self.mem_write(mem, mstate, new)
            if hebb:
                # Hebbian trace: every tick, each connection remembers how active its two
                # neurons were together (sender a tick earlier x receiver now). No wave.
                alive = (~done).to(TW.dtype)[..., None]          # a finished game adds nothing
                TW.mul_(credit_decay).addcmul_(pre[..., :, None], (new * alive)[..., None, :])
                if TM is not None:
                    TM.mul_(credit_decay).addcmul_(new[..., :, None], (mstate * alive)[..., None, :])
                pleasure.mul_(0.5); pain.mul_(0.5); newgame.mul_(0.5)
                return
            if bp:
                # no per-tick plasticity: remember the activation and wait for a reward
                hist[:-1] = hist[1:].clone()
                hist[-1] = new
                if learn_mem:
                    mhist[:-1] = mhist[1:].clone()
                    mhist[-1] = mstate
                pleasure.mul_(0.5)
                pain.mul_(0.5)
                newgame.mul_(0.5)
                return
            if cov:
                # covariance rule: only activity above/below the node's own running
                # average earns credit, so habitual firing stops reinforcing itself
                xavg = 0.95 * xavg + 0.05 * new
                e.mul_(lam4).addcmul_(pre[..., :, None], (new - xavg)[..., None, :])
            else:
                e.mul_(lam4).addcmul_(pre[..., :, None], new[..., None, :])
            r = (pleasure - pain)[..., None, None]
            W.add_(plast4 * r * e).clamp_(-self.wclip, self.wclip)
            pleasure.mul_(0.5)
            pain.mul_(0.5)
            newgame.mul_(0.5)

        def apply_bp(pi, li, r, h=None, mh=None, cells_override=None, mode="apply", cells_direct=False):
            """Reward r (scalar or (k,) tensor) for brains (pi, li): backpropagate from
            the output they fired, through the last K ticks, and update W along that
            path scaled by each synapse's own learning-rate gene. h/mh: history
            snapshots to use instead of the current ones (crediting an earlier move)."""
            if not bp or len(pi) == 0:
                return
            hist_, mhist_ = (hist if h is None else h), (mhist if mh is None else mh)
            if cells_override is None:
                cell = fired_cell[pi, li]
            elif cells_direct:
                cell = cells_override                     # already one cell per (pi, li)
            else:
                cell = cells_override[pi, li]
            ok = cell >= 0
            if not ok.any():
                return
            pi, li, cell = pi[ok], li[ok], cell[ok]
            r = r[ok] if torch.is_tensor(r) else torch.full((len(pi),), float(r), device=DEV)
            xT = hist_[-1]
            delta = torch.zeros((P, L, N), device=DEV)
            dfloor = BP_DERIV_FLOOR          # tanh' floor so saturated nodes still pass the wave
            # the chosen output gets r; the other eight share -r so the total urge to act is
            # unchanged and only the preference shifts (prevents drift into freezing or firing)
            outs = xT[pi, li, out_lo:]                                       # (k, 9)
            dact = torch.ones_like(outs)           # seed at pre-tanh level: saturation does not mute it
            seed = torch.full_like(outs, -1.0 / (N_OUT - 1))
            seed[torch.arange(len(pi), device=DEV), cell] = 1.0
            rr = r if torch.is_tensor(r) else torch.full((len(pi),), float(r), device=DEV)
            delta[pi, li, out_lo:] = rr[:, None] * seed * dact
            deltas, xs = [], []
            if learn_mem:
                rg, d0g, Mg, Gg = mem["r"], mem["d0"], mem["M"], mem["G"]
                dm_carry = torch.zeros((P, L, N), device=DEV)   # credit flowing to m_{t-2}
                dr = torch.zeros((P, L, N), device=DEV)
                dd0 = torch.zeros((P, L, N), device=DEV)
                dM_x, dM_d, dG_d = [], [], []
            for k in range(K):
                xprev = hist_[-2 - k]
                deltas.append(delta)
                xs.append(xprev)
                dx_mem = 0.0
                if learn_mem:
                    # x_t = tanh(x_{t-1} W + b + r * m_{t-1});  m_{t-1} = (d0 + x_{t-1} G) * m_{t-2} + x_{t-1} M
                    m_prev = mhist_[-2 - k]
                    m_prev2 = mhist_[-3 - k]
                    dr += delta * m_prev                          # read gain
                    dm = delta * rg.unsqueeze(1) + dm_carry       # dL/dm_{t-1}
                    dm = dm * (1 - m_prev ** 2).clamp_min(dfloor)   # through the tanh that made m_{t-1}
                    dd0 += dm * m_prev2                           # base multiplier
                    dM_x.append(xprev); dM_d.append(dm)           # write matrix:  x_{t-1}^T dm
                    dG_d.append(dm * m_prev2)                     # multiply matrix: x_{t-1}^T (dm * m_{t-2})
                    dx_mem = (torch.matmul((dm * m_prev2)[..., None, :], Gg.transpose(-1, -2)).squeeze(-2)
                              + torch.matmul(dm[..., None, :], Mg.transpose(-1, -2)).squeeze(-2))
                    factor = d0g.unsqueeze(1) + torch.matmul(xprev[..., None, :], Gg).squeeze(-2)
                    dm_carry = dm * factor                        # onward to m_{t-2}
                # flow one tick back: delta_prev_j = (sum_i W_ji delta_i + memory path) * tanh'(x_prev_j)
                delta = (torch.matmul(delta[..., None, :], W.transpose(-1, -2)).squeeze(-2) + dx_mem) \
                    * (1 - xprev ** 2).clamp_min(dfloor)
                delta[..., :N_IN] = 0             # board sensors are clamped: the wave stops there
            X = torch.stack(xs, -2)              # (P,L,K,N) pre-synaptic activity
            D = torch.stack(deltas, -2)          # (P,L,K,N) credit arriving at post nodes
            dW = torch.matmul(X.transpose(-1, -2), D)    # (P,L,N,N) sum_k x_{t-1} (x) delta_t
            n_ev = max(1, len(pi))
            if mode == "trace":
                # fade this life's traces by credit_decay, then add the new decision's wave
                TW[pi, li] = TW[pi, li] * credit_decay + dW[pi, li]
                if TM is not None:
                    Xm = torch.stack(dM_x, -2)
                    dM_all = torch.matmul(Xm.transpose(-1, -2), torch.stack(dM_d, -2))
                    dG_all = torch.matmul(Xm.transpose(-1, -2), torch.stack(dG_d, -2))
                    TM[pi, li] = TM[pi, li] * credit_decay + dM_all[pi, li]
                    TG[pi, li] = TG[pi, li] * credit_decay + dG_all[pi, li]
                    Tr[pi, li] = Tr[pi, li] * credit_decay + dr[pi, li]
                    Td0[pi, li] = Td0[pi, li] * credit_decay + dd0[pi, li]
                return
            if shared:
                dW = dW.sum(1, keepdim=True) / n_ev   # minibatch: average over the games that had an event
            W.add_(plast4 * dW).clamp_(-self.wclip, self.wclip)
            if learn_mem:
                lr = plast4[:, 0, 0, 0].view(P, 1)
                Xm = torch.stack(dM_x, -2)
                dM = torch.matmul(Xm.transpose(-1, -2), torch.stack(dM_d, -2)).sum(1) / n_ev
                dG = torch.matmul(Xm.transpose(-1, -2), torch.stack(dG_d, -2)).sum(1) / n_ev
                clip = lambda t: t.clamp(-BP_GRAD_CLIP, BP_GRAD_CLIP).nan_to_num(0.0)
                Mg.add_(lr.unsqueeze(-1) * clip(dM))
                Gg.add_(lr.unsqueeze(-1) * clip(dG))
                rg.add_(lr * clip(dr.sum(1) / n_ev))
                d0g.add_(lr * clip(dd0.sum(1) / n_ev))

        def apply_trace(pi, li, r):
            """Game over for lives (pi, li) with outcome r: every synapse moves by
            r x its trace, averaged over those lives (shared brain)."""
            if len(pi) == 0:
                return
            n_ev = len(pi)
            dW = TW[pi, li].sum(0) * (r / n_ev)                 # (N,N)
            W.add_(plast4[:, 0] * dW.unsqueeze(0)).clamp_(-self.wclip, self.wclip)
            if TM is not None:
                lr = plast4[:, 0, 0, 0].view(P, 1)
                clip = lambda t: t.clamp(-BP_GRAD_CLIP, BP_GRAD_CLIP).nan_to_num(0.0)
                mem["M"].add_(lr.unsqueeze(-1) * clip(TM[pi, li].sum(0) * (r / n_ev)).unsqueeze(0))
            if TG is not None:
                mem["G"].add_(lr.unsqueeze(-1) * clip(TG[pi, li].sum(0) * (r / n_ev)).unsqueeze(0))
                mem["r"].add_(lr * clip(Tr[pi, li].sum(0) * (r / n_ev)).unsqueeze(0))
                mem["d0"].add_(lr * clip(Td0[pi, li].sum(0) * (r / n_ev)).unsqueeze(0))
            TW[pi, li] = 0
            if TM is not None:
                TM[pi, li] = 0
            if TG is not None:
                TG[pi, li] = 0; Tr[pi, li] = 0; Td0[pi, li] = 0

        def mistake(pi, li, g, kind):
            outcome[pi, li, g] = kind
            pain[pi, li] = 1.0
            if kind == ILLEGAL:
                if hebb:
                    R[pi, li] = -1.0
                elif TW is not None:
                    apply_trace(pi, li, -1.0)
                else:
                    apply_bp(pi, li, -1.0)
            mistakes[pi, li] += 1
            d = mistakes[pi, li] >= self.plastic_lives
            pd, ld = pi[d], li[d]
            dead[pd, ld] = True
            died_at[pd, ld] = torch.minimum(died_at[pd, ld], torch.full_like(died_at[pd, ld], float(g)))

        for g in range(G):
            board.zero_()
            newgame.fill_(1.0)
            if TW is not None:                 # new game: every synapse's trace starts clean
                TW.zero_()
                if TM is not None:
                    TM.zero_()
                if TG is not None:
                    TG.zero_(); Tr.zero_(); Td0.zero_()
            if reset_mem and mstate is not None:
                mstate.zero_()                 # stored numbers start every game at zero
            if reset_state:
                state.zero_()                  # activations too: every game starts from a blank brain
            net_player.fill_(1 if (g + start_game) % 2 == 0 else 2)
            turn = torch.ones((P, L), dtype=torch.int8, device=DEV)
            done = dead.clone()

            def settle(pi, li, cells):
                board[pi, li, cells] = turn[pi, li]
                w = winner(board[pi, li])
                full = (board[pi, li] != 0).all(-1)
                mover_is_net = turn[pi, li] == net_player[pi, li]
                won = w != 0
                nw, nl = won & mover_is_net, won & ~mover_is_net
                outcome[pi[nw], li[nw], g] = WIN
                pleasure[pi[nw], li[nw]] = 1.0
                outcome[pi[nl], li[nl], g] = LOSS
                pain[pi[nl], li[nl]] = reward_loss
                if hebb:
                    R[pi[nw], li[nw]] = 1.0
                    R[pi[nl], li[nl]] = -reward_loss
                elif TW is not None:
                    # outcome reaches every synapse in proportion to its trace
                    # (how strongly and how recently it shaped a decision this game)
                    apply_trace(pi[nw], li[nw], 1.0)
                    apply_trace(pi[nl], li[nl], -reward_loss)
                else:
                    apply_bp(pi[nw], li[nw], 1.0)
                    apply_bp(pi[nl], li[nl], -reward_loss)
                fin = won | full
                done[pi[fin], li[fin]] = True
                turn[pi, li] = 3 - turn[pi, li]

            for _round in range(9):
                m = ~done & (turn != net_player)
                if m.any():
                    pi, li = torch.nonzero(m, as_tuple=True)
                    settle(pi, li, self.opponent_moves(board[pi, li], opponent))
                m = ~done & (turn == net_player)
                if not m.any():
                    break
                fired = torch.zeros((P, L), dtype=torch.bool, device=DEV)
                for t in range(self.max_ticks):
                    active = m & ~fired
                    if not active.any():
                        break
                    tick_learn()
                    if t < self.refractory:
                        continue
                    out = state[..., out_lo:]
                    if TW is not None and bp:
                        # every thinking tick: the output the brain currently leans toward sends a
                        # wave back through what drives it; that wave joins the fading traces
                        pa_, la_ = torch.nonzero(active, as_tuple=True)
                        lead = pre_out[pa_, la_].argmax(-1)
                        mag = out[pa_, la_].max(-1).values.clamp_min(0.0)
                        apply_bp(pa_, la_, mag, mode="trace", cells_override=lead, cells_direct=True)
                    if self.mask_illegal:
                        out = torch.where(board == 0, out, torch.full_like(out, -9.0))
                    fire = (out > self.threshold).any(-1) & active
                    if fire.any():
                        pi, li = torch.nonzero(fire, as_tuple=True)
                        cells = pre_out[pi, li].argmax(-1)     # saturated outputs tie at 1.0; pre-tanh does not
                        fired[pi, li] = True
                        ticks[pi, li, g] = t
                        fired_cell[pi, li] = cells
                        occupied = board[pi, li, cells] != 0
                        if occupied.any():
                            mistake(pi[occupied], li[occupied], g, ILLEGAL)
                            done[pi[occupied], li[occupied]] = True
                        ok = ~occupied
                        if ok.any():
                            po, lo = pi[ok], li[ok]
                            if reward_legal > 0:
                                pleasure[po, lo] = torch.maximum(pleasure[po, lo],
                                                                 torch.full_like(pleasure[po, lo], reward_legal))
                                apply_bp(po, lo, reward_legal)      # a legal move is a correct decision
                            nmoves[po, lo, g] += 1
                            settle(po, lo, cells[ok])
                froze = m & ~fired
                if froze.any():
                    pi, li = torch.nonzero(froze, as_tuple=True)
                    if bp or hebb:
                        # a freeze: the output that came closest to firing should have fired
                        # (hebb has no output to seed: a freeze is simply a bad game)
                        fired_cell[pi, li] = pre_out[pi, li].argmax(-1)
                        if hebb:
                            R[pi, li] = -reward_loss
                        elif TW is not None:
                            apply_trace(pi, li, reward_freeze)
                        else:
                            apply_bp(pi, li, reward_freeze)
                    mistake(pi, li, g, TIMEOUT)
                    ticks[pi, li, g] = self.max_ticks
                    done[pi, li] = True
                if done.all():
                    break
            # a few quiet ticks between games so end-of-game reward reaches the trace
            for _ in range(self.refractory):
                tick_learn()
            if hebb:
                # judge the whole game at once: each life's outcome relative to the other
                # lives of this same game (same seat, same brain), times its Hebbian trace
                Rc = R - R.mean(1, keepdim=True) if baseline else R
                dW = torch.einsum("pl,plij->pij", Rc, TW) / L
                W.add_((plast4[:, 0] * dW).unsqueeze(1)).clamp_(-self.wclip, self.wclip)
                if TM is not None:
                    lr = plast4[:, 0, 0, 0].view(P, 1, 1)
                    dM = torch.einsum("pl,plij->pij", Rc, TM) / L
                    mem["M"].add_(lr * dM.clamp(-BP_GRAD_CLIP, BP_GRAD_CLIP).nan_to_num(0.0))
                R.zero_()
        self.last_sat = (mstate.abs() > 0.99).float().mean().item() if mem is not None else None
        if hebb:
            oc = outcome[:, :, -1]
            rew = ((oc == WIN).float() * 1.0 - (oc == LOSS).float() * reward_loss
                   - (oc == ILLEGAL).float() * 1.0 - (oc == TIMEOUT).float() * reward_loss).mean().item()
            rbar = 0.9 * rbar + 0.1 * rew
        self.carry = {"W": W, "e": e, "state": state, "m": mstate, "xavg": xavg, "rbar": rbar}
        return outcome, ticks, nmoves, died_at


def mem_init(P, N, std=0.005):
    return {"r": torch.randn((P, N), device=DEV) * 0.1,
            "d0": torch.ones((P, N), device=DEV),      # factor 1 = perfect hold by default
            "M": torch.randn((P, N, N), device=DEV) * std,
            "G": torch.randn((P, N, N), device=DEV) * std}


def mem_index(mem, idx):
    return {k: v[idx] for k, v in mem.items()}


def mem_cat(a, b):
    return {k: torch.cat([a[k], b[k]]) for k in a}


def mem_mutate(mem, rate, std):
    out = {}
    for k, v in mem.items():
        mask = torch.rand(v.shape, device=DEV) < rate
        out[k] = v + mask * torch.randn(v.shape, device=DEV) * std
    return out


def stats(outcome):
    o = outcome.flatten().long()
    o = o[o >= 0]                       # -1 marks games never played (brain was dead)
    counts = torch.bincount(o, minlength=5).float() / max(1, o.numel())
    return {k: counts[i].item() for i, k in enumerate(OUTCOMES)}


def fmt(st):
    return " ".join(f"{k}={v:5.1%}" for k, v in st.items())


def evaluate(W, b, arena, games, eta=None, lam=None, mem=None):
    for opp in ("random", "minimax"):
        if arena.plastic:
            g = max(1, games // arena.lives)
            outcome, ticks, _, died = arena.play_plastic(W, b, eta, lam, opp, g, mem)
            played = torch.arange(g, device=DEV)[None, None, :] <= died[..., None]
            outcome = torch.where(played, outcome, torch.full_like(outcome, -1))
            ticks = torch.where(played, ticks, torch.full_like(ticks, float("nan")))
            extra = f"  life={died[0].mean().item():.1f}/{g}"
        else:
            outcome, ticks, _ = arena.play(W, b, opp, games, mem)
            extra = ""
        print(f"      eval vs {opp:7s} ({outcome[0].numel()} games): {fmt(stats(outcome[0]))}  "
              f"ticks={ticks[0].nanmean().item():.1f}{extra}", flush=True)


def save(path, W, b, a, eta=None, lam=None, mem=None):
    extra = {}
    if a.plastic:
        extra = dict(plastic=True, eta=eta.item(), lam=lam.item(), lives=a.lives,
                     plastic_lives=a.plastic_lives, wclip=a.wclip)
    if a.memory:
        for k, v in mem.items():
            extra["mem_" + k] = v.cpu().numpy()
    np.savez(path, W=W.cpu().numpy(), b=b.cpu().numpy(), hidden=a.hidden,
             threshold=a.threshold, max_ticks=a.max_ticks, refractory=a.refractory,
             mask_illegal=a.mask_illegal, **extra)


def train(a):
    torch.manual_seed(a.seed)
    arena = Arena(a.hidden, a.threshold, a.max_ticks, a.refractory, a.mask_illegal, a.seed,
                  plastic=a.plastic, lives=a.lives, plastic_lives=a.plastic_lives, wclip=a.wclip,
                  memory=a.memory)
    N, P, G = arena.N, a.pop, a.games
    print(f"device={DEV}  nodes: {arena.n_in} in + {a.hidden} hidden + {N_OUT} out = {N}  "
          f"(weights per brain: {N*N})  pop={P}  games/brain/gen={a.games}  "
          f"opponent={a.opponent}  kill_mistakes={a.kill_mistakes}  plastic={a.plastic}"
          + (f" lives={a.lives}x{a.games} mistakes_allowed={a.plastic_lives}" if a.plastic else "")
          + f"  memory={a.memory}", flush=True)
    W = torch.randn((P, N, N), device=DEV) * a.init_std
    b = torch.randn((P, N), device=DEV) * 0.1
    # learning-rule genes (plastic mode): eta = exp(log_eta), lam = sigmoid(lam_logit)
    log_eta = torch.randn(P, device=DEV) * 1.0 + np.log(a.eta0)
    lam_logit = torch.randn(P, device=DEV) * 1.0 + 2.0
    # per-node memory genes (memory mode): rows = read gain r, write gain w, base decay d0, activity->decay g
    mem = mem_init(P, N, a.mem_init) if a.memory else None
    n_elite = max(2, int(P * a.elite))
    n_child = P - n_elite
    best_fit = -99.0
    t0 = time.time()
    for gen in range(1, a.gens + 1):
        eta, lam = torch.exp(log_eta), torch.sigmoid(lam_logit)
        if a.plastic:
            outcome, ticks, nmoves, died = arena.play_plastic(W, b, eta, lam, a.opponent, G, mem)
            score = SCORE[outcome.long()]
            gidx = torch.arange(G, device=DEV)[None, None, :]
            finished = gidx < died[..., None]
            moves_alive = (nmoves * (gidx <= died[..., None])).sum(-1)
            fit = (moves_alive / (G * 5.0) + (score * finished).sum(-1) / G).mean(1)
            lifespan = died.mean(1)
            # games after death were never played: drop them from the stats
            played = gidx <= died[..., None]
            outcome = torch.where(played, outcome, torch.full_like(outcome, -1)).reshape(P, -1)
            ticks = torch.where(played, ticks, torch.full_like(ticks, float("nan"))).reshape(P, -1)
        else:
            outcome, ticks, nmoves = arena.play(W, b, a.opponent, a.games, mem)
            score = SCORE[outcome.long()]
        if a.plastic:
            pass
        elif a.kill_mistakes:
            # lifespan fitness: games are played in order and the brain dies at
            # its first illegal move or freeze. Fitness = legal moves it made
            # before dying (fine gradient) + score of the games it finished.
            mistake = (outcome == ILLEGAL) | (outcome == TIMEOUT)
            died_at = torch.where(mistake.any(1), mistake.float().argmax(1),
                                  torch.full((P,), G, device=DEV))
            gidx = torch.arange(G, device=DEV)[None, :]
            finished = gidx < died_at[:, None]
            moves_alive = (nmoves * (gidx <= died_at[:, None])).sum(1)
            fit = moves_alive / (G * 5.0) + (score * finished).sum(1) / G
            lifespan = died_at.float()
        else:
            fit = score.mean(1)
            lifespan = torch.full((P,), float(G), device=DEV)
        order = torch.argsort(fit, descending=True)
        elite = order[:n_elite]
        top = order[0]
        alive = int((lifespan >= G).sum().item())
        rule = (f"  eta={eta[elite].mean().item():.4f} lam={lam[elite].mean().item():.3f}"
                if a.plastic else "")
        if mem is not None:
            me = mem_index(mem, elite)
            rule += (f"  mem|r|={me['r'].abs().mean().item():.3f} |M|={me['M'].abs().mean().item():.4f}"
                     f" |G|={me['G'].abs().mean().item():.4f} d0={me['d0'].mean().item():.3f}"
                     f" sat={arena.last_sat:.1%}")
        if gen % a.log_every == 0 or gen == 1 or gen == a.gens:
            st = stats(outcome[top])
            print(f"gen {gen:5d}  best={fit[top].item():+.3f}  mean={fit.mean().item():+.3f}  "
                  f"survivors={alive:4d}  life={lifespan.mean().item():5.1f}/{lifespan.max().item():.0f}  ticks={ticks[top].nanmean().item():5.1f}  {fmt(st)}  "
                  f"[{time.time()-t0:5.0f}s  vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB]{rule}", flush=True)
        if fit[top].item() >= best_fit:
            best_fit = fit[top].item()
            save(a.out, W[top], b[top], a, eta[top], lam[top],
                 mem_index(mem, top) if mem is not None else None)
        if gen % a.eval_every == 0 or gen == a.gens:
            evaluate(W[top:top+1], b[top:top+1], arena, a.eval_games,
                     eta[top:top+1], lam[top:top+1],
                     mem_index(mem, slice(top, top + 1)) if mem is not None else None)

        p1 = elite[torch.randint(n_elite, (n_child,), device=DEV)]
        p2 = elite[torch.randint(n_elite, (n_child,), device=DEV)]
        if a.no_crossover:
            childW = W[p1].clone()
            childb = b[p1].clone()
        else:
            cross = torch.rand((n_child, N, N), device=DEV) < 0.5
            childW = torch.where(cross, W[p1], W[p2])
            childb = torch.where(torch.rand((n_child, N), device=DEV) < 0.5, b[p1], b[p2])
        mut = torch.rand((n_child, N, N), device=DEV) < a.mut_rate
        childW += mut * torch.randn((n_child, N, N), device=DEV) * a.mut_std
        childb += (torch.rand((n_child, N), device=DEV) < a.mut_rate) * \
            torch.randn((n_child, N), device=DEV) * a.mut_std
        W = torch.cat([W[elite], childW])
        b = torch.cat([b[elite], childb])
        child_log_eta = log_eta[p1] + torch.randn(n_child, device=DEV) * 0.3
        child_lam = lam_logit[p1] + torch.randn(n_child, device=DEV) * 0.3
        log_eta = torch.cat([log_eta[elite], child_log_eta])
        lam_logit = torch.cat([lam_logit[elite], child_lam])
        if mem is not None:
            mem = mem_cat(mem_index(mem, elite), mem_mutate(mem_index(mem, p1), a.mut_rate, a.mut_std))
    print(f"saved best brain to {a.out}")


def load(path):
    d = np.load(path)
    plastic = bool(d["plastic"]) if "plastic" in d else False
    arena = Arena(int(d["hidden"]), float(d["threshold"]), int(d["max_ticks"]),
                  int(d["refractory"]), bool(d["mask_illegal"]), plastic=plastic,
                  lives=int(d["lives"]) if plastic else 4,
                  plastic_lives=int(d["plastic_lives"]) if plastic else 3,
                  wclip=float(d["wclip"]) if plastic else 5.0)
    W = torch.tensor(d["W"], device=DEV)[None]
    b = torch.tensor(d["b"], device=DEV)[None]
    eta = torch.tensor([float(d["eta"])], device=DEV) if plastic else None
    lam = torch.tensor([float(d["lam"])], device=DEV) if plastic else None
    mem = None
    if "mem_r" in d:
        mem = {k: torch.tensor(d["mem_" + k], device=DEV)[None] for k in ("r", "d0", "M", "G")}
    arena.memory = mem is not None
    return W, b, arena, eta, lam, mem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--hidden", type=int, default=82)
    t.add_argument("--pop", type=int, default=4000)
    t.add_argument("--gens", type=int, default=1000)
    t.add_argument("--games", type=int, default=100)
    t.add_argument("--opponent", default="random", choices=["random", "minimax", "mixed"])
    t.add_argument("--threshold", type=float, default=0.8)
    t.add_argument("--max-ticks", type=int, default=100)
    t.add_argument("--refractory", type=int, default=3)
    t.add_argument("--mask-illegal", action="store_true")
    t.add_argument("--kill-mistakes", action="store_true")
    t.add_argument("--elite", type=float, default=0.2)
    t.add_argument("--no-crossover", action="store_true",
                   help="children are mutated copies of one parent (no recombination)")
    t.add_argument("--mut-rate", type=float, default=0.02)
    t.add_argument("--mut-std", type=float, default=0.15)
    t.add_argument("--init-std", type=float, default=0.15)
    t.add_argument("--plastic", action="store_true",
                   help="within-lifetime reward-modulated Hebbian learning; evolution tunes the rule")
    t.add_argument("--lives", type=int, default=4, help="parallel lives per brain (plastic)")
    t.add_argument("--plastic-lives", type=int, default=3,
                   help="mistakes a plastic brain survives before dying")
    t.add_argument("--eta0", type=float, default=0.01, help="initial learning-rate gene centre")
    t.add_argument("--wclip", type=float, default=5.0)
    t.add_argument("--mem-init", type=float, default=0.005,
                   help="init std of the memory write (M) and multiply (G) matrices")
    t.add_argument("--memory", action="store_true",
                   help="every node gets a slow memory scalar; fully connected N x N write and forget matrices")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--eval-games", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="best_gpu.npz")
    e = sub.add_parser("eval")
    e.add_argument("path")
    e.add_argument("--games", type=int, default=10000)
    p = sub.add_parser("play")
    p.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        W, b, arena, eta, lam, mem = load(a.path)
        evaluate(W, b, arena, a.games, eta, lam, mem)
    else:
        d = np.load(a.path)
        if "plastic" in d or "mem_r" in d:
            raise SystemExit("play mode does not support plastic/memory brains yet; use eval")
        play_human(a.path)


if __name__ == "__main__":
    main()
