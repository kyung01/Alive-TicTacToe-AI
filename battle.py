"""
Self-play: the latest saved brain plays tic-tac-toe against a copy of itself.
Both sides share the same weights but have their own activations and stored
numbers. Returns the move list for animation.

    python battle.py best_single.npz
"""
import os
import sys

import numpy as np
import torch

import brain_ttt_gpu as B
import perfect as PF

DEV = B.DEV
_cache = {"path": None, "mtime": None, "brain": None}


def latest_path(path):
    """'latest' -> newest numbered save (best_single_0000123.npz), else the path itself."""
    if path != "latest":
        return path
    import glob
    cands = glob.glob("best_single_*.npz") + glob.glob("best_evo_*.npz")
    cands = sorted(cands, key=os.path.getmtime)          # newest save of either trainer
    return cands[-1] if cands else "best_single.npz"


def load_brain(path):
    path = latest_path(path)
    mtime = os.path.getmtime(path)
    if _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["brain"]
    with np.load(path) as d:          # close the handle at once so the trainer can rotate saves
        d = {k: d[k] for k in d.files}
    hidden = int(d["hidden"])
    N = d["W"].shape[0]
    n_in = N - hidden - B.N_OUT
    brain = {
        "W": torch.tensor(d["W"], device=DEV), "b": torch.tensor(d["b"], device=DEV),
        "hidden": hidden, "N": N, "n_in": n_in,
        "threshold": float(d["threshold"]), "max_ticks": int(d["max_ticks"]),
        "refractory": int(d["refractory"]),
        "mem": ({k: torch.tensor(d["mem_" + k], device=DEV) for k in ("r", "d0", "M", "G")}
                if "mem_r" in d else None),
        "step": int(d["game"]) if "game" in d else 0,
        "noise": torch.tensor(d["noise"], dtype=torch.float32, device=DEV) if "noise" in d else torch.zeros((), device=DEV),
        "hmask": torch.tensor(d["hmask"], device=DEV) if "hmask" in d else None,
        "leak": torch.tensor(d["leak"], dtype=torch.float32, device=DEV) if "leak" in d else None,
        # continuous brains: 27 cell sensors (empty/X/O) + 2 seat sensors, may fire at any tick
        "continuous": bool(d["continuous"]) if "continuous" in d else False,
        # spiking brains: an output fires the tick it crosses the threshold from below (no refractory timer)
        "spiking": bool(d["spiking"]) if "spiking" in d else False,
        "mtime": mtime,
    }
    brain["edges"] = strongest_edges(brain, 600)
    _cache.update(path=path, mtime=mtime, brain=brain)
    return brain


def strongest_edges(brain, k):
    """The k largest weights as [from, to, w] for the dashboard's wiring view. Edges into sensor
    nodes are dropped (sensors are overwritten every tick) and so are edges touching hidden nodes
    the genome switched off."""
    W = brain["W"].clone()
    n_in = brain["n_in"]
    W[:, :n_in] = 0
    if brain["hmask"] is not None:
        m = brain["hmask"].to(W.dtype)
        W = W * m[:, None] * m[None, :]
    flat = W.abs().flatten()
    k = min(k, int((flat > 0).sum()))
    if k == 0:
        return []
    idx = torch.topk(flat, k).indices
    N = W.shape[0]
    return [[int(i) // N, int(i) % N, round(float(W.flatten()[i]), 3)] for i in idx.tolist()]


def encode_vision(board, me):
    """27 cell sensors (empty, X, O per cell) + 2 seat sensors (I am X / I am O)."""
    s = torch.tensor([me == 1, me == 2], device=DEV)
    return torch.cat([board == 0, board == 1, board == 2, s]).float()


def early_draw(board, nxt):
    """One cell left and filling it cannot win for the player to move: a draw now, so X does not get a
    fifth legal move for a cell that changes nothing (the same rule as in training)."""
    if int((board != 0).sum()) != 8:
        return False
    b = board.clone(); b[int((b == 0).to(torch.int8).argmax())] = nxt
    return int(B.winner(b)) == 0


class Side:
    """One player's running state (activations + stored numbers)."""
    def __init__(self, brain):
        self.x = torch.zeros(brain["N"], device=DEV)
        self.s = torch.zeros(brain["N"], device=DEV) if brain["mem"] is not None else None

    def tick(self, brain, board, me):
        W, b, mem = brain["W"], brain["b"], brain["mem"]
        cur = self.x @ W + b
        if brain["leak"] is not None:
            cur = cur + brain["leak"] * self.x                        # merged node: keeps a fraction of itself
        if mem is not None:
            cur = cur + mem["r"] * self.s
        if bool((brain["noise"] > 0).any()):
            cur = cur + brain["noise"] * torch.randn_like(cur)      # same neural noise as in training
        self.pre = cur
        new = torch.tanh(cur)
        if brain["hmask"] is not None:
            new = new * brain["hmask"]                               # evolved brains may switch hidden nodes off
        if brain["continuous"]:
            new[:brain["n_in"]] = encode_vision(board, me)
        else:
            new[:B.N_IN] = B.encode(board, torch.tensor(me, dtype=torch.int8, device=DEV))
        self.x = new
        if mem is not None:
            factor = mem["d0"] + new @ mem["G"]
            self.s = torch.tanh(factor * self.s + new @ mem["M"])


def play_turns(brain, board, sides, out_lo):
    """Old brains: one move window per turn, refractory ticks first."""
    moves = []
    turn = 1
    result = None
    for _ in range(9):
        me = sides[turn]
        fired = None
        for t in range(brain["max_ticks"]):
            me.tick(brain, board, turn)
            if t < brain["refractory"]:
                continue
            out = me.x[out_lo:]
            if (out > brain["threshold"]).any():
                cell = int(me.pre[out_lo:].argmax())      # choose by pre-tanh preference
                fired = (cell, t, float(out[cell]))
                break
        if fired is None:
            result = {"kind": "timeout", "loser": turn}
            break
        cell, t, val = fired
        if int(board[cell]) != 0:
            moves.append({"side": turn, "cell": cell, "ticks": t, "value": round(val, 3), "illegal": True})
            result = {"kind": "illegal", "loser": turn}
            break
        board[cell] = turn
        moves.append({"side": turn, "cell": cell, "ticks": t, "value": round(val, 3), "illegal": False})
        w = int(B.winner(board))
        if w:
            result = {"kind": "win", "winner": w}
            break
        if bool((board != 0).all()):
            result = {"kind": "draw"}
            break
        turn = 3 - turn
    return moves, result


def play_continuous(brain, board, sides, out_lo):
    """Continuous brains: both bots tick every tick of the whole game. Any output above threshold
    (outside the refractory period after a fire) is a move; on a taken cell or out of turn it is
    illegal. Sitting on one's own turn for more than max_ticks ticks is a freeze."""
    moves, result = [], None
    turn = 1
    refr_left = {1: 0, 2: 0}
    over_prev = {1: None, 2: None}
    clock = 0
    hist = {1: [], 2: []}                       # every node's activation after every tick, for the dashboard
    for t in range(400):
        for s in (1, 2):
            sides[s].tick(brain, board, s)
            hist[s].append((sides[s].x * 127).round().to(torch.int8).cpu().tolist())
        for s in (1, 2):
            sd = sides[s]
            out = sd.x[out_lo:]
            if brain["spiking"]:
                over = out > brain["threshold"]
                rising = over if over_prev[s] is None else (over & ~over_prev[s])
                over_prev[s] = over
                if not bool(rising.any()):
                    continue
                cell = int(torch.where(rising, out, torch.full_like(out, -2.0)).argmax())   # the output that crossed
            else:
                refr_left[s] -= 1
                if refr_left[s] > 0 or not bool((out > brain["threshold"]).any()):
                    continue
                refr_left[s] = brain["refractory"]
                cell = int(sd.pre[out_lo:].argmax())
            val = round(float(out[cell]), 3)
            if s != turn or int(board[cell]) != 0:
                # illegal: the board ignores it, the game goes on (shown, not scored)
                moves.append({"side": s, "cell": cell, "ticks": t, "value": val, "illegal": True,
                              "reason": "fired out of turn" if s != turn else "played an occupied cell"})
                continue
            board[cell] = s
            moves.append({"side": s, "cell": cell, "ticks": t, "value": val, "illegal": False})
            w = int(B.winner(board))
            if w:
                result = {"kind": "win", "winner": w}
            elif bool((board != 0).all()) or early_draw(board, 3 - s):
                result = {"kind": "draw"}
            else:
                turn = 3 - s
                clock = 0
            break
        if result:
            break
        clock += 1
        if clock > brain["max_ticks"]:
            result = {"kind": "timeout", "loser": turn}
            break
    if result is None:
        result = {"kind": "timeout", "loser": turn}
    return moves, result, hist


_live = {"bots": None, "games": 0, "brain_games": 0, "mtime": None}
_ptab = None


def perfect_cell(board):
    """The perfect player's move for this board (lookup table)."""
    global _ptab
    if _ptab is None:
        _ptab = PF.perfect_table()
    code = sum(int(v) * 3 ** i for i, v in enumerate(board.tolist()))
    return int(_ptab[code])


def play_vs_perfect(brain, board, side, me, out_lo, opp_delay=30):
    """The brain (seat `me`) against the perfect player, who answers after a random 1..opp_delay
    ticks like the training opponent. Same spiking rule and clock as play_continuous."""
    import random
    moves, result = [], None
    turn = 1
    over_prev = None
    clock = 0
    opp_wait = random.randint(1, opp_delay)
    hist = []
    opp = 3 - me
    for t in range(400):
        side.tick(brain, board, me)
        hist.append((side.x * 127).round().to(torch.int8).cpu().tolist())
        out = side.x[out_lo:]
        over = out > brain["threshold"]
        rising = over if over_prev is None else (over & ~over_prev)
        over_prev = over
        if bool(rising.any()):
            cell = int(torch.where(rising, out, torch.full_like(out, -2.0)).argmax())
            val = round(float(out[cell]), 3)
            if me != turn or int(board[cell]) != 0:
                # illegal: the board ignores it, the game goes on (shown, not scored)
                moves.append({"side": me, "cell": cell, "ticks": t, "value": val, "illegal": True,
                              "reason": "fired out of turn" if me != turn else "played an occupied cell"})
            else:
                board[cell] = me
                moves.append({"side": me, "cell": cell, "ticks": t, "value": val, "illegal": False})
                w = int(B.winner(board))
                if w:
                    result = {"kind": "win", "winner": w}; break
                if bool((board != 0).all()) or early_draw(board, opp):
                    result = {"kind": "draw"}; break
                turn = opp; clock = 0
                opp_wait = random.randint(1, opp_delay)
        if turn == me:
            clock += 1
            if clock > brain["max_ticks"]:
                result = {"kind": "timeout", "loser": me}; break
        else:
            opp_wait -= 1
            if opp_wait <= 0:
                cell = perfect_cell(board)
                board[cell] = opp
                moves.append({"side": opp, "cell": cell, "ticks": t, "value": 1.0, "illegal": False, "scripted": True})
                w = int(B.winner(board))
                if w:
                    result = {"kind": "win", "winner": w}; break
                if bool((board != 0).all()) or early_draw(board, me):
                    result = {"kind": "draw"}; break
                turn = me; clock = 0
    if result is None:
        result = {"kind": "timeout", "loser": me}
    return moves, result, hist


def self_play(path, opponent="alternate"):
    """One live game. opponent: "self" (brain vs a copy of itself), "perfect" (brain vs the minimax
    player), or "alternate" (odd games self, even games perfect)."""
    brain = load_brain(path)
    board = torch.zeros(9, dtype=torch.int8, device=DEV)
    # Two bots, A and B. Like a training slot each starts every game with zeroed
    # activations and stored numbers, alternates sides every game, and when a newer
    # brain is saved the new weights are swapped in.
    if _live["bots"] is None:
        _live["bots"] = {"A": Side(brain), "B": Side(brain)}
    if _live["mtime"] != brain["mtime"]:
        _live["mtime"] = brain["mtime"]
        _live["brain_games"] = 0
    _live["games"] += 1
    _live["brain_games"] += 1
    if opponent == "alternate":
        opponent = "self" if _live["games"] % 2 == 1 else "perfect"
    first = "A" if (_live["games"] // 2) % 2 == 0 else "B"
    second = "B" if first == "A" else "A"
    sides = {1: _live["bots"][first], 2: _live["bots"][second]}
    who = {1: first, 2: second}
    for sd in sides.values():                  # like training: a blank brain at every game start
        sd.x.zero_()
        if sd.s is not None:
            sd.s.zero_()
    out_lo = brain["n_in"] + brain["hidden"]
    trace = None
    perfect_bot = None
    if opponent == "perfect" and brain["continuous"] and brain["spiking"]:
        # bot A is always the brain here; bot B's slot is the perfect player (no brain, no trace)
        me = 1 if who[1] == "A" else 2
        perfect_bot = who[3 - me]
        moves, result, hist = play_vs_perfect(brain, board, sides[me], me, out_lo)
        trace = {who[me]: hist, perfect_bot: None}
    elif brain["continuous"]:
        moves, result, hist = play_continuous(brain, board, sides, out_lo)
        trace = {who[1]: hist[1], who[2]: hist[2]}            # by bot name, values in -127..127 (= x * 127)
    else:
        moves, result = play_turns(brain, board, sides, out_lo)
    # a few quiet ticks after the game, like training, so the end state settles into memory
    for side in (1, 2):
        for _ in range(brain["refractory"]):
            sides[side].tick(brain, board, side)
    mem_abs = ({s: round(float(sides[s].s.abs().mean()), 3) for s in (1, 2)}
               if sides[1].s is not None else None)
    return {"step": brain["step"], "saved_at": brain["mtime"], "moves": moves, "result": result,
            "max_ticks": brain["max_ticks"], "continuous": brain["continuous"], "trace": trace,
            "layout": {"n_in": brain["n_in"], "hidden": brain["hidden"], "out_lo": out_lo, "threshold": brain["threshold"],
                       "edges": brain["edges"],
                       "leak": (brain["leak"].tolist() if brain["leak"] is not None else None),
                       "hmask": (brain["hmask"].tolist() if brain["hmask"] is not None else None)},
            "board": board.tolist(), "games_with_this_brain": _live["brain_games"],
            "games_total": _live["games"], "x_bot": who[1], "o_bot": who[2],
            "opponent": "perfect" if perfect_bot else "self", "perfect_bot": perfect_bot,
            "memory_level": {who[s]: v for s, v in mem_abs.items()} if mem_abs else None}


if __name__ == "__main__":
    import json
    print(json.dumps(self_play(sys.argv[1] if len(sys.argv) > 1 else "best_single.npz"), indent=1))
