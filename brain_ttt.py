"""
A fully-connected recurrent "brain" that plays tic-tac-toe, trained by evolution.

Nodes:  18 sensors (9 cells x mine / theirs)
       + H hidden nodes
       + 9 output nodes (one per cell)
Every node connects to every node. The net ticks continuously and is never
reset during a game. A move happens when an output node crosses the firing
threshold; the strongest firing output is the chosen cell. If that cell is
occupied the move is illegal and the game is lost. If no output fires within
max_ticks the net "froze" and the game is lost.

Training: a population of weight matrices plays games, the best survive,
children are made by crossover + mutation. No gradients, no hand-written rule.

Usage:
    python brain_ttt.py train  [--gens 300 --opponent random|minimax|mixed ...]
    python brain_ttt.py eval   best_brain.npz
    python brain_ttt.py play   best_brain.npz        (you vs the brain)
"""
import argparse
import time

import numpy as np

N_IN, N_OUT = 18, 9
LINES = np.array([(0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7), (2, 5, 8),
                  (0, 4, 8), (2, 4, 6)])
OUTCOMES = ["win", "draw", "loss", "illegal", "timeout"]
WIN, DRAW, LOSS, ILLEGAL, TIMEOUT = range(5)
SCORE = np.array([1.0, 0.5, -1.0, -1.5, -1.5])


# ----------------------------------------------------------------- board utils
def winner(boards):
    """boards (...,9) int -> (...,) 0 none, 1 X wins, 2 O wins."""
    cells = boards[..., LINES]                       # (...,8,3)
    same = (cells[..., 0] == cells[..., 1]) & (cells[..., 1] == cells[..., 2])
    won = same & (cells[..., 0] != 0)                # (...,8)
    val = (cells[..., 0] * won).max(-1)
    return val.astype(np.int8)


def encode(boards, player):
    """boards (...,9), player (...) -> (...,18) float: [mine(9), theirs(9)]."""
    mine = boards == player[..., None]
    theirs = (boards != 0) & ~mine
    return np.concatenate([mine, theirs], axis=-1).astype(np.float32)


def build_minimax_table():
    """state tuple -> (value for player to move, list of optimal cells)."""
    table = {}

    def solve(b):
        if b in table:
            return table[b][0]
        w = winner(np.array(b, dtype=np.int8))
        if w:
            table[b] = (-1, [])          # player to move has already lost
            return -1
        empties = [i for i in range(9) if b[i] == 0]
        if not empties:
            table[b] = (0, [])
            return 0
        p = 1 if len(empties) % 2 == 1 else 2
        best, moves = -2, []
        for c in empties:
            nb = list(b)
            nb[c] = p
            v = -solve(tuple(nb))
            if v > best:
                best, moves = v, [c]
            elif v == best:
                moves.append(c)
        table[b] = (best, moves)
        return best

    solve((0,) * 9)
    return table


# ------------------------------------------------------------------- the arena
class Arena:
    def __init__(self, hidden, threshold, max_ticks, refractory, mask_illegal, rng):
        self.H = hidden
        self.N = N_IN + hidden + N_OUT
        self.threshold = threshold
        self.max_ticks = max_ticks
        self.refractory = refractory
        self.mask_illegal = mask_illegal
        self.rng = rng
        self.table = None

    def opponent_moves(self, boards, kind):
        K = len(boards)
        if kind == "mixed":
            kinds = np.where(self.rng.random(K) < 0.5, "random", "minimax")
        else:
            kinds = np.full(K, kind)
        moves = np.zeros(K, dtype=np.int64)
        rnd = kinds == "random"
        if rnd.any():
            empties = boards[rnd] == 0
            moves[rnd] = (self.rng.random(empties.shape) * empties).argmax(-1)
        mm = ~rnd
        if mm.any():
            if self.table is None:
                self.table = build_minimax_table()
            for i in np.nonzero(mm)[0]:
                opts = self.table[tuple(boards[i].tolist())][1]
                moves[i] = opts[self.rng.integers(len(opts))]
        return moves

    def play(self, W, b, opponent, games):
        """W (P,N,N), b (P,N). Returns outcome (P,G) int, ticks (P,G)."""
        P, N = W.shape[0], self.N
        G = games
        board = np.zeros((P, G, 9), dtype=np.int8)
        state = np.zeros((P, G, N), dtype=np.float32)
        net_player = np.broadcast_to(
            np.where(np.arange(G) % 2 == 0, 1, 2).astype(np.int8), (P, G)).copy()
        turn = np.ones((P, G), dtype=np.int8)          # X moves first
        done = np.zeros((P, G), dtype=bool)
        outcome = np.full((P, G), DRAW, dtype=np.int8)
        ticks = np.zeros((P, G), dtype=np.float32)
        bb = b[:, None, :]
        out_lo = N_IN + self.H

        def settle(pi, gi, cells):
            """Place marks and resolve wins/draws for games (pi,gi)."""
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
            # --- opponent's turn
            m = ~done & (turn != net_player)
            if m.any():
                pi, gi = np.nonzero(m)
                moves = self.opponent_moves(board[pi, gi], opponent)
                settle(pi, gi, moves)

            # --- net's turn: tick until an output fires
            m = ~done & (turn == net_player)
            if not m.any():
                break
            fired = np.zeros((P, G), dtype=bool)
            for t in range(self.max_ticks):
                active = m & ~fired
                pa = active.any(1)                    # individuals still thinking
                if not pa.any():
                    break
                s = state[pa]
                s = np.tanh(s @ W[pa] + bb[pa])
                s[..., :N_IN] = encode(board[pa], net_player[pa])
                state[pa] = s
                if t < self.refractory:
                    continue
                out = state[..., out_lo:]
                if self.mask_illegal:
                    out = np.where(board == 0, out, -9.0)
                fire = (out > self.threshold).any(-1) & active
                if fire.any():
                    pi, gi = np.nonzero(fire)
                    cells = out[pi, gi].argmax(-1)
                    fired[pi, gi] = True
                    ticks[pi, gi] = t
                    occupied = board[pi, gi, cells] != 0
                    if occupied.any():
                        outcome[pi[occupied], gi[occupied]] = ILLEGAL
                        done[pi[occupied], gi[occupied]] = True
                    ok = ~occupied
                    if ok.any():
                        settle(pi[ok], gi[ok], cells[ok])
            froze = m & ~fired
            if froze.any():
                outcome[froze] = TIMEOUT
                ticks[froze] = self.max_ticks
                done[froze] = True
            if done.all():
                break
        return outcome, ticks


def stats(outcome):
    counts = np.bincount(outcome.ravel(), minlength=5) / outcome.size
    return {k: counts[i] for i, k in enumerate(OUTCOMES)}


def fmt(st):
    return " ".join(f"{k}={v:5.1%}" for k, v in st.items())


# --------------------------------------------------------------------- training
def train(a):
    rng = np.random.default_rng(a.seed)
    arena = Arena(a.hidden, a.threshold, a.max_ticks, a.refractory, a.mask_illegal, rng)
    N, P = arena.N, a.pop
    print(f"nodes: {N_IN} in + {a.hidden} hidden + {N_OUT} out = {N}  "
          f"(weights per brain: {N*N})  pop={P}  games/brain/gen={a.games}  "
          f"opponent={a.opponent}")
    W = rng.normal(0, a.init_std, (P, N, N)).astype(np.float32)
    b = rng.normal(0, 0.1, (P, N)).astype(np.float32)
    n_elite = max(2, int(P * a.elite))
    best_fit = -9
    t0 = time.time()
    for gen in range(1, a.gens + 1):
        outcome, ticks = arena.play(W, b, a.opponent, a.games)
        fit = SCORE[outcome].mean(1)
        if a.kill_mistakes:
            # a brain that made even one illegal move or froze is dead:
            # it ranks below every clean brain no matter how many games it won
            mistake = ((outcome == ILLEGAL) | (outcome == TIMEOUT)).any(1)
            fit = np.where(mistake, fit - 10.0, fit)
        order = np.argsort(-fit)
        elite = order[:n_elite]
        top = order[0]
        st = stats(outcome[top])
        alive = int((fit > -5).sum()) if a.kill_mistakes else P
        print(f"gen {gen:4d}  best={fit[top]:+.3f}  mean={fit.mean():+.3f}  alive={alive:3d}  "
              f"ticks={ticks[top].mean():5.1f}  {fmt(st)}  [{time.time()-t0:5.0f}s]",
              flush=True)
        if fit[top] >= best_fit:
            best_fit = fit[top]
            np.savez(a.out, W=W[top], b=b[top], hidden=a.hidden, threshold=a.threshold,
                     max_ticks=a.max_ticks, refractory=a.refractory,
                     mask_illegal=a.mask_illegal)
        if gen % a.eval_every == 0 or gen == a.gens:
            evaluate(W[top:top+1], b[top:top+1], arena, a.eval_games)

        # --- next generation
        n_child = P - n_elite
        p1 = rng.choice(elite, n_child)
        p2 = rng.choice(elite, n_child)
        cross = rng.random((n_child, N, N)) < 0.5
        childW = np.where(cross, W[p1], W[p2])
        childb = np.where(rng.random((n_child, N)) < 0.5, b[p1], b[p2])
        mut = rng.random((n_child, N, N)) < a.mut_rate
        childW += mut * rng.normal(0, a.mut_std, (n_child, N, N)).astype(np.float32)
        childb += (rng.random((n_child, N)) < a.mut_rate) * \
            rng.normal(0, a.mut_std, (n_child, N)).astype(np.float32)
        W = np.concatenate([W[elite], childW]).astype(np.float32)
        b = np.concatenate([b[elite], childb]).astype(np.float32)
    print(f"saved best brain to {a.out}")


def evaluate(W, b, arena, games):
    for opp in ("random", "minimax"):
        outcome, ticks = arena.play(W, b, opp, games)
        print(f"      eval vs {opp:7s} ({games} games): {fmt(stats(outcome[0]))}  "
              f"ticks={ticks[0].mean():.1f}", flush=True)


def load(path):
    d = np.load(path)
    arena = Arena(int(d["hidden"]), float(d["threshold"]), int(d["max_ticks"]),
                  int(d["refractory"]), bool(d["mask_illegal"]), np.random.default_rng())
    return d["W"][None], d["b"][None], arena


# ------------------------------------------------------------------ human play
def show(board):
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(3):
        print("   " + " ".join(sym[int(board[r*3+c])] for c in range(3))
              + "     " + " ".join(str(r*3+c) for c in range(3)))


def play_human(path):
    W, b, arena = load(path)
    W, b = W[0], b[0]
    board = np.zeros(9, dtype=np.int8)
    state = np.zeros(arena.N, dtype=np.float32)
    human = 1 if input("Play as X (first) or O? [X/o] ").strip().lower() != "o" else 2
    net = 3 - human
    turn = 1
    out_lo = N_IN + arena.H
    while True:
        show(board)
        w = winner(board)
        if w or (board != 0).all():
            print("you win!" if w == human else "brain wins!" if w else "draw.")
            return
        if turn == human:
            try:
                c = int(input("your cell (0-8): "))
            except ValueError:
                continue
            if not 0 <= c < 9 or board[c] != 0:
                print("occupied / invalid")
                continue
            board[c] = human
        else:
            for t in range(arena.max_ticks):
                state = np.tanh(state @ W + b)
                state[:N_IN] = encode(board, np.int8(net))
                if t < arena.refractory:
                    continue
                out = state[out_lo:]
                if arena.mask_illegal:
                    out = np.where(board == 0, out, -9.0)
                if (out > arena.threshold).any():
                    c = int(out.argmax())
                    print(f"brain fires cell {c} after {t} ticks "
                          f"(signal {out[c]:.2f})")
                    if board[c] != 0:
                        print("brain made an ILLEGAL move. brain loses.")
                        return
                    board[c] = net
                    break
            else:
                print("brain froze (no output fired). brain loses.")
                return
        turn = 3 - turn


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--hidden", type=int, default=82, help="100 non-output nodes total by default")
    t.add_argument("--pop", type=int, default=200)
    t.add_argument("--gens", type=int, default=200)
    t.add_argument("--games", type=int, default=20, help="games per brain per generation")
    t.add_argument("--opponent", default="random", choices=["random", "minimax", "mixed"])
    t.add_argument("--threshold", type=float, default=0.8)
    t.add_argument("--max-ticks", type=int, default=100)
    t.add_argument("--refractory", type=int, default=3)
    t.add_argument("--mask-illegal", action="store_true",
                   help="only let empty cells fire (easier; off = faithful to the theory)")
    t.add_argument("--kill-mistakes", action="store_true",
                   help="a brain that makes any illegal move or freezes cannot reproduce")
    t.add_argument("--elite", type=float, default=0.2)
    t.add_argument("--mut-rate", type=float, default=0.03)
    t.add_argument("--mut-std", type=float, default=0.3)
    t.add_argument("--init-std", type=float, default=0.15)
    t.add_argument("--eval-every", type=int, default=25)
    t.add_argument("--eval-games", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="best_brain.npz")
    e = sub.add_parser("eval")
    e.add_argument("path")
    e.add_argument("--games", type=int, default=1000)
    p = sub.add_parser("play")
    p.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        W, b, arena = load(a.path)
        evaluate(W, b, arena, a.games)
    else:
        play_human(a.path)


if __name__ == "__main__":
    main()
