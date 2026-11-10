"""
Perfect tic-tac-toe player, as a lookup table: board code -> best cell.

Depth-aware minimax (prefers the quickest win and the slowest loss) with the
classic tie-break order when several moves are equally good:
center, then corners, then edges. So going first it always takes the center,
it always wins when a win exists, always blocks when it must, and never loses.

    table = perfect_table()          # numpy int8 (3**9,), -1 for terminal / full boards
    cell  = table[code]              # code = sum(board[i] * 3**i)

Self-test:  python perfect.py
"""
import functools

import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
PREFER = [4, 0, 2, 6, 8, 1, 3, 5, 7]          # tie-break: center, corners, edges


def _winner(b):
    for i, j, k in LINES:
        if b[i] and b[i] == b[j] == b[k]:
            return b[i]
    return 0


@functools.lru_cache(maxsize=None)
def _solve(b):
    """b: tuple of 9 in {0,1,2}. Returns (value for the player to move, best cell).
    value: +k for a forced win in (10-k) plies, -k for a forced loss, 0 for a draw."""
    w = _winner(b)
    if w:
        return -10, -1                            # the player to move has already lost
    empties = [i for i in range(9) if b[i] == 0]
    if not empties:
        return 0, -1
    p = 1 if len(empties) % 2 == 1 else 2         # X moves on odd empties
    best_v, best_c = -99, -1
    for c in sorted(empties, key=PREFER.index):   # preference order breaks ties
        nb = list(b); nb[c] = p
        v, _ = _solve(tuple(nb))
        v = -v
        v = v - 1 if v > 0 else (v + 1 if v < 0 else 0)   # one ply deeper: win later, lose later
        if v > best_v:
            best_v, best_c = v, c
    return best_v, best_c


def perfect_table():
    """numpy int8 array (3**9,): best cell for every board code, -1 where no move exists."""
    tab = np.full((3 ** 9,), -1, dtype=np.int8)
    for code in range(3 ** 9):
        b, x = [], code
        for _ in range(9):
            b.append(x % 3); x //= 3
        nx, no = b.count(1), b.count(2)
        if nx - no not in (0, 1):                 # unreachable position
            continue
        tab[code] = _solve(tuple(b))[1]
    return tab


def board_code(b):
    return sum(int(v) * 3 ** i for i, v in enumerate(b))


if __name__ == "__main__":
    import random
    tab = perfect_table()
    print("first move (empty board):", tab[0], "(4 = center)")
    # perfect vs perfect: always a draw
    b = [0] * 9; p = 1
    while tab[board_code(b)] >= 0:
        b[tab[board_code(b)]] = p; p = 3 - p
    print("perfect vs perfect ->", "draw" if _winner(b) == 0 else "WINNER %d (bug)" % _winner(b))
    # perfect vs random, both seats, 2000 games each: must never lose
    for seat in (1, 2):
        res = {"win": 0, "draw": 0, "loss": 0}
        for _ in range(2000):
            b = [0] * 9; p = 1
            while True:
                if p == seat:
                    c = tab[board_code(b)]
                else:
                    c = random.choice([i for i in range(9) if b[i] == 0])
                b[c] = p
                w = _winner(b)
                if w or all(b):
                    res["win" if w == seat else ("loss" if w else "draw")] += 1
                    break
                p = 3 - p
        print(f"perfect as {'X' if seat == 1 else 'O'} vs random:", res)
