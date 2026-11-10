"""
Export an evolved brain for the 3D viewer (brain3d/index.html) and as an STL model for the README.

    python tools/export_brain3d.py                       # newest best_evo_*.npz in the repo root
    python tools/export_brain3d.py best_evo_01303.npz

Writes brain3d/brain.json (everything the viewer needs to draw the brain AND run it live in the browser:
every working synapse, bias, leak, noise, threshold) and brain3d/champion.stl (a monochrome 3D model for
3D printers and model viewers: live neurons as balls, the strongest synapses as rods).
"""
import argparse
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CELL_IN, N_SEAT_IN, N_OUT = 27, 2, 9
N_IN = N_CELL_IN + N_SEAT_IN


def layout(n, live, W, out_lo, seed=1):
    """3D positions. Data flows left to right: the eye (x = -7) -> hidden neurons (placed by a rubber-band
    simulation over their live connections, see below) -> the 9 outputs (x = +7). Switched-off hidden
    neurons get a faint outer shell so the pruning is visible."""
    pos = np.zeros((n, 3))
    # The eye and the outputs are layers standing in the y-z plane, facing the hidden neurons between them.
    # Both read like the board when seen from the hidden side (left column on the left).
    # The eye: three 3x3 retinas stacked in one wall at x = -7, X on top, empty in the middle, O at the bottom.
    for ch, y0 in ((1, 2.6), (0, 0.0), (2, -2.6)):      # channel 0 = empty, 1 = X, 2 = O
        for c in range(9):
            r, k = divmod(c, 3)
            pos[ch * 9 + c] = (-7.0, y0 + (1 - r) * 0.55, (1 - k) * 0.55)
    pos[27] = (-7.0, -4.4, 0.3)                          # I am X
    pos[28] = (-7.0, -4.4, -0.3)                         # I am O
    # the hands: 9 outputs as a board in a wall at x = +7
    for c in range(9):
        r, k = divmod(c, 3)
        pos[out_lo + c] = (7.0, (1 - r) * 0.9, (k - 1) * 0.9)
    rng = np.random.default_rng(seed)
    hid = [j for j in range(N_IN, out_lo) if live[j]]
    dead = [j for j in range(N_IN, out_lo) if not live[j]]
    # Rubber bands. Every live connection is one identical spring, whatever its weight: it pulls when
    # stretched past its rest length and pushes when squeezed under it. A pair wired both ways has two
    # bands. On top, hidden neurons push each other apart when closer than REPEL. The eye and the outputs
    # are pinned; switched-off neurons and connections exert no force at all.
    REST, REPEL, K_SPRING, K_REPEL = 3.0, 2.4, 0.06, 16.0
    conn = (W != 0) & live[:, None] & live[None, :]
    conn[:, :N_IN] = False                                # nothing flows into the eye (sensors are overwritten)
    np.fill_diagonal(conn, False)                         # a self-loop pulls on nothing
    ii, jj = np.nonzero(conn)
    keep = np.isin(ii, hid) | np.isin(jj, hid)            # bands with a hidden neuron on at least one end
    ii, jj = ii[keep], jj[keep]
    is_hid = np.zeros(n, bool); is_hid[hid] = True
    for j in hid:
        v = rng.normal(size=3)
        pos[j] = v / np.linalg.norm(v) * rng.uniform(0.3, 2.0)
    vel = np.zeros_like(pos)
    H = np.array(hid)
    DT = 0.2
    for it in range(12000):
        F = np.zeros_like(pos)
        d = pos[jj] - pos[ii]
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        f = K_SPRING * (L - REST) * d / L                 # same band for every connection
        np.add.at(F, ii, f)
        np.add.at(F, jj, -f)
        dh = pos[H][None, :, :] - pos[H][:, None, :]      # hidden <-> hidden
        Lh = np.linalg.norm(dh, axis=2, keepdims=True) + 1e-9
        push = K_REPEL * np.clip(REPEL - Lh, 0, None) * dh / Lh
        np.fill_diagonal(push[..., 0], 0); np.fill_diagonal(push[..., 1], 0); np.fill_diagonal(push[..., 2], 0)
        F[H] -= push.sum(1)
        vel[H] = (vel[H] + F[H] * DT) * 0.8
        stepv = vel[H] * DT
        m = np.linalg.norm(stepv, axis=1, keepdims=True)
        pos[H] += stepv * np.minimum(1.0, 0.1 / (m + 1e-9))   # at most 0.1 per step: stiff pushes stay stable
    stretch = np.linalg.norm(pos[jj] - pos[ii], axis=1)
    print(f"layout: {len(ii)} bands, length {stretch.min():.2f}..{stretch.max():.2f} (rest {REST}), "
          f"settled speed {np.abs(vel[H]).max():.4f}")
    # switched-off neurons: a faint shell around the cloud (golden-angle spiral)
    R = float(np.linalg.norm(pos[hid], axis=1).max()) + 1.2 if hid else 4.6
    for a, j in enumerate(dead):
        t = (a + 0.5) / max(len(dead), 1)
        phi = math.acos(1 - 2 * t); th = math.pi * (1 + 5 ** 0.5) * a
        pos[j] = (R * math.cos(th) * math.sin(phi), R * math.cos(phi), R * math.sin(th) * math.sin(phi))
    return pos


def stl(pos, live, out_lo, edges, k_edges):
    """ASCII STL: live neurons as octahedra, the k strongest synapses as thin triangular rods."""
    facets = []

    def tri(a, b, c):
        facets.append((a, b, c))

    g = (1 + 5 ** 0.5) / 2                                  # icosahedron: a round-looking ball in 20 faces
    ico_v = np.array([(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g), (0, -1, -g), (0, 1, -g),
                      (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)], float)
    ico_v /= np.linalg.norm(ico_v[0])
    ico_f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]

    def ball(p, s):
        v = np.asarray(p) + ico_v * s
        for a, b, c in ico_f:
            tri(v[a], v[b], v[c])

    def rod(p, q, r):
        p, q = np.asarray(p), np.asarray(q)
        d = q - p; L = np.linalg.norm(d)
        if L < 1e-6:
            return
        d /= L
        u = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(u) < 1e-3:
            u = np.cross(d, [0, 1.0, 0])
        u /= np.linalg.norm(u); w = np.cross(d, u)
        ring = [u * r * math.cos(a) + w * r * math.sin(a) for a in (0, 2.094, 4.189)]
        for i in range(3):
            a0, a1 = ring[i], ring[(i + 1) % 3]
            tri(p + a0, q + a0, q + a1)
            tri(p + a0, q + a1, p + a1)

    for j in range(len(live)):
        if not live[j]:
            continue
        s = 0.24 if j < N_IN else (0.48 if j >= out_lo else 0.42)
        ball(pos[j], s)
    for i, j, w in sorted(edges, key=lambda e: -abs(e[2]))[:k_edges]:
        if i != j:
            rod(pos[i], pos[j], 0.06)
    # GitHub's STL viewer stands models on Z; the layout's "up" is Y, so turn it upright: (x, y, z) -> (x, -z, y)
    up = lambda v: np.array([v[0], -v[2], v[1]])
    facets = [(up(a), up(b), up(c)) for a, b, c in facets]
    f = lambda v: " ".join(f"{x:.2f}".rstrip("0").rstrip(".") if "." in f"{x:.2f}" else f"{x:.2f}" for x in v)
    lines = ["solid brain"]
    for a, b, c in facets:
        nrm = np.cross(b - a, c - a); m = np.linalg.norm(nrm)
        nrm = nrm / m if m > 0 else nrm
        lines += [f"facet normal {f(nrm)}", "outer loop", f"vertex {f(a)}", f"vertex {f(b)}", f"vertex {f(c)}",
                  "endloop", "endfacet"]
    lines.append("endsolid brain")
    return "\n".join(lines) + "\n", len(facets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", nargs="?", default=None, help="best_evo_*.npz (default: newest in the repo root)")
    ap.add_argument("--stl-edges", type=int, default=60, help="strongest synapses drawn in the STL model")
    a = ap.parse_args()
    path = a.brain or sorted(glob.glob(os.path.join(ROOT, "best_evo_*.npz")), key=os.path.getmtime)[-1]
    d = np.load(path)
    W = d["W"].astype(np.float64)
    n = W.shape[0]
    out_lo = n - N_OUT
    live = d["hmask"] > 0.5
    live[:N_IN] = True; live[out_lo:] = True
    gen = int(d["game"]) if "game" in d.files else None
    # working synapses: from a live neuron to a live hidden/output neuron (sensors are overwritten every tick)
    edges = [(int(i), int(j), float(W[i, j])) for i in range(n) for j in range(N_IN, n)
             if live[i] and live[j] and W[i, j] != 0]
    pos = layout(n, live, W, out_lo)
    out_dir = os.path.join(ROOT, "brain3d")
    os.makedirs(out_dir, exist_ok=True)
    hidden_live = int(live[N_IN:out_lo].sum())
    data = {
        "source": os.path.basename(path), "generation": gen,
        "n": n, "n_in": N_IN, "out_lo": out_lo, "hidden_max": out_lo - N_IN, "hidden_live": hidden_live,
        "threshold": float(d["threshold"]), "max_ticks": int(d["max_ticks"]),
        "synapses": len(edges), "synapses_full": n * (n - N_IN),
        "live": [int(v) for v in live],
        "pos": [[round(float(c), 3) for c in p] for p in pos],
        "bias": [round(float(v), 4) for v in d["b"]],
        "leak": [round(float(v), 4) for v in d["leak"]],
        "noise": [round(float(v), 4) for v in np.broadcast_to(d["noise"], (n,))],
        "edges": [[i, j, round(w, 4)] for i, j, w in edges],
    }
    with open(os.path.join(out_dir, "brain.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    text, nf = stl(pos, live, out_lo, edges, a.stl_edges)
    with open(os.path.join(out_dir, "champion.stl"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{os.path.basename(path)}: generation {gen}, {int(live.sum())} live neurons ({hidden_live} hidden), "
          f"{len(edges)} working synapses -> brain3d/brain.json; STL {nf} facets, {len(text) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
