# AI Brain — project docs

An experiment testing one idea: **modern AI is limited because humans designed its
structure. What happens if we stop designing and let structure emerge?**

The test bed is tic-tac-toe. It is small enough to run tens of thousands of
candidate brains per minute on one GPU, and it has a perfect player (minimax) to
measure against.

| Doc | What it covers |
|---|---|
| [problem-structure.md](problem-structure.md) | Problem 1: the architecture is one-directional and start-stop; a brain loops and never stops |
| [problem-thought.md](problem-thought.md) | Problem 2: the model holds thought only as text; a mind holds it as unspoken internal state |
| [theory.md](theory.md) | Problem 3 and the theory: the wiring is designed by a human instead of selected |
| [approach.md](approach.md) | The proposed fix and how the tic-tac-toe brain implements it |
| [experiments.md](experiments.md) | Every run so far, what happened, what we learned, what is next |
| [diary.md](diary.md) | Day-by-day notes: what I noticed and what I was thinking |

## Files in the repo

| File | Purpose |
|---|---|
| `brain_ttt.py` | CPU version: brain, game arena, evolution, human-vs-brain play |
| `brain_ttt_gpu.py` | Same, on PyTorch/CUDA, for populations of 50,000+ |
| `dashboard.py` + `dashboard.html` | Live chart of a training log, refreshes every 5 s |
| `train_*.log` | Logs of each training run |
| `best_*.npz` | Best brain saved from each run |

## Quick start

```bash
python brain_ttt_gpu.py train --pop 50000 --games 150 --kill-mistakes
python dashboard.py            # then open http://localhost:8766
python brain_ttt.py play best_gpu_big.npz
```
