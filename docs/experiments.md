# Experiments

Chronological. Each entry records the setup, the result, and what it taught us.
Numbers are for the single best brain of the run unless stated.

Date: 2026-11-09.

## 1. Baseline — CPU, penalty for mistakes

`brain_ttt.py train --gens 200 --pop 200 --games 20`
Illegal move = -1.5 score, no kill.

| | vs random | vs minimax |
|---|---|---|
| win | 38% | 0% |
| illegal | 57% | 89% |

Fitness plateaued at +0.25 by generation 100. The brain learned to fire at the
first allowed tick every time and plays a mostly fixed pattern. A penalty is not
enough: it can afford mistakes and still rank well.

## 2. Diagnostic — mask occupied cells (not faithful to the theory)

`--mask-illegal`, 500 generations, 200 brains, 40 games.
Occupied cells cannot fire, so legality is handled for the brain.

| | vs random | vs minimax |
|---|---|---|
| win | 78% | 0% |
| draw | 6% | 6% |
| loss | 16% | 94% |

**Lesson:** the architecture itself can learn to play. Decision time also grew
from 3 ticks to 4 to 7, meaning the brain was actually using its recurrent
dynamics. The entire difficulty in the faithful setup is learning legality.

## 3. Kill rule, version 1 — any mistake across all games is fatal

CPU, 200 brains, 40 games; then GPU, 4,000 brains, 100 games.

Result: **zero survivors** every generation in both runs. Not one brain out of
4,000 got through 100 games without an illegal move. With everyone dead,
ranking fell back to score and the run behaved like experiment 1.

**Lesson:** pure "any mistake is fatal" has no gradient at the start. Random
brains make an illegal move within their first few moves.

## 4. Kill rule, version 2 — lifespan in games

GPU, 4,000 brains, 100 games, 390 generations.
A brain dies at its first mistake; fitness = score of games finished before death.

Result: longest life stayed at 4 to 6 games, average 0.1. Fitness flat.

**Lesson:** still too coarse. A brain that survives 4 moves and one that survives
1 move both die in game 1 and both score zero.

## 5. Kill rule, version 3 — lifespan in moves (current)

Fitness = legal moves made before death / (games x 5) + score of finished games.
Mutation softened (2% of weights, std 0.15) so good structure is not scrambled.

Runs:

| Population | Games | VRAM | Time/gen | Status |
|---|---|---|---|---|
| 4,000 | 100 | 4.4 GB | 2.5 s | flat at generation 50, stopped |
| 70,000 | 150 | 25 GB | 40 s | stopped to increase population |
| 85,000 | 150 | 30.6 GB alloc, card full at 47.7/49 GB | 48 s | too much memory, stopped |
| **50,000** | **150** | **18 GB alloc, 29 GB total** | **35 s** | **running, 400 generations** |

At generation 13 the longest life is 5 to 6 games and about 25% of the best
brain's moves are still illegal. No improvement over the random start yet.

## What we know so far

1. The brain design works as a player when legality is handled for it
   (experiment 2). The recurrent, continuously ticking, fully connected design is
   not the obstacle.
2. Learning legality from survival alone has not happened yet. Fitness is flat
   across every faithful run.
3. Why legality is hard here: an output node is driven by 82 hidden nodes with
   random weights, giving it a noise of roughly ±1.4 before tanh. To suppress
   firing on an occupied cell, the two sensor-to-output weights for that cell must
   become strongly negative (around -3 to -5) to beat that noise, for all 9 cells.
   That is 18 specific weights out of 11,881, and uniform crossover between
   unrelated parents breaks partial solutions.

## Options on the table

In rough order of how faithful they are to the theory:

1. **Let it run.** 400 generations of 50,000 brains is the biggest search yet.
2. **Sequential play, not parallel.** Currently games are parallel and only
   reinterpreted as a sequence. True sequential play would let the brain carry
   state between games, as a real lifespan would.
3. **Less destructive reproduction.** Clone-and-mutate instead of crossover, or
   crossover only between near relatives, so a brain that has learned one
   cell's legality is not torn apart.
4. **Sparser initial wiring.** Start with most weights at zero so outputs are
   not swamped by noise. Still no designed structure, just quieter.
5. **Lower threshold noise.** A smaller initial weight scale (std 0.05 instead of
   0.15) for the same reason.

None of these tell the brain what a legal move is. They change how harsh the
starting conditions are, not what is learned.
