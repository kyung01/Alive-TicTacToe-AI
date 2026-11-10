# The approach: a brain that is selected, not designed

## Design principles

Each principle maps directly to one of the three brain properties in
[theory.md](theory.md).

| Brain property | Design choice here |
|---|---|
| No prescribed wiring | Every node connects to every other node, including itself. No layers. |
| Never stops | The net ticks continuously during a game. State carries over between moves. |
| Mistakes are fatal | A brain that makes an illegal move or freezes dies. Only clean brains reproduce. |

Nothing tells the brain what tic-tac-toe is, what a legal move is, or what a
win is. It sees the board, it has outputs wired to cells, and it either survives
or does not.

## The brain

```
 18 sensor nodes   9 cells x (mine / theirs)   clamped to the board every tick
 82 hidden nodes   free                        100 non-output nodes total
  9 output nodes   one per cell                a move fires when one crosses threshold
────────────────
109 nodes, 109 x 109 = 11,881 weights, plus one bias per node
```

Every tick, each node's new value is `tanh(sum of all nodes x weights + bias)`.
Then the 18 sensor nodes are overwritten with the current board so the brain is
always "looking at" it. There is no reset between moves.

## How a move happens

1. After the opponent moves, the brain keeps ticking. For the first 3 ticks
   outputs are ignored (a refractory period, so it cannot fire on stale state).
2. On each later tick, if any output node exceeds the threshold (0.8), the
   strongest one is the chosen cell.
3. If that cell is empty, the mark is placed. If it is occupied, that is an
   **illegal move** and the brain dies.
4. If no output fires within 100 ticks, the brain **froze** and dies.

Deciding takes as many ticks as the brain needs. Untrained brains fire on tick 3.
Trained ones have been observed taking 4 to 12 ticks.

## Training by selection

There is no gradient and no teacher. The learning rule is natural selection:

1. Start with a population of random brains (currently 50,000).
2. Each brain plays a sequence of games (currently 150) against a random
   opponent, half as X and half as O.
3. **Lifespan fitness.** The brain dies at its first mistake. Its fitness is the
   number of legal moves it made before dying, plus the score of games it finished
   (win +1, draw +0.5, loss -1).
4. The top 20% become parents. Each child takes each weight at random from one
   of two parents (crossover), then a small fraction of weights get Gaussian noise
   (mutation).
5. Repeat.

The kill rule is the central idea. A brain is not corrected after an error; it is
removed. Lifespan measured in moves rather than games exists because without it
nearly every random brain dies in its first game and selection has nothing to
grade on.

## What is deliberately *not* done

- No masking of occupied cells. The brain must learn legality by seeing the
  board. (A `--mask-illegal` flag exists only as a diagnostic baseline.)
- No hand-coded features, no "is this a winning line" input.
- No supervised signal from the minimax oracle. Minimax is used only to measure
  the brain afterward.
- No backprop. The ticking-until-threshold mechanism is not differentiable, and
  that is fine: the point is that structure emerges from selection.

## Measuring success

| Level | Meaning |
|---|---|
| Survives all games | Learned legality without being told the rules |
| Beats a random player most of the time | Learned the goal from survival alone |
| Draws against minimax | Learned the game fully (tic-tac-toe cannot be won against perfect play) |

## Hardware

An RTX A6000 (48 GB). Populations of 50,000 brains x 150 games fit in about
18 GB allocated (about 29 GB total on the card including PyTorch's cache) and take
about 35 seconds per generation. The arena is fully vectorized: all brains and all
their games tick in lockstep as one batched matrix multiply.
