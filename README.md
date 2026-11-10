<p align="center">
  <img src="assets/robot-blue.png" width="120" alt="X brain">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/robot-red.png" width="120" alt="O brain">
</p>

<h1 align="center">Alive Tic-Tac-Toe AI</h1>

<p align="center">
  <b>A neural network that never stops running: it watches the board, remembers, waits its turn and acts.<br>
  No human designed its architecture or its learning rule. Evolution did.</b>
</p>

---

## The idea in one paragraph

A normal AI is a function. You call it with an input, it returns an output, and between calls it does not exist.
A brain is different: it runs **all the time**. It sees the world change, keeps an internal state, and decides
*by itself* **when** to act, not only **what** to do. This project tests whether that kind of agent can come out of
pure evolution. The test is tic-tac-toe. It is small enough to simulate **1.5 million brains per generation on one
GPU**, and it has a perfect player to measure against. The brains start as random wiring. After about 1,100
generations of natural selection they sit through the opponent's turn instead of firing into it, they stop
pressing several buttons at once, and they go from losing almost every game against a perfect minimax player to
standing up to it. Nobody programmed any of that. And along the way **the network shrank itself**: evolution found more than half
of the neurons unnecessary and switched them off, leaving a brain that uses **a tenth of the connections** of a
fully connected one.

---

## Why this is different from a normal tic-tac-toe bot

A classic tic-tac-toe AI (minimax, a Q-table, a trained MLP) gets a perfectly packaged question: *"here is the
board, it is your turn, pick one of 9 cells."* The program handles the rest for it:

- **whose turn it is** (it is only ever called on its own turn),
- **when to answer** (it answers exactly once, instantly),
- **how many moves to make** (exactly one; the argmax picks it),
- **which moves are legal** (illegal moves are usually masked out).

**This brain is given none of that.** It lives in a continuous stream of time:

| | Traditional bot | Alive brain (this project) |
|---|---|---|
| How does it get the board? | Handed a data structure on its turn | Through an artificial **eye** that streams the board into it every tick |
| When does it run? | Only when called | **Every tick**, during its turn *and* the opponent's |
| Who says it is my turn? | The program | **Nobody.** It has to see the board change |
| How is a move made? | `argmax` over 9 outputs | An output neuron **spikes** when it crosses a threshold |
| Firing twice / out of turn | Impossible | Possible, and wasteful: it has to *learn* not to |
| Doing nothing | Impossible | Possible: if it waits too long on its own turn, it **times out and loses** |
| Memory | None (or a hand-built buffer) | Every neuron has its own **leak gene**: how much of its value it keeps from tick to tick |
| Architecture | Designed by a human | **Evolved**: neurons and synapses switch on and off by mutation |
| Learning rule | Designed (backprop, TD, ...) | **None.** Only selection: brains that survive have children |

So before it can play well, the brain has to discover by itself that:

1. **The game has turns.** After I move, the world is not mine for a while.
2. **Waiting is part of the game.** Sitting through the opponent's turn (1–30 ticks, random) is correct. Firing
   during it is wasted effort.
3. **One move per turn.** Spiking several outputs, or the same cell again, does nothing useful.
4. **But not too much waiting.** On my own turn I have 30 ticks to act, or I lose.
5. **Then** the actual tactics: blocking, taking the centre, finishing lines.

---

## The brain

```mermaid
flowchart LR
    subgraph Eyes["the eye: 29 sensors"]
      E1["9 x 'cell is empty'"]
      E2["9 x 'cell is X'"]
      E3["9 x 'cell is O'"]
      E4["I am X / I am O"]
    end
    subgraph Hidden["up to 82 hidden neurons<br/>(each can be switched off by evolution)"]
      H(("recurrent web<br/>every neuron may<br/>connect to every neuron"))
    end
    subgraph Hands["9 output neurons"]
      O["one per cell<br/>spike = move"]
    end
    Eyes --> H
    H --> H
    H --> O
    O --> H
```

Every tick, every neuron updates:

```
x  =  tanh( d0 * x   +   (W0 * conn) · x   +   b   +   noise )
          └ memory ┘   └ evolved wiring  ┘   bias   per-neuron random kick
```

- `d0` (per neuron, 0–1) is the **leak**: 0 = forgets instantly, 1 = holds its value. It is the brain's memory.
  Nothing else stores the past.
- An output **spikes the tick it crosses its threshold from below**, and must fall back under before it can spike
  again. No refractory timer, no move window, no argmax. The spike *is* the move.
- Every neuron gets a fresh random kick every tick (seeded from OS entropy), so no two lives are the same.
- The weights **never change during a life**. A brain is exactly its genome from first tick to last.

### The eye: it *sees* the board, nobody tells it what is happening

A normal bot receives the board as a data structure, only on its turn. This brain gets something closer to
vision: we built a simple artificial **eye**. Each of the 9 cells is watched by three "photoreceptor" sensor
neurons, one that lights up when the cell is empty, one for X and one for O, like three kinds of cone cell each
tuned to a single colour. That makes 27 receptors, plus two that tell the brain which side it plays.

```
     the board                      what the eye sends into the brain, every tick

     X │   │                    "empty"        "X"           "O"
    ───┼───┼───                  · ● ●        ● · ·         · · ·
       │ O │                     ● · ●        · · ·         · ● ·
    ───┼───┼───                  ● ● ●        · · ·         · · ·
       │   │                                                       + "I am X" ●   "I am O" ·
```

- **The eye never closes.** It streams the board into the brain every single tick, on the brain's own turn and
  during the opponent's, the way your eyes keep seeing while you wait.
- **It shows only what is there**: the marks. There is no "your turn" light, no "the opponent just moved" signal,
  and no list of legal moves. When the opponent moves, a receptor flips, and that change in the picture is the
  only clue. Noticing it, and working out what it means, is the brain's job.
- **It sees X and O, not "mine" and "theirs".** To know which marks are its own, the brain has to combine what it
  sees with the seat receptor.

Everything the brain knows about the game comes in through this eye, and it acts only through its 9 output
neurons, one per cell. Everything in between is evolved.

### Everything is a gene

| Gene | What it is |
|---|---|
| `W0` | strength of every possible synapse (120 x 120) |
| `conn` | **whether each synapse exists at all** (on/off) |
| `alive` | **whether each hidden neuron exists at all** (on/off) |
| `b` | per-neuron bias |
| `d0` | per-neuron leak (memory) |
| `noise` | per-neuron noise level |
| `thr` | firing threshold of the outputs |
| `log_mw` | the mutation step size itself (evolution tunes its own search) |

The only things fixed by hand are physics: the tensor size (at most 82 hidden neurons), `tanh`, the weight clamp
±5, and the rules of the world.

---

## The world and natural selection

```mermaid
flowchart TD
    P["3,000 genomes<br/>(lineages)"] --> B["each genome is born as 512 brains<br/>= 1,536,000 brains"]
    B --> G["brains are paired at random<br/>and play one game, both ticking continuously"]
    G --> S{"who survives the game?"}
    S -->|win on the board| W1["winner survives"]
    S -->|30 ticks on own turn<br/>without a legal move| W2["the other one survives"]
    S -->|draw| W3["fewer wasted spikes survives"]
    W1 & W2 & W3 --> F["fitness = share of its 512 games<br/>a genome survived"]
    F --> E["top 10% carry over unchanged<br/>the other 90% are their children:<br/>neuron-wise crossover + mutation<br/>1 child in 10 is a 'freak' mutated 20x harder"]
    E --> P
```

- **Brains only ever play each other.** There is no teacher in training. The random mover and the perfect
  minimax player appear only as **exams** (every 5th generation, logged, never selected on).
- **An illegal spike is ignored by the board**: out of turn, or on a taken cell. It costs nothing but a 0.001
  tie-break in a draw, and the clock keeps running. So no rule says "don't do that". Wasted spikes just don't
  help you survive.
- **No reward for a win, a legal move or waiting.** The game decides who survives. That's all.
- These rules were refined over five full runs. Each time evolution found a shortcut in the world's rules
  (out-waiting the opponent, banking a lead by ending the game early, a seat advantage), the rule was closed
  and evolution had to find the real skill instead. Every step is written up in [docs/DESIGN.md](docs/DESIGN.md).

---

## Results (run 5, the log included in this repo)

1,149 generations, 3,000 genomes x 512 brains, ~12 hours on one RTX A6000 (≈30 s per generation).
Numbers are taken straight from [`train_evo.log`](train_evo.log).

### 1. The network shrank itself to fit the task

The search started with big, randomly wired brains: about 57 of the 82 possible hidden neurons switched on, and
every neuron free to connect to every other one. Nobody told evolution how big the brain should be, and nothing
penalised size. Still, selection kept the lineages that **switched off neurons they did not need**, and every
synapse to and from those neurons went with them. The network decided its own size.

| Generation | 0 | 25 | 50 | 100 | 300 | 600 | 900 | 1149 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active hidden neurons in the best brain (of 82) | 57 | 39 | 34 | 25 | 27 | 29 | 27 | **26** |

**This is selection, not chance.** Mutation switches a neuron off exactly as often as it switches one on, so
random drift alone would settle at about half of them, 41 neurons. The brains went far below that and stayed
there. Across the *whole* population of 3,000 genomes the average is now **29 hidden neurons**. Smaller brains
won.

Counting only the synapses that can actually carry a signal (from a live neuron to a live hidden or output neuron;
the sensors are overwritten by the board every tick):

| | Live neurons | Working synapses |
|---|---:|---:|
| Fully connected, all 82 hidden on | 120 | 10,920 (100%) |
| Best brain, generation 0 | 95 | ≈ 3,100 (29%) |
| **Best brain, generation 1149** (`best_evo_01149.npz`, exact count) | **64** | **1,135 (10%)** |

The evolved brain does its job with about **a third of the computation per tick of the generation-0 brain**,
and **a tenth of a fully connected one**. On sparse or neuromorphic hardware, where a missing synapse costs
nothing, that means less energy per decision. Evolution found the lean design by itself, exactly as natural
selection does in biology.

### 2. It learned to wait, and to press once

Wasted spikes (firing out of turn, or on a taken cell) per brain per game, across the *whole* population:

| Generation | 0 | 100 | 300 | 600 | 900 | 1100 |
|---|---:|---:|---:|---:|---:|---:|
| wasted spikes / brain / game | 15.0 | 9.6 | 8.2 | 6.5 | 6.0 | **5.6** |

No rule forbids an out-of-turn spike; the board just ignores it. The brains discovered by selection alone that
the opponent's turn is for waiting.

### 3. It learned to stand up to a perfect player

At generation 0 the best brain lost almost every game against the perfect minimax player (which never loses).
Through evolution alone it learned to stand up to it, even though it had only ever played other brains: the
perfect player is an exam, never a teacher.

---

## Watch it: the live dashboard

`dashboard.py` serves a web page that reads the training log live (refreshing every few seconds) and shows:

- **Population play**: games that reach a board result, draws, timeouts, wasted spikes, ticks per game
- **Best lineage**: its outcomes and its genes (threshold, active neurons, connection density, noise, leak)
- **Exams**: draws vs the perfect player, wins vs the random mover
- **Live battle**: the newest saved brain plays itself or the perfect player, tick by tick. Every neuron and
  every strong synapse is drawn (blue = positive, red = negative), along with what each brain sees, what it
  remembers, and which output it wants to fire.

```bash
python dashboard.py
# open http://localhost:8766
```

It works straight after cloning: the run-5 log and the generation-1149 champion (`best_evo_01149.npz`) are
included. The battle runs on CPU if you have no GPU.

---

## Reproduce it

```bash
git clone https://github.com/kyung01/Alive-TicTacToe-AI.git
cd Alive-TicTacToe-AI
pip install -r requirements.txt      # numpy + PyTorch (install the CUDA build for your GPU)
```

**Full run 5 (the results above; run on one 48 GB RTX A6000, ~30 s per generation):**

```bash
python evo_rule.py --rules 3000 --brains 512 --chunk 300 --life 1 \
  --opponent peer --select pair --elite 0.1 --mutation 2.5 \
  --opp-delay 30 --reward-attempt -0.001
```

**Smaller run for a normal GPU** (same rules, fewer brains, noisier selection):

```bash
python evo_rule.py --rules 512 --brains 128 --chunk 128 --life 1 \
  --opponent peer --select pair --elite 0.1 --mutation 2.5 \
  --opp-delay 30 --reward-attempt -0.001 2>&1 | tee my_run.log
```

Then, in a second terminal (the log picker at the top of the page switches between `my_run.log` and the
included `train_evo.log`):

```bash
python dashboard.py --log my_run.log
```

- Each generation saves the whole population (`evo_pop_XXXXX.npz`, large, git-ignored) and the best brain
  (`best_evo_XXXXX.npz`). The last 3 of each are kept. The dashboard's live battle always shows the newest brain.
- `--resume latest` continues from the newest population.
- Create a file named `STOP` in the folder to save and exit cleanly.
- `python evo_rule.py --help` lists every option.

---

## Files

| File | Purpose |
|---|---|
| [`evo_rule.py`](evo_rule.py) | **The experiment**: genome, continuous game, pair selection, mutation, crossover, exams |
| [`brain_ttt_gpu.py`](brain_ttt_gpu.py), [`brain_ttt.py`](brain_ttt.py) | Shared game logic and the earlier learning-rule experiments |
| [`perfect.py`](perfect.py) | Perfect minimax player as a lookup table (exam only) |
| [`battle.py`](battle.py) | Replays a saved brain vs itself or vs the perfect player, recording every neuron every tick |
| [`dashboard.py`](dashboard.py) + [`dashboard.html`](dashboard.html) | Live training dashboard and battle viewer |
| [`train_evo.log`](train_evo.log) | Full log of run 5 (generations 0–1149) |
| [`best_evo_01149.npz`](best_evo_01149.npz) | The best evolved brain at generation 1149 |
| [`runs/`](runs) | Logs of runs 1–4, the earlier runs that shaped the final design |
| [`single.py`](single.py), [`continuous.py`](continuous.py), [`long_life.py`](long_life.py) | Earlier learning-rule experiments described in the lab notebook |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Full lab notebook: every design decision, every loophole, every fix |
| [`docs/`](docs) | The theory behind the project ([theory](docs/theory.md), [approach](docs/approach.md), [experiments](docs/experiments.md), [diary](docs/diary.md)) |

---

## The bigger question

Most of today's AI is a pipeline a human designed: fixed layers, a fixed learning rule, called once per input,
gone in between. Brains are not like that. They loop, they keep state, they choose their own timing, and their
wiring was shaped by selection, not drawn by an engineer.

This project is a small proof of concept for the other path: **provide primitives, never the plan.**
Give evolution neurons, synapses, memory, noise and a world with consequences, and let structure emerge.
Turn-taking, patience, restraint, a compact network, and solid tic-tac-toe all emerged here without being
programmed in.

Tic-tac-toe is tiny. The question it opens is not: what else emerges when we stop designing the mind and let it be
selected?

---

<sub>Author: <a href="https://github.com/kyung01">@kyung01</a>. Replications and contributions welcome: open an issue.</sub>
