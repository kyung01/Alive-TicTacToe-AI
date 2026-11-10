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

<p align="center">
  <a href="https://kyung01.github.io/Alive-TicTacToe-AI/brain3d/"><img src="assets/brain3d-preview.png" alt="The champion brain in 3D: the eye on the left, the hidden neurons evolution kept in the middle, the 9 outputs on the right"></a>
  <br>
  <b><a href="https://kyung01.github.io/Alive-TicTacToe-AI/brain3d/">▶ Open the champion brain in 3D</a></b>: rotate it, hover any neuron, watch it think, or play against it.
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

### The champion, in 3D

Drag to rotate the model below; GitHub draws it right here. Each solid is a live neuron and each rod is one of
the 220 strongest synapses of the generation-1149 champion. The eye is on the left (three 3x3 retinas: X, empty,
O), the outputs on the right (one per cell), and the hidden neurons evolution kept are in between. For colour,
signals flowing through it live, and a game against it, open the [interactive version](https://kyung01.github.io/Alive-TicTacToe-AI/brain3d/).

The hidden neurons are placed by a **rubber-band simulation**. Every live connection is one identical spring, the
same strength whatever its weight: it pulls when stretched and pushes when squeezed. Hidden neurons also push
each other apart to keep their distance. The eye and the outputs are pinned; switched-off neurons and connections
exert no force. Neurons that share many connections pull together, so the picture is the brain's real wiring.
The cloud sits closer to the eye because the hidden neurons are wired far more to the 29 eye receptors than to
the 9 outputs.

<!-- brain-stl:start -->

```stl
solid brain
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 0.55
vertex -7.55 -0 0.77
vertex -7.55 -0.22 0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 0.55
vertex -7.55 -0 0.77
vertex -7.55 0.22 0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 0.55
vertex -7.55 -0 0.33
vertex -7.55 -0.22 0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 0.55
vertex -7.55 -0 0.33
vertex -7.55 0.22 0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 0.55
vertex -7.55 -0 0.77
vertex -7.55 -0.22 0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 0.55
vertex -7.55 -0 0.77
vertex -7.55 0.22 0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 0.55
vertex -7.55 -0 0.33
vertex -7.55 -0.22 0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 0.55
vertex -7.55 -0 0.33
vertex -7.55 0.22 0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 0.55
vertex -7 -0 0.77
vertex -7 -0.22 0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 0.55
vertex -7 -0 0.77
vertex -7 0.22 0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 0.55
vertex -7 -0 0.33
vertex -7 -0.22 0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 0.55
vertex -7 -0 0.33
vertex -7 0.22 0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 0.55
vertex -7 -0 0.77
vertex -7 -0.22 0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 0.55
vertex -7 -0 0.77
vertex -7 0.22 0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 0.55
vertex -7 -0 0.33
vertex -7 -0.22 0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 0.55
vertex -7 -0 0.33
vertex -7 0.22 0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 0.55
vertex -6.45 -0 0.77
vertex -6.45 -0.22 0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 0.55
vertex -6.45 -0 0.77
vertex -6.45 0.22 0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 0.55
vertex -6.45 -0 0.33
vertex -6.45 -0.22 0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 0.55
vertex -6.45 -0 0.33
vertex -6.45 0.22 0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 0.55
vertex -6.45 -0 0.77
vertex -6.45 -0.22 0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 0.55
vertex -6.45 -0 0.77
vertex -6.45 0.22 0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 0.55
vertex -6.45 -0 0.33
vertex -6.45 -0.22 0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 0.55
vertex -6.45 -0 0.33
vertex -6.45 0.22 0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 0
vertex -7.55 -0 0.22
vertex -7.55 -0.22 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 0
vertex -7.55 -0 0.22
vertex -7.55 0.22 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 0
vertex -7.55 -0 -0.22
vertex -7.55 -0.22 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 0
vertex -7.55 -0 -0.22
vertex -7.55 0.22 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 0
vertex -7.55 -0 0.22
vertex -7.55 -0.22 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 0
vertex -7.55 -0 0.22
vertex -7.55 0.22 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 0
vertex -7.55 -0 -0.22
vertex -7.55 -0.22 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 0
vertex -7.55 -0 -0.22
vertex -7.55 0.22 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 0
vertex -7 -0 0.22
vertex -7 -0.22 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 0
vertex -7 -0 0.22
vertex -7 0.22 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 0
vertex -7 -0 -0.22
vertex -7 -0.22 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 0
vertex -7 -0 -0.22
vertex -7 0.22 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 0
vertex -7 -0 0.22
vertex -7 -0.22 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 0
vertex -7 -0 0.22
vertex -7 0.22 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 0
vertex -7 -0 -0.22
vertex -7 -0.22 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 0
vertex -7 -0 -0.22
vertex -7 0.22 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 0
vertex -6.45 -0 0.22
vertex -6.45 -0.22 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 0
vertex -6.45 -0 0.22
vertex -6.45 0.22 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 0
vertex -6.45 -0 -0.22
vertex -6.45 -0.22 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 0
vertex -6.45 -0 -0.22
vertex -6.45 0.22 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 0
vertex -6.45 -0 0.22
vertex -6.45 -0.22 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 0
vertex -6.45 -0 0.22
vertex -6.45 0.22 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 0
vertex -6.45 -0 -0.22
vertex -6.45 -0.22 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 0
vertex -6.45 -0 -0.22
vertex -6.45 0.22 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 -0.55
vertex -7.55 -0 -0.33
vertex -7.55 -0.22 -0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 -0.55
vertex -7.55 -0 -0.33
vertex -7.55 0.22 -0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 -0.55
vertex -7.55 -0 -0.77
vertex -7.55 -0.22 -0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 -0.55
vertex -7.55 -0 -0.77
vertex -7.55 0.22 -0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 -0.55
vertex -7.55 -0 -0.33
vertex -7.55 -0.22 -0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 -0.55
vertex -7.55 -0 -0.33
vertex -7.55 0.22 -0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 -0.55
vertex -7.55 -0 -0.77
vertex -7.55 -0.22 -0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 -0.55
vertex -7.55 -0 -0.77
vertex -7.55 0.22 -0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 -0.55
vertex -7 -0 -0.33
vertex -7 -0.22 -0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 -0.55
vertex -7 -0 -0.33
vertex -7 0.22 -0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 -0.55
vertex -7 -0 -0.77
vertex -7 -0.22 -0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 -0.55
vertex -7 -0 -0.77
vertex -7 0.22 -0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 -0.55
vertex -7 -0 -0.33
vertex -7 -0.22 -0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 -0.55
vertex -7 -0 -0.33
vertex -7 0.22 -0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 -0.55
vertex -7 -0 -0.77
vertex -7 -0.22 -0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 -0.55
vertex -7 -0 -0.77
vertex -7 0.22 -0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 -0.55
vertex -6.45 -0 -0.33
vertex -6.45 -0.22 -0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 -0.55
vertex -6.45 -0 -0.33
vertex -6.45 0.22 -0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 -0.55
vertex -6.45 -0 -0.77
vertex -6.45 -0.22 -0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 -0.55
vertex -6.45 -0 -0.77
vertex -6.45 0.22 -0.55
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 -0.55
vertex -6.45 -0 -0.33
vertex -6.45 -0.22 -0.55
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 -0.55
vertex -6.45 -0 -0.33
vertex -6.45 0.22 -0.55
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 -0.55
vertex -6.45 -0 -0.77
vertex -6.45 -0.22 -0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 -0.55
vertex -6.45 -0 -0.77
vertex -6.45 0.22 -0.55
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 3.15
vertex -7.55 -0 3.37
vertex -7.55 -0.22 3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 3.15
vertex -7.55 -0 3.37
vertex -7.55 0.22 3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 3.15
vertex -7.55 -0 2.93
vertex -7.55 -0.22 3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 3.15
vertex -7.55 -0 2.93
vertex -7.55 0.22 3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 3.15
vertex -7.55 -0 3.37
vertex -7.55 -0.22 3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 3.15
vertex -7.55 -0 3.37
vertex -7.55 0.22 3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 3.15
vertex -7.55 -0 2.93
vertex -7.55 -0.22 3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 3.15
vertex -7.55 -0 2.93
vertex -7.55 0.22 3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 3.15
vertex -7 -0 3.37
vertex -7 -0.22 3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 3.15
vertex -7 -0 3.37
vertex -7 0.22 3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 3.15
vertex -7 -0 2.93
vertex -7 -0.22 3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 3.15
vertex -7 -0 2.93
vertex -7 0.22 3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 3.15
vertex -7 -0 3.37
vertex -7 -0.22 3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 3.15
vertex -7 -0 3.37
vertex -7 0.22 3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 3.15
vertex -7 -0 2.93
vertex -7 -0.22 3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 3.15
vertex -7 -0 2.93
vertex -7 0.22 3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 3.15
vertex -6.45 -0 3.37
vertex -6.45 -0.22 3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 3.15
vertex -6.45 -0 3.37
vertex -6.45 0.22 3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 3.15
vertex -6.45 -0 2.93
vertex -6.45 -0.22 3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 3.15
vertex -6.45 -0 2.93
vertex -6.45 0.22 3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 3.15
vertex -6.45 -0 3.37
vertex -6.45 -0.22 3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 3.15
vertex -6.45 -0 3.37
vertex -6.45 0.22 3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 3.15
vertex -6.45 -0 2.93
vertex -6.45 -0.22 3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 3.15
vertex -6.45 -0 2.93
vertex -6.45 0.22 3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 2.6
vertex -7.55 -0 2.82
vertex -7.55 -0.22 2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 2.6
vertex -7.55 -0 2.82
vertex -7.55 0.22 2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 2.6
vertex -7.55 -0 2.38
vertex -7.55 -0.22 2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 2.6
vertex -7.55 -0 2.38
vertex -7.55 0.22 2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 2.6
vertex -7.55 -0 2.82
vertex -7.55 -0.22 2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 2.6
vertex -7.55 -0 2.82
vertex -7.55 0.22 2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 2.6
vertex -7.55 -0 2.38
vertex -7.55 -0.22 2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 2.6
vertex -7.55 -0 2.38
vertex -7.55 0.22 2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 2.6
vertex -7 -0 2.82
vertex -7 -0.22 2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 2.6
vertex -7 -0 2.82
vertex -7 0.22 2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 2.6
vertex -7 -0 2.38
vertex -7 -0.22 2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 2.6
vertex -7 -0 2.38
vertex -7 0.22 2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 2.6
vertex -7 -0 2.82
vertex -7 -0.22 2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 2.6
vertex -7 -0 2.82
vertex -7 0.22 2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 2.6
vertex -7 -0 2.38
vertex -7 -0.22 2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 2.6
vertex -7 -0 2.38
vertex -7 0.22 2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 2.6
vertex -6.45 -0 2.82
vertex -6.45 -0.22 2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 2.6
vertex -6.45 -0 2.82
vertex -6.45 0.22 2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 2.6
vertex -6.45 -0 2.38
vertex -6.45 -0.22 2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 2.6
vertex -6.45 -0 2.38
vertex -6.45 0.22 2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 2.6
vertex -6.45 -0 2.82
vertex -6.45 -0.22 2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 2.6
vertex -6.45 -0 2.82
vertex -6.45 0.22 2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 2.6
vertex -6.45 -0 2.38
vertex -6.45 -0.22 2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 2.6
vertex -6.45 -0 2.38
vertex -6.45 0.22 2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 2.05
vertex -7.55 -0 2.27
vertex -7.55 -0.22 2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 2.05
vertex -7.55 -0 2.27
vertex -7.55 0.22 2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 2.05
vertex -7.55 -0 1.83
vertex -7.55 -0.22 2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 2.05
vertex -7.55 -0 1.83
vertex -7.55 0.22 2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 2.05
vertex -7.55 -0 2.27
vertex -7.55 -0.22 2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 2.05
vertex -7.55 -0 2.27
vertex -7.55 0.22 2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 2.05
vertex -7.55 -0 1.83
vertex -7.55 -0.22 2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 2.05
vertex -7.55 -0 1.83
vertex -7.55 0.22 2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 2.05
vertex -7 -0 2.27
vertex -7 -0.22 2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 2.05
vertex -7 -0 2.27
vertex -7 0.22 2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 2.05
vertex -7 -0 1.83
vertex -7 -0.22 2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 2.05
vertex -7 -0 1.83
vertex -7 0.22 2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 2.05
vertex -7 -0 2.27
vertex -7 -0.22 2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 2.05
vertex -7 -0 2.27
vertex -7 0.22 2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 2.05
vertex -7 -0 1.83
vertex -7 -0.22 2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 2.05
vertex -7 -0 1.83
vertex -7 0.22 2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 2.05
vertex -6.45 -0 2.27
vertex -6.45 -0.22 2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 2.05
vertex -6.45 -0 2.27
vertex -6.45 0.22 2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 2.05
vertex -6.45 -0 1.83
vertex -6.45 -0.22 2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 2.05
vertex -6.45 -0 1.83
vertex -6.45 0.22 2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 2.05
vertex -6.45 -0 2.27
vertex -6.45 -0.22 2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 2.05
vertex -6.45 -0 2.27
vertex -6.45 0.22 2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 2.05
vertex -6.45 -0 1.83
vertex -6.45 -0.22 2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 2.05
vertex -6.45 -0 1.83
vertex -6.45 0.22 2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 -2.05
vertex -7.55 -0 -1.83
vertex -7.55 -0.22 -2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 -2.05
vertex -7.55 -0 -1.83
vertex -7.55 0.22 -2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 -2.05
vertex -7.55 -0 -2.27
vertex -7.55 -0.22 -2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 -2.05
vertex -7.55 -0 -2.27
vertex -7.55 0.22 -2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 -2.05
vertex -7.55 -0 -1.83
vertex -7.55 -0.22 -2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 -2.05
vertex -7.55 -0 -1.83
vertex -7.55 0.22 -2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 -2.05
vertex -7.55 -0 -2.27
vertex -7.55 -0.22 -2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 -2.05
vertex -7.55 -0 -2.27
vertex -7.55 0.22 -2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 -2.05
vertex -7 -0 -1.83
vertex -7 -0.22 -2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 -2.05
vertex -7 -0 -1.83
vertex -7 0.22 -2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 -2.05
vertex -7 -0 -2.27
vertex -7 -0.22 -2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 -2.05
vertex -7 -0 -2.27
vertex -7 0.22 -2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 -2.05
vertex -7 -0 -1.83
vertex -7 -0.22 -2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 -2.05
vertex -7 -0 -1.83
vertex -7 0.22 -2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 -2.05
vertex -7 -0 -2.27
vertex -7 -0.22 -2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 -2.05
vertex -7 -0 -2.27
vertex -7 0.22 -2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 -2.05
vertex -6.45 -0 -1.83
vertex -6.45 -0.22 -2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 -2.05
vertex -6.45 -0 -1.83
vertex -6.45 0.22 -2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 -2.05
vertex -6.45 -0 -2.27
vertex -6.45 -0.22 -2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 -2.05
vertex -6.45 -0 -2.27
vertex -6.45 0.22 -2.05
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 -2.05
vertex -6.45 -0 -1.83
vertex -6.45 -0.22 -2.05
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 -2.05
vertex -6.45 -0 -1.83
vertex -6.45 0.22 -2.05
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 -2.05
vertex -6.45 -0 -2.27
vertex -6.45 -0.22 -2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 -2.05
vertex -6.45 -0 -2.27
vertex -6.45 0.22 -2.05
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 -2.6
vertex -7.55 -0 -2.38
vertex -7.55 -0.22 -2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 -2.6
vertex -7.55 -0 -2.38
vertex -7.55 0.22 -2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 -2.6
vertex -7.55 -0 -2.82
vertex -7.55 -0.22 -2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 -2.6
vertex -7.55 -0 -2.82
vertex -7.55 0.22 -2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 -2.6
vertex -7.55 -0 -2.38
vertex -7.55 -0.22 -2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 -2.6
vertex -7.55 -0 -2.38
vertex -7.55 0.22 -2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 -2.6
vertex -7.55 -0 -2.82
vertex -7.55 -0.22 -2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 -2.6
vertex -7.55 -0 -2.82
vertex -7.55 0.22 -2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 -2.6
vertex -7 -0 -2.38
vertex -7 -0.22 -2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 -2.6
vertex -7 -0 -2.38
vertex -7 0.22 -2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 -2.6
vertex -7 -0 -2.82
vertex -7 -0.22 -2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 -2.6
vertex -7 -0 -2.82
vertex -7 0.22 -2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 -2.6
vertex -7 -0 -2.38
vertex -7 -0.22 -2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 -2.6
vertex -7 -0 -2.38
vertex -7 0.22 -2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 -2.6
vertex -7 -0 -2.82
vertex -7 -0.22 -2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 -2.6
vertex -7 -0 -2.82
vertex -7 0.22 -2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 -2.6
vertex -6.45 -0 -2.38
vertex -6.45 -0.22 -2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 -2.6
vertex -6.45 -0 -2.38
vertex -6.45 0.22 -2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 -2.6
vertex -6.45 -0 -2.82
vertex -6.45 -0.22 -2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 -2.6
vertex -6.45 -0 -2.82
vertex -6.45 0.22 -2.6
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 -2.6
vertex -6.45 -0 -2.38
vertex -6.45 -0.22 -2.6
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 -2.6
vertex -6.45 -0 -2.38
vertex -6.45 0.22 -2.6
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 -2.6
vertex -6.45 -0 -2.82
vertex -6.45 -0.22 -2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 -2.6
vertex -6.45 -0 -2.82
vertex -6.45 0.22 -2.6
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.33 -0 -3.15
vertex -7.55 -0 -2.93
vertex -7.55 -0.22 -3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.33 -0 -3.15
vertex -7.55 -0 -2.93
vertex -7.55 0.22 -3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.33 -0 -3.15
vertex -7.55 -0 -3.37
vertex -7.55 -0.22 -3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.33 -0 -3.15
vertex -7.55 -0 -3.37
vertex -7.55 0.22 -3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.77 -0 -3.15
vertex -7.55 -0 -2.93
vertex -7.55 -0.22 -3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.77 -0 -3.15
vertex -7.55 -0 -2.93
vertex -7.55 0.22 -3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.77 -0 -3.15
vertex -7.55 -0 -3.37
vertex -7.55 -0.22 -3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.77 -0 -3.15
vertex -7.55 -0 -3.37
vertex -7.55 0.22 -3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.78 -0 -3.15
vertex -7 -0 -2.93
vertex -7 -0.22 -3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.78 -0 -3.15
vertex -7 -0 -2.93
vertex -7 0.22 -3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.78 -0 -3.15
vertex -7 -0 -3.37
vertex -7 -0.22 -3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.78 -0 -3.15
vertex -7 -0 -3.37
vertex -7 0.22 -3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.22 -0 -3.15
vertex -7 -0 -2.93
vertex -7 -0.22 -3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.22 -0 -3.15
vertex -7 -0 -2.93
vertex -7 0.22 -3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.22 -0 -3.15
vertex -7 -0 -3.37
vertex -7 -0.22 -3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.22 -0 -3.15
vertex -7 -0 -3.37
vertex -7 0.22 -3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.23 -0 -3.15
vertex -6.45 -0 -2.93
vertex -6.45 -0.22 -3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.23 -0 -3.15
vertex -6.45 -0 -2.93
vertex -6.45 0.22 -3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.23 -0 -3.15
vertex -6.45 -0 -3.37
vertex -6.45 -0.22 -3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.23 -0 -3.15
vertex -6.45 -0 -3.37
vertex -6.45 0.22 -3.15
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.67 -0 -3.15
vertex -6.45 -0 -2.93
vertex -6.45 -0.22 -3.15
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.67 -0 -3.15
vertex -6.45 -0 -2.93
vertex -6.45 0.22 -3.15
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.67 -0 -3.15
vertex -6.45 -0 -3.37
vertex -6.45 -0.22 -3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.67 -0 -3.15
vertex -6.45 -0 -3.37
vertex -6.45 0.22 -3.15
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.08 -0 -4.4
vertex -7.3 -0 -4.18
vertex -7.3 -0.22 -4.4
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.08 -0 -4.4
vertex -7.3 -0 -4.18
vertex -7.3 0.22 -4.4
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.08 -0 -4.4
vertex -7.3 -0 -4.62
vertex -7.3 -0.22 -4.4
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.08 -0 -4.4
vertex -7.3 -0 -4.62
vertex -7.3 0.22 -4.4
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -7.52 -0 -4.4
vertex -7.3 -0 -4.18
vertex -7.3 -0.22 -4.4
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -7.52 -0 -4.4
vertex -7.3 -0 -4.18
vertex -7.3 0.22 -4.4
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -7.52 -0 -4.4
vertex -7.3 -0 -4.62
vertex -7.3 -0.22 -4.4
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -7.52 -0 -4.4
vertex -7.3 -0 -4.62
vertex -7.3 0.22 -4.4
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.48 -0 -4.4
vertex -6.7 -0 -4.18
vertex -6.7 -0.22 -4.4
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.48 -0 -4.4
vertex -6.7 -0 -4.18
vertex -6.7 0.22 -4.4
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.48 -0 -4.4
vertex -6.7 -0 -4.62
vertex -6.7 -0.22 -4.4
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.48 -0 -4.4
vertex -6.7 -0 -4.62
vertex -6.7 0.22 -4.4
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -6.92 -0 -4.4
vertex -6.7 -0 -4.18
vertex -6.7 -0.22 -4.4
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -6.92 -0 -4.4
vertex -6.7 -0 -4.18
vertex -6.7 0.22 -4.4
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -6.92 -0 -4.4
vertex -6.7 -0 -4.62
vertex -6.7 -0.22 -4.4
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -6.92 -0 -4.4
vertex -6.7 -0 -4.62
vertex -6.7 0.22 -4.4
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 1.97 1.43 -1.22
vertex 1.57 1.43 -0.82
vertex 1.57 1.03 -1.22
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 1.97 1.43 -1.22
vertex 1.57 1.43 -0.82
vertex 1.57 1.83 -1.22
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 1.97 1.43 -1.22
vertex 1.57 1.43 -1.62
vertex 1.57 1.03 -1.22
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 1.97 1.43 -1.22
vertex 1.57 1.43 -1.62
vertex 1.57 1.83 -1.22
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 1.17 1.43 -1.22
vertex 1.57 1.43 -0.82
vertex 1.57 1.03 -1.22
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 1.17 1.43 -1.22
vertex 1.57 1.43 -0.82
vertex 1.57 1.83 -1.22
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 1.17 1.43 -1.22
vertex 1.57 1.43 -1.62
vertex 1.57 1.03 -1.22
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 1.17 1.43 -1.22
vertex 1.57 1.43 -1.62
vertex 1.57 1.83 -1.22
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.79 0.93 -3.25
vertex -1.19 0.93 -2.85
vertex -1.19 0.53 -3.25
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.79 0.93 -3.25
vertex -1.19 0.93 -2.85
vertex -1.19 1.33 -3.25
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.79 0.93 -3.25
vertex -1.19 0.93 -3.65
vertex -1.19 0.53 -3.25
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.79 0.93 -3.25
vertex -1.19 0.93 -3.65
vertex -1.19 1.33 -3.25
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.59 0.93 -3.25
vertex -1.19 0.93 -2.85
vertex -1.19 0.53 -3.25
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.59 0.93 -3.25
vertex -1.19 0.93 -2.85
vertex -1.19 1.33 -3.25
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.59 0.93 -3.25
vertex -1.19 0.93 -3.65
vertex -1.19 0.53 -3.25
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.59 0.93 -3.25
vertex -1.19 0.93 -3.65
vertex -1.19 1.33 -3.25
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 1.54 -0.89 -0.97
vertex 1.14 -0.89 -0.57
vertex 1.14 -1.29 -0.97
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 1.54 -0.89 -0.97
vertex 1.14 -0.89 -0.57
vertex 1.14 -0.49 -0.97
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 1.54 -0.89 -0.97
vertex 1.14 -0.89 -1.37
vertex 1.14 -1.29 -0.97
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 1.54 -0.89 -0.97
vertex 1.14 -0.89 -1.37
vertex 1.14 -0.49 -0.97
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 0.74 -0.89 -0.97
vertex 1.14 -0.89 -0.57
vertex 1.14 -1.29 -0.97
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 0.74 -0.89 -0.97
vertex 1.14 -0.89 -0.57
vertex 1.14 -0.49 -0.97
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 0.74 -0.89 -0.97
vertex 1.14 -0.89 -1.37
vertex 1.14 -1.29 -0.97
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 0.74 -0.89 -0.97
vertex 1.14 -0.89 -1.37
vertex 1.14 -0.49 -0.97
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.51 1.61 0.38
vertex -3.91 1.61 0.78
vertex -3.91 1.21 0.38
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.51 1.61 0.38
vertex -3.91 1.61 0.78
vertex -3.91 2.01 0.38
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.51 1.61 0.38
vertex -3.91 1.61 -0.02
vertex -3.91 1.21 0.38
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.51 1.61 0.38
vertex -3.91 1.61 -0.02
vertex -3.91 2.01 0.38
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -4.31 1.61 0.38
vertex -3.91 1.61 0.78
vertex -3.91 1.21 0.38
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -4.31 1.61 0.38
vertex -3.91 1.61 0.78
vertex -3.91 2.01 0.38
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -4.31 1.61 0.38
vertex -3.91 1.61 -0.02
vertex -3.91 1.21 0.38
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -4.31 1.61 0.38
vertex -3.91 1.61 -0.02
vertex -3.91 2.01 0.38
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.39 0.93 0.7
vertex -1.79 0.93 1.1
vertex -1.79 0.53 0.7
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.39 0.93 0.7
vertex -1.79 0.93 1.1
vertex -1.79 1.33 0.7
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.39 0.93 0.7
vertex -1.79 0.93 0.3
vertex -1.79 0.53 0.7
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.39 0.93 0.7
vertex -1.79 0.93 0.3
vertex -1.79 1.33 0.7
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.19 0.93 0.7
vertex -1.79 0.93 1.1
vertex -1.79 0.53 0.7
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.19 0.93 0.7
vertex -1.79 0.93 1.1
vertex -1.79 1.33 0.7
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.19 0.93 0.7
vertex -1.79 0.93 0.3
vertex -1.79 0.53 0.7
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.19 0.93 0.7
vertex -1.79 0.93 0.3
vertex -1.79 1.33 0.7
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.59 -2.39 1.14
vertex -1.99 -2.39 1.54
vertex -1.99 -2.79 1.14
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.59 -2.39 1.14
vertex -1.99 -2.39 1.54
vertex -1.99 -1.99 1.14
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.59 -2.39 1.14
vertex -1.99 -2.39 0.74
vertex -1.99 -2.79 1.14
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.59 -2.39 1.14
vertex -1.99 -2.39 0.74
vertex -1.99 -1.99 1.14
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.39 -2.39 1.14
vertex -1.99 -2.39 1.54
vertex -1.99 -2.79 1.14
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.39 -2.39 1.14
vertex -1.99 -2.39 1.54
vertex -1.99 -1.99 1.14
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.39 -2.39 1.14
vertex -1.99 -2.39 0.74
vertex -1.99 -2.79 1.14
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.39 -2.39 1.14
vertex -1.99 -2.39 0.74
vertex -1.99 -1.99 1.14
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.39 -2.39 -0.31
vertex -3.79 -2.39 0.09
vertex -3.79 -2.79 -0.31
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.39 -2.39 -0.31
vertex -3.79 -2.39 0.09
vertex -3.79 -1.99 -0.31
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.39 -2.39 -0.31
vertex -3.79 -2.39 -0.71
vertex -3.79 -2.79 -0.31
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.39 -2.39 -0.31
vertex -3.79 -2.39 -0.71
vertex -3.79 -1.99 -0.31
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -4.19 -2.39 -0.31
vertex -3.79 -2.39 0.09
vertex -3.79 -2.79 -0.31
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -4.19 -2.39 -0.31
vertex -3.79 -2.39 0.09
vertex -3.79 -1.99 -0.31
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -4.19 -2.39 -0.31
vertex -3.79 -2.39 -0.71
vertex -3.79 -2.79 -0.31
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -4.19 -2.39 -0.31
vertex -3.79 -2.39 -0.71
vertex -3.79 -1.99 -0.31
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.32 -0.58 2.61
vertex -1.72 -0.58 3.01
vertex -1.72 -0.98 2.61
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.32 -0.58 2.61
vertex -1.72 -0.58 3.01
vertex -1.72 -0.18 2.61
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.32 -0.58 2.61
vertex -1.72 -0.58 2.21
vertex -1.72 -0.98 2.61
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.32 -0.58 2.61
vertex -1.72 -0.58 2.21
vertex -1.72 -0.18 2.61
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.12 -0.58 2.61
vertex -1.72 -0.58 3.01
vertex -1.72 -0.98 2.61
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.12 -0.58 2.61
vertex -1.72 -0.58 3.01
vertex -1.72 -0.18 2.61
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.12 -0.58 2.61
vertex -1.72 -0.58 2.21
vertex -1.72 -0.98 2.61
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.12 -0.58 2.61
vertex -1.72 -0.58 2.21
vertex -1.72 -0.18 2.61
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.1 0.59 -1.1
vertex -0.5 0.59 -0.7
vertex -0.5 0.19 -1.1
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.1 0.59 -1.1
vertex -0.5 0.59 -0.7
vertex -0.5 0.99 -1.1
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.1 0.59 -1.1
vertex -0.5 0.59 -1.5
vertex -0.5 0.19 -1.1
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.1 0.59 -1.1
vertex -0.5 0.59 -1.5
vertex -0.5 0.99 -1.1
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.9 0.59 -1.1
vertex -0.5 0.59 -0.7
vertex -0.5 0.19 -1.1
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.9 0.59 -1.1
vertex -0.5 0.59 -0.7
vertex -0.5 0.99 -1.1
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.9 0.59 -1.1
vertex -0.5 0.59 -1.5
vertex -0.5 0.19 -1.1
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.9 0.59 -1.1
vertex -0.5 0.59 -1.5
vertex -0.5 0.99 -1.1
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.84 -2.73 -2.03
vertex -2.24 -2.73 -1.63
vertex -2.24 -3.13 -2.03
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.84 -2.73 -2.03
vertex -2.24 -2.73 -1.63
vertex -2.24 -2.33 -2.03
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.84 -2.73 -2.03
vertex -2.24 -2.73 -2.43
vertex -2.24 -3.13 -2.03
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.84 -2.73 -2.03
vertex -2.24 -2.73 -2.43
vertex -2.24 -2.33 -2.03
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.64 -2.73 -2.03
vertex -2.24 -2.73 -1.63
vertex -2.24 -3.13 -2.03
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.64 -2.73 -2.03
vertex -2.24 -2.73 -1.63
vertex -2.24 -2.33 -2.03
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.64 -2.73 -2.03
vertex -2.24 -2.73 -2.43
vertex -2.24 -3.13 -2.03
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.64 -2.73 -2.03
vertex -2.24 -2.73 -2.43
vertex -2.24 -2.33 -2.03
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.84 -0.21 -1.01
vertex -4.24 -0.21 -0.61
vertex -4.24 -0.61 -1.01
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.84 -0.21 -1.01
vertex -4.24 -0.21 -0.61
vertex -4.24 0.19 -1.01
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.84 -0.21 -1.01
vertex -4.24 -0.21 -1.41
vertex -4.24 -0.61 -1.01
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.84 -0.21 -1.01
vertex -4.24 -0.21 -1.41
vertex -4.24 0.19 -1.01
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -4.64 -0.21 -1.01
vertex -4.24 -0.21 -0.61
vertex -4.24 -0.61 -1.01
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -4.64 -0.21 -1.01
vertex -4.24 -0.21 -0.61
vertex -4.24 0.19 -1.01
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -4.64 -0.21 -1.01
vertex -4.24 -0.21 -1.41
vertex -4.24 -0.61 -1.01
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -4.64 -0.21 -1.01
vertex -4.24 -0.21 -1.41
vertex -4.24 0.19 -1.01
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.02 1.33 2.38
vertex -0.42 1.33 2.78
vertex -0.42 0.93 2.38
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.02 1.33 2.38
vertex -0.42 1.33 2.78
vertex -0.42 1.73 2.38
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.02 1.33 2.38
vertex -0.42 1.33 1.98
vertex -0.42 0.93 2.38
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.02 1.33 2.38
vertex -0.42 1.33 1.98
vertex -0.42 1.73 2.38
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.82 1.33 2.38
vertex -0.42 1.33 2.78
vertex -0.42 0.93 2.38
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.82 1.33 2.38
vertex -0.42 1.33 2.78
vertex -0.42 1.73 2.38
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.82 1.33 2.38
vertex -0.42 1.33 1.98
vertex -0.42 0.93 2.38
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.82 1.33 2.38
vertex -0.42 1.33 1.98
vertex -0.42 1.73 2.38
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 0.98 -0.77 2.74
vertex 0.58 -0.77 3.14
vertex 0.58 -1.17 2.74
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 0.98 -0.77 2.74
vertex 0.58 -0.77 3.14
vertex 0.58 -0.37 2.74
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 0.98 -0.77 2.74
vertex 0.58 -0.77 2.34
vertex 0.58 -1.17 2.74
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 0.98 -0.77 2.74
vertex 0.58 -0.77 2.34
vertex 0.58 -0.37 2.74
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 0.18 -0.77 2.74
vertex 0.58 -0.77 3.14
vertex 0.58 -1.17 2.74
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 0.18 -0.77 2.74
vertex 0.58 -0.77 3.14
vertex 0.58 -0.37 2.74
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 0.18 -0.77 2.74
vertex 0.58 -0.77 2.34
vertex 0.58 -1.17 2.74
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 0.18 -0.77 2.74
vertex 0.58 -0.77 2.34
vertex 0.58 -0.37 2.74
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.21 -1.19 -2.53
vertex -0.61 -1.19 -2.13
vertex -0.61 -1.59 -2.53
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.21 -1.19 -2.53
vertex -0.61 -1.19 -2.13
vertex -0.61 -0.79 -2.53
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.21 -1.19 -2.53
vertex -0.61 -1.19 -2.93
vertex -0.61 -1.59 -2.53
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.21 -1.19 -2.53
vertex -0.61 -1.19 -2.93
vertex -0.61 -0.79 -2.53
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.01 -1.19 -2.53
vertex -0.61 -1.19 -2.13
vertex -0.61 -1.59 -2.53
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.01 -1.19 -2.53
vertex -0.61 -1.19 -2.13
vertex -0.61 -0.79 -2.53
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.01 -1.19 -2.53
vertex -0.61 -1.19 -2.93
vertex -0.61 -1.59 -2.53
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.01 -1.19 -2.53
vertex -0.61 -1.19 -2.93
vertex -0.61 -0.79 -2.53
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.7 -0.89 -0.69
vertex -2.1 -0.89 -0.29
vertex -2.1 -1.29 -0.69
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.7 -0.89 -0.69
vertex -2.1 -0.89 -0.29
vertex -2.1 -0.49 -0.69
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.7 -0.89 -0.69
vertex -2.1 -0.89 -1.09
vertex -2.1 -1.29 -0.69
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.7 -0.89 -0.69
vertex -2.1 -0.89 -1.09
vertex -2.1 -0.49 -0.69
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.5 -0.89 -0.69
vertex -2.1 -0.89 -0.29
vertex -2.1 -1.29 -0.69
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.5 -0.89 -0.69
vertex -2.1 -0.89 -0.29
vertex -2.1 -0.49 -0.69
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.5 -0.89 -0.69
vertex -2.1 -0.89 -1.09
vertex -2.1 -1.29 -0.69
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.5 -0.89 -0.69
vertex -2.1 -0.89 -1.09
vertex -2.1 -0.49 -0.69
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 0.41 2.33 0.33
vertex 0.01 2.33 0.73
vertex 0.01 1.93 0.33
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 0.41 2.33 0.33
vertex 0.01 2.33 0.73
vertex 0.01 2.73 0.33
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 0.41 2.33 0.33
vertex 0.01 2.33 -0.07
vertex 0.01 1.93 0.33
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 0.41 2.33 0.33
vertex 0.01 2.33 -0.07
vertex 0.01 2.73 0.33
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.39 2.33 0.33
vertex 0.01 2.33 0.73
vertex 0.01 1.93 0.33
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.39 2.33 0.33
vertex 0.01 2.33 0.73
vertex 0.01 2.73 0.33
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.39 2.33 0.33
vertex 0.01 2.33 -0.07
vertex 0.01 1.93 0.33
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.39 2.33 0.33
vertex 0.01 2.33 -0.07
vertex 0.01 2.73 0.33
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.03 -2.55 -0.64
vertex -0.43 -2.55 -0.24
vertex -0.43 -2.95 -0.64
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.03 -2.55 -0.64
vertex -0.43 -2.55 -0.24
vertex -0.43 -2.15 -0.64
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.03 -2.55 -0.64
vertex -0.43 -2.55 -1.04
vertex -0.43 -2.95 -0.64
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.03 -2.55 -0.64
vertex -0.43 -2.55 -1.04
vertex -0.43 -2.15 -0.64
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.83 -2.55 -0.64
vertex -0.43 -2.55 -0.24
vertex -0.43 -2.95 -0.64
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.83 -2.55 -0.64
vertex -0.43 -2.55 -0.24
vertex -0.43 -2.15 -0.64
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.83 -2.55 -0.64
vertex -0.43 -2.55 -1.04
vertex -0.43 -2.95 -0.64
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.83 -2.55 -0.64
vertex -0.43 -2.55 -1.04
vertex -0.43 -2.15 -0.64
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.09 -0.58 1.16
vertex -3.49 -0.58 1.56
vertex -3.49 -0.98 1.16
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.09 -0.58 1.16
vertex -3.49 -0.58 1.56
vertex -3.49 -0.18 1.16
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.09 -0.58 1.16
vertex -3.49 -0.58 0.76
vertex -3.49 -0.98 1.16
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.09 -0.58 1.16
vertex -3.49 -0.58 0.76
vertex -3.49 -0.18 1.16
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.89 -0.58 1.16
vertex -3.49 -0.58 1.56
vertex -3.49 -0.98 1.16
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.89 -0.58 1.16
vertex -3.49 -0.58 1.56
vertex -3.49 -0.18 1.16
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.89 -0.58 1.16
vertex -3.49 -0.58 0.76
vertex -3.49 -0.98 1.16
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.89 -0.58 1.16
vertex -3.49 -0.58 0.76
vertex -3.49 -0.18 1.16
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.67 1.3 2.54
vertex -3.07 1.3 2.94
vertex -3.07 0.9 2.54
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.67 1.3 2.54
vertex -3.07 1.3 2.94
vertex -3.07 1.7 2.54
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.67 1.3 2.54
vertex -3.07 1.3 2.14
vertex -3.07 0.9 2.54
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.67 1.3 2.54
vertex -3.07 1.3 2.14
vertex -3.07 1.7 2.54
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.47 1.3 2.54
vertex -3.07 1.3 2.94
vertex -3.07 0.9 2.54
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.47 1.3 2.54
vertex -3.07 1.3 2.94
vertex -3.07 1.7 2.54
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.47 1.3 2.54
vertex -3.07 1.3 2.14
vertex -3.07 0.9 2.54
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.47 1.3 2.54
vertex -3.07 1.3 2.14
vertex -3.07 1.7 2.54
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.45 -0.6 -2.83
vertex -2.85 -0.6 -2.43
vertex -2.85 -1 -2.83
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.45 -0.6 -2.83
vertex -2.85 -0.6 -2.43
vertex -2.85 -0.2 -2.83
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.45 -0.6 -2.83
vertex -2.85 -0.6 -3.23
vertex -2.85 -1 -2.83
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.45 -0.6 -2.83
vertex -2.85 -0.6 -3.23
vertex -2.85 -0.2 -2.83
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3.25 -0.6 -2.83
vertex -2.85 -0.6 -2.43
vertex -2.85 -1 -2.83
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3.25 -0.6 -2.83
vertex -2.85 -0.6 -2.43
vertex -2.85 -0.2 -2.83
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3.25 -0.6 -2.83
vertex -2.85 -0.6 -3.23
vertex -2.85 -1 -2.83
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3.25 -0.6 -2.83
vertex -2.85 -0.6 -3.23
vertex -2.85 -0.2 -2.83
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.2 1.31 -1.48
vertex -2.6 1.31 -1.08
vertex -2.6 0.91 -1.48
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.2 1.31 -1.48
vertex -2.6 1.31 -1.08
vertex -2.6 1.71 -1.48
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.2 1.31 -1.48
vertex -2.6 1.31 -1.88
vertex -2.6 0.91 -1.48
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.2 1.31 -1.48
vertex -2.6 1.31 -1.88
vertex -2.6 1.71 -1.48
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -3 1.31 -1.48
vertex -2.6 1.31 -1.08
vertex -2.6 0.91 -1.48
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -3 1.31 -1.48
vertex -2.6 1.31 -1.08
vertex -2.6 1.71 -1.48
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -3 1.31 -1.48
vertex -2.6 1.31 -1.88
vertex -2.6 0.91 -1.48
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -3 1.31 -1.48
vertex -2.6 1.31 -1.88
vertex -2.6 1.71 -1.48
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.33 2.72 -1.83
vertex -0.73 2.72 -1.43
vertex -0.73 2.32 -1.83
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.33 2.72 -1.83
vertex -0.73 2.72 -1.43
vertex -0.73 3.12 -1.83
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.33 2.72 -1.83
vertex -0.73 2.72 -2.23
vertex -0.73 2.32 -1.83
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.33 2.72 -1.83
vertex -0.73 2.72 -2.23
vertex -0.73 3.12 -1.83
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.13 2.72 -1.83
vertex -0.73 2.72 -1.43
vertex -0.73 2.32 -1.83
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.13 2.72 -1.83
vertex -0.73 2.72 -1.43
vertex -0.73 3.12 -1.83
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.13 2.72 -1.83
vertex -0.73 2.72 -2.23
vertex -0.73 2.32 -1.83
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.13 2.72 -1.83
vertex -0.73 2.72 -2.23
vertex -0.73 3.12 -1.83
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 1.6 -2.17 0.96
vertex 1.2 -2.17 1.36
vertex 1.2 -2.57 0.96
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 1.6 -2.17 0.96
vertex 1.2 -2.17 1.36
vertex 1.2 -1.77 0.96
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 1.6 -2.17 0.96
vertex 1.2 -2.17 0.56
vertex 1.2 -2.57 0.96
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 1.6 -2.17 0.96
vertex 1.2 -2.17 0.56
vertex 1.2 -1.77 0.96
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 0.8 -2.17 0.96
vertex 1.2 -2.17 1.36
vertex 1.2 -2.57 0.96
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 0.8 -2.17 0.96
vertex 1.2 -2.17 1.36
vertex 1.2 -1.77 0.96
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 0.8 -2.17 0.96
vertex 1.2 -2.17 0.56
vertex 1.2 -2.57 0.96
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 0.8 -2.17 0.96
vertex 1.2 -2.17 0.56
vertex 1.2 -1.77 0.96
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.07 -0.76 0.74
vertex -0.47 -0.76 1.14
vertex -0.47 -1.16 0.74
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.07 -0.76 0.74
vertex -0.47 -0.76 1.14
vertex -0.47 -0.36 0.74
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.07 -0.76 0.74
vertex -0.47 -0.76 0.34
vertex -0.47 -1.16 0.74
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.07 -0.76 0.74
vertex -0.47 -0.76 0.34
vertex -0.47 -0.36 0.74
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -0.87 -0.76 0.74
vertex -0.47 -0.76 1.14
vertex -0.47 -1.16 0.74
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -0.87 -0.76 0.74
vertex -0.47 -0.76 1.14
vertex -0.47 -0.36 0.74
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -0.87 -0.76 0.74
vertex -0.47 -0.76 0.34
vertex -0.47 -1.16 0.74
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -0.87 -0.76 0.74
vertex -0.47 -0.76 0.34
vertex -0.47 -0.36 0.74
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -1.73 3.06 0.01
vertex -2.13 3.06 0.41
vertex -2.13 2.66 0.01
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -1.73 3.06 0.01
vertex -2.13 3.06 0.41
vertex -2.13 3.46 0.01
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -1.73 3.06 0.01
vertex -2.13 3.06 -0.39
vertex -2.13 2.66 0.01
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -1.73 3.06 0.01
vertex -2.13 3.06 -0.39
vertex -2.13 3.46 0.01
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex -2.53 3.06 0.01
vertex -2.13 3.06 0.41
vertex -2.13 2.66 0.01
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex -2.53 3.06 0.01
vertex -2.13 3.06 0.41
vertex -2.13 3.46 0.01
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex -2.53 3.06 0.01
vertex -2.13 3.06 -0.39
vertex -2.13 2.66 0.01
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex -2.53 3.06 0.01
vertex -2.13 3.06 -0.39
vertex -2.13 3.46 0.01
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 1.71 0.54 0.92
vertex 1.31 0.54 1.32
vertex 1.31 0.14 0.92
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 1.71 0.54 0.92
vertex 1.31 0.54 1.32
vertex 1.31 0.94 0.92
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 1.71 0.54 0.92
vertex 1.31 0.54 0.52
vertex 1.31 0.14 0.92
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 1.71 0.54 0.92
vertex 1.31 0.54 0.52
vertex 1.31 0.94 0.92
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 0.91 0.54 0.92
vertex 1.31 0.54 1.32
vertex 1.31 0.14 0.92
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 0.91 0.54 0.92
vertex 1.31 0.54 1.32
vertex 1.31 0.94 0.92
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 0.91 0.54 0.92
vertex 1.31 0.54 0.52
vertex 1.31 0.14 0.92
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 0.91 0.54 0.92
vertex 1.31 0.54 0.52
vertex 1.31 0.94 0.92
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 6.6 -0 0.9
vertex 6.1 -0 1.4
vertex 6.1 -0.5 0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 6.6 -0 0.9
vertex 6.1 -0 1.4
vertex 6.1 0.5 0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 6.6 -0 0.9
vertex 6.1 -0 0.4
vertex 6.1 -0.5 0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 6.6 -0 0.9
vertex 6.1 -0 0.4
vertex 6.1 0.5 0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 5.6 -0 0.9
vertex 6.1 -0 1.4
vertex 6.1 -0.5 0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 5.6 -0 0.9
vertex 6.1 -0 1.4
vertex 6.1 0.5 0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 5.6 -0 0.9
vertex 6.1 -0 0.4
vertex 6.1 -0.5 0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 5.6 -0 0.9
vertex 6.1 -0 0.4
vertex 6.1 0.5 0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 7.5 -0 0.9
vertex 7 -0 1.4
vertex 7 -0.5 0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 7.5 -0 0.9
vertex 7 -0 1.4
vertex 7 0.5 0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 7.5 -0 0.9
vertex 7 -0 0.4
vertex 7 -0.5 0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 7.5 -0 0.9
vertex 7 -0 0.4
vertex 7 0.5 0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 6.5 -0 0.9
vertex 7 -0 1.4
vertex 7 -0.5 0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 6.5 -0 0.9
vertex 7 -0 1.4
vertex 7 0.5 0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 6.5 -0 0.9
vertex 7 -0 0.4
vertex 7 -0.5 0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 6.5 -0 0.9
vertex 7 -0 0.4
vertex 7 0.5 0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 8.4 -0 0.9
vertex 7.9 -0 1.4
vertex 7.9 -0.5 0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 8.4 -0 0.9
vertex 7.9 -0 1.4
vertex 7.9 0.5 0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 8.4 -0 0.9
vertex 7.9 -0 0.4
vertex 7.9 -0.5 0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 8.4 -0 0.9
vertex 7.9 -0 0.4
vertex 7.9 0.5 0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 7.4 -0 0.9
vertex 7.9 -0 1.4
vertex 7.9 -0.5 0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 7.4 -0 0.9
vertex 7.9 -0 1.4
vertex 7.9 0.5 0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 7.4 -0 0.9
vertex 7.9 -0 0.4
vertex 7.9 -0.5 0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 7.4 -0 0.9
vertex 7.9 -0 0.4
vertex 7.9 0.5 0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 6.6 -0 0
vertex 6.1 -0 0.5
vertex 6.1 -0.5 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 6.6 -0 0
vertex 6.1 -0 0.5
vertex 6.1 0.5 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 6.6 -0 0
vertex 6.1 -0 -0.5
vertex 6.1 -0.5 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 6.6 -0 0
vertex 6.1 -0 -0.5
vertex 6.1 0.5 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 5.6 -0 0
vertex 6.1 -0 0.5
vertex 6.1 -0.5 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 5.6 -0 0
vertex 6.1 -0 0.5
vertex 6.1 0.5 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 5.6 -0 0
vertex 6.1 -0 -0.5
vertex 6.1 -0.5 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 5.6 -0 0
vertex 6.1 -0 -0.5
vertex 6.1 0.5 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 7.5 -0 0
vertex 7 -0 0.5
vertex 7 -0.5 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 7.5 -0 0
vertex 7 -0 0.5
vertex 7 0.5 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 7.5 -0 0
vertex 7 -0 -0.5
vertex 7 -0.5 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 7.5 -0 0
vertex 7 -0 -0.5
vertex 7 0.5 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 6.5 -0 0
vertex 7 -0 0.5
vertex 7 -0.5 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 6.5 -0 0
vertex 7 -0 0.5
vertex 7 0.5 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 6.5 -0 0
vertex 7 -0 -0.5
vertex 7 -0.5 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 6.5 -0 0
vertex 7 -0 -0.5
vertex 7 0.5 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 8.4 -0 0
vertex 7.9 -0 0.5
vertex 7.9 -0.5 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 8.4 -0 0
vertex 7.9 -0 0.5
vertex 7.9 0.5 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 8.4 -0 0
vertex 7.9 -0 -0.5
vertex 7.9 -0.5 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 8.4 -0 0
vertex 7.9 -0 -0.5
vertex 7.9 0.5 0
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 7.4 -0 0
vertex 7.9 -0 0.5
vertex 7.9 -0.5 0
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 7.4 -0 0
vertex 7.9 -0 0.5
vertex 7.9 0.5 0
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 7.4 -0 0
vertex 7.9 -0 -0.5
vertex 7.9 -0.5 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 7.4 -0 0
vertex 7.9 -0 -0.5
vertex 7.9 0.5 0
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 6.6 -0 -0.9
vertex 6.1 -0 -0.4
vertex 6.1 -0.5 -0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 6.6 -0 -0.9
vertex 6.1 -0 -0.4
vertex 6.1 0.5 -0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 6.6 -0 -0.9
vertex 6.1 -0 -1.4
vertex 6.1 -0.5 -0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 6.6 -0 -0.9
vertex 6.1 -0 -1.4
vertex 6.1 0.5 -0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 5.6 -0 -0.9
vertex 6.1 -0 -0.4
vertex 6.1 -0.5 -0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 5.6 -0 -0.9
vertex 6.1 -0 -0.4
vertex 6.1 0.5 -0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 5.6 -0 -0.9
vertex 6.1 -0 -1.4
vertex 6.1 -0.5 -0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 5.6 -0 -0.9
vertex 6.1 -0 -1.4
vertex 6.1 0.5 -0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 7.5 -0 -0.9
vertex 7 -0 -0.4
vertex 7 -0.5 -0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 7.5 -0 -0.9
vertex 7 -0 -0.4
vertex 7 0.5 -0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 7.5 -0 -0.9
vertex 7 -0 -1.4
vertex 7 -0.5 -0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 7.5 -0 -0.9
vertex 7 -0 -1.4
vertex 7 0.5 -0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 6.5 -0 -0.9
vertex 7 -0 -0.4
vertex 7 -0.5 -0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 6.5 -0 -0.9
vertex 7 -0 -0.4
vertex 7 0.5 -0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 6.5 -0 -0.9
vertex 7 -0 -1.4
vertex 7 -0.5 -0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 6.5 -0 -0.9
vertex 7 -0 -1.4
vertex 7 0.5 -0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 8.4 -0 -0.9
vertex 7.9 -0 -0.4
vertex 7.9 -0.5 -0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 8.4 -0 -0.9
vertex 7.9 -0 -0.4
vertex 7.9 0.5 -0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 8.4 -0 -0.9
vertex 7.9 -0 -1.4
vertex 7.9 -0.5 -0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 8.4 -0 -0.9
vertex 7.9 -0 -1.4
vertex 7.9 0.5 -0.9
endloop
endfacet
facet normal 0.58 0.58 -0.58
outer loop
vertex 7.4 -0 -0.9
vertex 7.9 -0 -0.4
vertex 7.9 -0.5 -0.9
endloop
endfacet
facet normal -0.58 0.58 0.58
outer loop
vertex 7.4 -0 -0.9
vertex 7.9 -0 -0.4
vertex 7.9 0.5 -0.9
endloop
endfacet
facet normal -0.58 -0.58 -0.58
outer loop
vertex 7.4 -0 -0.9
vertex 7.9 -0 -1.4
vertex 7.9 -0.5 -0.9
endloop
endfacet
facet normal 0.58 -0.58 0.58
outer loop
vertex 7.4 -0 -0.9
vertex 7.9 -0 -1.4
vertex 7.9 0.5 -0.9
endloop
endfacet
facet normal -0.07 -0.87 0.5
outer loop
vertex -7.54 -0 -2.12
vertex 7.01 -0 -0.07
vertex 6.99 0.06 0.04
endloop
endfacet
facet normal -0.07 -0.87 0.5
outer loop
vertex -7.54 -0 -2.12
vertex 6.99 0.06 0.04
vertex -7.56 0.06 -2.01
endloop
endfacet
facet normal 0.14 -0 -0.99
outer loop
vertex -7.56 0.06 -2.01
vertex 6.99 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal 0.14 -0 -0.99
outer loop
vertex -7.56 0.06 -2.01
vertex 6.99 -0.06 0.04
vertex -7.56 -0.06 -2.01
endloop
endfacet
facet normal -0.07 0.87 0.5
outer loop
vertex -7.56 -0.06 -2.01
vertex 6.99 -0.06 0.04
vertex 7.01 -0 -0.07
endloop
endfacet
facet normal -0.07 0.87 0.5
outer loop
vertex -7.56 -0.06 -2.01
vertex 7.01 -0 -0.07
vertex -7.54 -0 -2.12
endloop
endfacet
facet normal -0.02 -0.87 0.5
outer loop
vertex -6.45 -0 -0.62
vertex 7 -0 -0.07
vertex 7 0.06 0.04
endloop
endfacet
facet normal -0.02 -0.87 0.5
outer loop
vertex -6.45 -0 -0.62
vertex 7 0.06 0.04
vertex -6.45 0.06 -0.51
endloop
endfacet
facet normal 0.04 -0 -1
outer loop
vertex -6.45 0.06 -0.51
vertex 7 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal 0.04 -0 -1
outer loop
vertex -6.45 0.06 -0.51
vertex 7 -0.06 0.04
vertex -6.45 -0.06 -0.51
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -6.45 -0.06 -0.51
vertex 7 -0.06 0.04
vertex 7 -0 -0.07
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -6.45 -0.06 -0.51
vertex 7 -0 -0.07
vertex -6.45 -0 -0.62
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -7.54 -0 -2.67
vertex 7.01 -0 -0.07
vertex 6.99 0.06 0.04
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -7.54 -0 -2.67
vertex 6.99 0.06 0.04
vertex -7.56 0.06 -2.56
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -7.56 0.06 -2.56
vertex 6.99 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -7.56 0.06 -2.56
vertex 6.99 -0.06 0.04
vertex -7.56 -0.06 -2.56
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7.56 -0.06 -2.56
vertex 6.99 -0.06 0.04
vertex 7.01 -0 -0.07
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7.56 -0.06 -2.56
vertex 7.01 -0 -0.07
vertex -7.54 -0 -2.67
endloop
endfacet
facet normal 0.1 -0.86 0.49
outer loop
vertex -3.5 -0.58 1.09
vertex 6.99 -0 -0.07
vertex 7 0.06 0.04
endloop
endfacet
facet normal 0.1 -0.86 0.49
outer loop
vertex -3.5 -0.58 1.09
vertex 7 0.06 0.04
vertex -3.49 -0.52 1.2
endloop
endfacet
facet normal -0.11 -0 -0.99
outer loop
vertex -3.49 -0.52 1.2
vertex 7 0.06 0.04
vertex 7.01 -0.06 0.04
endloop
endfacet
facet normal -0.11 -0 -0.99
outer loop
vertex -3.49 -0.52 1.2
vertex 7.01 -0.06 0.04
vertex -3.49 -0.65 1.2
endloop
endfacet
facet normal 0.01 0.86 0.5
outer loop
vertex -3.49 -0.65 1.2
vertex 7.01 -0.06 0.04
vertex 6.99 -0 -0.07
endloop
endfacet
facet normal 0.01 0.86 0.5
outer loop
vertex -3.49 -0.65 1.2
vertex 6.99 -0 -0.07
vertex -3.5 -0.58 1.09
endloop
endfacet
facet normal -0.11 -0.87 0.49
outer loop
vertex -7.53 -0 -3.22
vertex 7.02 -0 -0.07
vertex 6.99 0.06 0.04
endloop
endfacet
facet normal -0.11 -0.87 0.49
outer loop
vertex -7.53 -0 -3.22
vertex 6.99 0.06 0.04
vertex -7.56 0.06 -3.11
endloop
endfacet
facet normal 0.21 -0 -0.98
outer loop
vertex -7.56 0.06 -3.11
vertex 6.99 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal 0.21 -0 -0.98
outer loop
vertex -7.56 0.06 -3.11
vertex 6.99 -0.06 0.04
vertex -7.56 -0.06 -3.11
endloop
endfacet
facet normal -0.11 0.87 0.49
outer loop
vertex -7.56 -0.06 -3.11
vertex 6.99 -0.06 0.04
vertex 7.02 -0 -0.07
endloop
endfacet
facet normal -0.11 0.87 0.49
outer loop
vertex -7.56 -0.06 -3.11
vertex 7.02 -0 -0.07
vertex -7.53 -0 -3.22
endloop
endfacet
facet normal 0.11 -0.87 0.49
outer loop
vertex -6.47 -0 3.08
vertex 6.98 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal 0.11 -0.87 0.49
outer loop
vertex -6.47 -0 3.08
vertex 7.01 0.06 0.04
vertex -6.44 0.06 3.19
endloop
endfacet
facet normal -0.23 -0 -0.97
outer loop
vertex -6.44 0.06 3.19
vertex 7.01 0.06 0.04
vertex 7.01 -0.06 0.04
endloop
endfacet
facet normal -0.23 -0 -0.97
outer loop
vertex -6.44 0.06 3.19
vertex 7.01 -0.06 0.04
vertex -6.44 -0.06 3.19
endloop
endfacet
facet normal 0.11 0.87 0.49
outer loop
vertex -6.44 -0.06 3.19
vertex 7.01 -0.06 0.04
vertex 6.98 -0 -0.07
endloop
endfacet
facet normal 0.11 0.87 0.49
outer loop
vertex -6.44 -0.06 3.19
vertex 6.98 -0 -0.07
vertex -6.47 -0 3.08
endloop
endfacet
facet normal 0.2 -0.86 0.46
outer loop
vertex -1.74 -0.58 2.53
vertex 6.98 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal 0.2 -0.86 0.46
outer loop
vertex -1.74 -0.58 2.53
vertex 7.01 0.06 0.04
vertex -1.71 -0.52 2.64
endloop
endfacet
facet normal -0.29 -0 -0.96
outer loop
vertex -1.71 -0.52 2.64
vertex 7.01 0.06 0.04
vertex 7.01 -0.06 0.03
endloop
endfacet
facet normal -0.29 -0 -0.96
outer loop
vertex -1.71 -0.52 2.64
vertex 7.01 -0.06 0.03
vertex -1.71 -0.65 2.64
endloop
endfacet
facet normal 0.09 0.86 0.49
outer loop
vertex -1.71 -0.65 2.64
vertex 7.01 -0.06 0.03
vertex 6.98 -0 -0.07
endloop
endfacet
facet normal 0.09 0.86 0.49
outer loop
vertex -1.71 -0.65 2.64
vertex 6.98 -0 -0.07
vertex -1.74 -0.58 2.53
endloop
endfacet
facet normal -0.15 -0.87 0.48
outer loop
vertex -7.28 -0 -4.47
vertex 7.02 -0 -0.07
vertex 6.99 0.06 0.04
endloop
endfacet
facet normal -0.15 -0.87 0.48
outer loop
vertex -7.28 -0 -4.47
vertex 6.99 0.06 0.04
vertex -7.31 0.06 -4.36
endloop
endfacet
facet normal 0.29 -0 -0.96
outer loop
vertex -7.31 0.06 -4.36
vertex 6.99 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal 0.29 -0 -0.96
outer loop
vertex -7.31 0.06 -4.36
vertex 6.99 -0.06 0.04
vertex -7.31 -0.06 -4.36
endloop
endfacet
facet normal -0.15 0.87 0.48
outer loop
vertex -7.31 -0.06 -4.36
vertex 6.99 -0.06 0.04
vertex 7.02 -0 -0.07
endloop
endfacet
facet normal -0.15 0.87 0.48
outer loop
vertex -7.31 -0.06 -4.36
vertex 7.02 -0 -0.07
vertex -7.28 -0 -4.47
endloop
endfacet
facet normal 0.24 -0.82 0.52
outer loop
vertex -0.42 -2.55 -0.72
vertex 7.01 -0 -0.07
vertex 6.98 0.06 0.04
endloop
endfacet
facet normal 0.24 -0.82 0.52
outer loop
vertex -0.42 -2.55 -0.72
vertex 6.98 0.06 0.04
vertex -0.45 -2.49 -0.6
endloop
endfacet
facet normal 0.09 -0 -1
outer loop
vertex -0.45 -2.49 -0.6
vertex 6.98 0.06 0.04
vertex 7.02 -0.06 0.04
endloop
endfacet
facet normal 0.09 -0 -1
outer loop
vertex -0.45 -2.49 -0.6
vertex 7.02 -0.06 0.04
vertex -0.41 -2.61 -0.6
endloop
endfacet
facet normal -0.32 0.82 0.47
outer loop
vertex -0.41 -2.61 -0.6
vertex 7.02 -0.06 0.04
vertex 7.01 -0 -0.07
endloop
endfacet
facet normal -0.32 0.82 0.47
outer loop
vertex -0.41 -2.61 -0.6
vertex 7.01 -0 -0.07
vertex -0.42 -2.55 -0.72
endloop
endfacet
facet normal -0 -0.87 -0.5
outer loop
vertex 7.9 -0 0.08
vertex 7 -0 0.08
vertex 7 0.06 -0.04
endloop
endfacet
facet normal 0 -0.87 -0.5
outer loop
vertex 7.9 -0 0.08
vertex 7 0.06 -0.04
vertex 7.9 0.06 -0.04
endloop
endfacet
facet normal 0 -0 1
outer loop
vertex 7.9 0.06 -0.04
vertex 7 0.06 -0.04
vertex 7 -0.06 -0.04
endloop
endfacet
facet normal 0 -0 1
outer loop
vertex 7.9 0.06 -0.04
vertex 7 -0.06 -0.04
vertex 7.9 -0.06 -0.04
endloop
endfacet
facet normal 0 0.87 -0.5
outer loop
vertex 7.9 -0.06 -0.04
vertex 7 -0.06 -0.04
vertex 7 -0 0.08
endloop
endfacet
facet normal 0 0.87 -0.5
outer loop
vertex 7.9 -0.06 -0.04
vertex 7 -0 0.08
vertex 7.9 -0 0.08
endloop
endfacet
facet normal -0.02 -0.87 0.5
outer loop
vertex -7 -0 -0.62
vertex 7 -0 -0.07
vertex 7 0.06 0.04
endloop
endfacet
facet normal -0.02 -0.87 0.5
outer loop
vertex -7 -0 -0.62
vertex 7 0.06 0.04
vertex -7 0.06 -0.51
endloop
endfacet
facet normal 0.04 -0 -1
outer loop
vertex -7 0.06 -0.51
vertex 7 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal 0.04 -0 -1
outer loop
vertex -7 0.06 -0.51
vertex 7 -0.06 0.04
vertex -7 -0.06 -0.51
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -7 -0.06 -0.51
vertex 7 -0.06 0.04
vertex 7 -0 -0.07
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -7 -0.06 -0.51
vertex 7 -0 -0.07
vertex -7 -0 -0.62
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -6.45 -0 -0.08
vertex 7 -0 -0.08
vertex 7 0.06 0.04
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -6.45 -0 -0.08
vertex 7 0.06 0.04
vertex -6.45 0.06 0.04
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -6.45 0.06 0.04
vertex 7 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -6.45 0.06 0.04
vertex 7 -0.06 0.04
vertex -6.45 -0.06 0.04
endloop
endfacet
facet normal -0 0.87 0.5
outer loop
vertex -6.45 -0.06 0.04
vertex 7 -0.06 0.04
vertex 7 -0 -0.08
endloop
endfacet
facet normal 0 0.87 0.5
outer loop
vertex -6.45 -0.06 0.04
vertex 7 -0 -0.08
vertex -6.45 -0 -0.08
endloop
endfacet
facet normal -0.39 -0.82 0.42
outer loop
vertex -0.71 2.72 -1.91
vertex 7.02 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal -0.39 -0.82 0.42
outer loop
vertex -0.71 2.72 -1.91
vertex 7.01 0.06 0.04
vertex -0.72 2.78 -1.79
endloop
endfacet
facet normal 0.23 -0 -0.97
outer loop
vertex -0.72 2.78 -1.79
vertex 7.01 0.06 0.04
vertex 6.97 -0.06 0.03
endloop
endfacet
facet normal 0.23 -0 -0.97
outer loop
vertex -0.72 2.78 -1.79
vertex 6.97 -0.06 0.03
vertex -0.76 2.66 -1.8
endloop
endfacet
facet normal 0.16 0.82 0.55
outer loop
vertex -0.76 2.66 -1.8
vertex 6.97 -0.06 0.03
vertex 7.02 -0 -0.07
endloop
endfacet
facet normal 0.16 0.82 0.55
outer loop
vertex -0.76 2.66 -1.8
vertex 7.02 -0 -0.07
vertex -0.71 2.72 -1.91
endloop
endfacet
facet normal -0.27 -0.82 0.5
outer loop
vertex -2.13 3.06 -0.06
vertex 7 -0 -0.07
vertex 7.02 0.06 0.04
endloop
endfacet
facet normal -0.27 -0.82 0.5
outer loop
vertex -2.13 3.06 -0.06
vertex 7.02 0.06 0.04
vertex -2.1 3.12 0.05
endloop
endfacet
facet normal -0 -0 -1
outer loop
vertex -2.1 3.12 0.05
vertex 7.02 0.06 0.04
vertex 6.98 -0.06 0.04
endloop
endfacet
facet normal -0 -0 -1
outer loop
vertex -2.1 3.12 0.05
vertex 6.98 -0.06 0.04
vertex -2.15 3 0.05
endloop
endfacet
facet normal 0.28 0.82 0.5
outer loop
vertex -2.15 3 0.05
vertex 6.98 -0.06 0.04
vertex 7 -0 -0.07
endloop
endfacet
facet normal 0.28 0.82 0.5
outer loop
vertex -2.15 3 0.05
vertex 7 -0 -0.07
vertex -2.13 3.06 -0.06
endloop
endfacet
facet normal -0.11 -0.86 0.5
outer loop
vertex -3.91 1.61 0.31
vertex 7 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal -0.11 -0.86 0.5
outer loop
vertex -3.91 1.61 0.31
vertex 7.01 0.06 0.04
vertex -3.9 1.67 0.42
endloop
endfacet
facet normal -0.03 -0 -1
outer loop
vertex -3.9 1.67 0.42
vertex 7.01 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal -0.03 -0 -1
outer loop
vertex -3.9 1.67 0.42
vertex 6.99 -0.06 0.04
vertex -3.92 1.54 0.42
endloop
endfacet
facet normal 0.14 0.86 0.5
outer loop
vertex -3.92 1.54 0.42
vertex 6.99 -0.06 0.04
vertex 7 -0 -0.07
endloop
endfacet
facet normal 0.14 0.86 0.5
outer loop
vertex -3.92 1.54 0.42
vertex 7 -0 -0.07
vertex -3.91 1.61 0.31
endloop
endfacet
facet normal -0.05 -0.86 0.51
outer loop
vertex -1.79 0.93 0.62
vertex 6.99 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal -0.05 -0.86 0.51
outer loop
vertex -1.79 0.93 0.62
vertex 7.01 0.06 0.04
vertex -1.78 0.99 0.73
endloop
endfacet
facet normal -0.08 -0 -1
outer loop
vertex -1.78 0.99 0.73
vertex 7.01 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal -0.08 -0 -1
outer loop
vertex -1.78 0.99 0.73
vertex 7 -0.06 0.04
vertex -1.79 0.86 0.73
endloop
endfacet
facet normal 0.13 0.86 0.49
outer loop
vertex -1.79 0.86 0.73
vertex 7 -0.06 0.04
vertex 6.99 -0 -0.07
endloop
endfacet
facet normal 0.13 0.86 0.49
outer loop
vertex -1.79 0.86 0.73
vertex 6.99 -0 -0.07
vertex -1.79 0.93 0.62
endloop
endfacet
facet normal 0 -0.86 0.51
outer loop
vertex 1.3 0.54 0.85
vertex 6.99 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal 0 -0.86 0.51
outer loop
vertex 1.3 0.54 0.85
vertex 7.01 0.06 0.04
vertex 1.32 0.6 0.96
endloop
endfacet
facet normal -0.16 -0 -0.99
outer loop
vertex 1.32 0.6 0.96
vertex 7.01 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal -0.16 -0 -0.99
outer loop
vertex 1.32 0.6 0.96
vertex 7 -0.06 0.04
vertex 1.31 0.47 0.96
endloop
endfacet
facet normal 0.16 0.86 0.48
outer loop
vertex 1.31 0.47 0.96
vertex 7 -0.06 0.04
vertex 6.99 -0 -0.07
endloop
endfacet
facet normal 0.16 0.86 0.48
outer loop
vertex 1.31 0.47 0.96
vertex 6.99 -0 -0.07
vertex 1.3 0.54 0.85
endloop
endfacet
facet normal 0.11 -0.87 0.49
outer loop
vertex -7.57 -0 3.08
vertex 6.98 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal 0.11 -0.87 0.49
outer loop
vertex -7.57 -0 3.08
vertex 7.01 0.06 0.04
vertex -7.54 0.06 3.19
endloop
endfacet
facet normal -0.21 -0 -0.98
outer loop
vertex -7.54 0.06 3.19
vertex 7.01 0.06 0.04
vertex 7.01 -0.06 0.04
endloop
endfacet
facet normal -0.21 -0 -0.98
outer loop
vertex -7.54 0.06 3.19
vertex 7.01 -0.06 0.04
vertex -7.54 -0.06 3.19
endloop
endfacet
facet normal 0.11 0.87 0.49
outer loop
vertex -7.54 -0.06 3.19
vertex 7.01 -0.06 0.04
vertex 6.98 -0 -0.07
endloop
endfacet
facet normal 0.11 0.87 0.49
outer loop
vertex -7.54 -0.06 3.19
vertex 6.98 -0 -0.07
vertex -7.57 -0 3.08
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -6.99 -0 -2.67
vertex 7.01 -0 -0.07
vertex 6.99 0.06 0.04
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -6.99 -0 -2.67
vertex 6.99 0.06 0.04
vertex -7.01 0.06 -2.56
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -7.01 0.06 -2.56
vertex 6.99 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -7.01 0.06 -2.56
vertex 6.99 -0.06 0.04
vertex -7.01 -0.06 -2.56
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7.01 -0.06 -2.56
vertex 6.99 -0.06 0.04
vertex 7.01 -0 -0.07
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7.01 -0.06 -2.56
vertex 7.01 -0 -0.07
vertex -6.99 -0 -2.67
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -7.55 -0 -0.08
vertex 7 -0 -0.08
vertex 7 0.06 0.04
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -7.55 -0 -0.08
vertex 7 0.06 0.04
vertex -7.55 0.06 0.04
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -7.55 0.06 0.04
vertex 7 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -7.55 0.06 0.04
vertex 7 -0.06 0.04
vertex -7.55 -0.06 0.04
endloop
endfacet
facet normal -0 0.87 0.5
outer loop
vertex -7.55 -0.06 0.04
vertex 7 -0.06 0.04
vertex 7 -0 -0.08
endloop
endfacet
facet normal 0 0.87 0.5
outer loop
vertex -7.55 -0.06 0.04
vertex 7 -0 -0.08
vertex -7.55 -0 -0.08
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -7 -0 -0.08
vertex 7 -0 -0.08
vertex 7 0.06 0.04
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -7 -0 -0.08
vertex 7 0.06 0.04
vertex -7 0.06 0.04
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -7 0.06 0.04
vertex 7 0.06 0.04
vertex 7 -0.06 0.04
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -7 0.06 0.04
vertex 7 -0.06 0.04
vertex -7 -0.06 0.04
endloop
endfacet
facet normal -0 0.87 0.5
outer loop
vertex -7 -0.06 0.04
vertex 7 -0.06 0.04
vertex 7 -0 -0.08
endloop
endfacet
facet normal 0 0.87 0.5
outer loop
vertex -7 -0.06 0.04
vertex 7 -0 -0.08
vertex -7 -0 -0.08
endloop
endfacet
facet normal -0.32 -0.84 0.44
outer loop
vertex 1.58 1.43 -1.3
vertex 7.02 -0 -0.07
vertex 7.01 0.06 0.04
endloop
endfacet
facet normal -0.32 -0.84 0.44
outer loop
vertex 1.58 1.43 -1.3
vertex 7.01 0.06 0.04
vertex 1.58 1.49 -1.18
endloop
endfacet
facet normal 0.22 -0 -0.98
outer loop
vertex 1.58 1.49 -1.18
vertex 7.01 0.06 0.04
vertex 6.98 -0.06 0.03
endloop
endfacet
facet normal 0.22 -0 -0.98
outer loop
vertex 1.58 1.49 -1.18
vertex 6.98 -0.06 0.03
vertex 1.54 1.37 -1.19
endloop
endfacet
facet normal 0.1 0.84 0.54
outer loop
vertex 1.54 1.37 -1.19
vertex 6.98 -0.06 0.03
vertex 7.02 -0 -0.07
endloop
endfacet
facet normal 0.1 0.84 0.54
outer loop
vertex 1.54 1.37 -1.19
vertex 7.02 -0 -0.07
vertex 1.58 1.43 -1.3
endloop
endfacet
facet normal -0.11 -0.87 0.49
outer loop
vertex -6.98 -0 -3.22
vertex 7.02 -0 -0.07
vertex 6.99 0.06 0.04
endloop
endfacet
facet normal -0.11 -0.87 0.49
outer loop
vertex -6.98 -0 -3.22
vertex 6.99 0.06 0.04
vertex -7.01 0.06 -3.11
endloop
endfacet
facet normal 0.22 -0 -0.98
outer loop
vertex -7.01 0.06 -3.11
vertex 6.99 0.06 0.04
vertex 6.99 -0.06 0.04
endloop
endfacet
facet normal 0.22 -0 -0.98
outer loop
vertex -7.01 0.06 -3.11
vertex 6.99 -0.06 0.04
vertex -7.01 -0.06 -3.11
endloop
endfacet
facet normal -0.11 0.87 0.49
outer loop
vertex -7.01 -0.06 -3.11
vertex 6.99 -0.06 0.04
vertex 7.02 -0 -0.07
endloop
endfacet
facet normal -0.11 0.87 0.49
outer loop
vertex -7.01 -0.06 -3.11
vertex 7.02 -0 -0.07
vertex -6.98 -0 -3.22
endloop
endfacet
facet normal 0.14 -0.86 0.49
outer loop
vertex -0.47 -0.76 0.67
vertex 6.99 -0 -0.07
vertex 7 0.06 0.03
endloop
endfacet
facet normal 0.14 -0.86 0.49
outer loop
vertex -0.47 -0.76 0.67
vertex 7 0.06 0.03
vertex -0.47 -0.7 0.77
endloop
endfacet
facet normal -0.1 -0 -1
outer loop
vertex -0.47 -0.7 0.77
vertex 7 0.06 0.03
vertex 7.01 -0.06 0.03
endloop
endfacet
facet normal -0.1 -0 -1
outer loop
vertex -0.47 -0.7 0.77
vertex 7.01 -0.06 0.03
vertex -0.46 -0.82 0.77
endloop
endfacet
facet normal -0.04 0.86 0.51
outer loop
vertex -0.46 -0.82 0.77
vertex 7.01 -0.06 0.03
vertex 6.99 -0 -0.07
endloop
endfacet
facet normal -0.04 0.86 0.51
outer loop
vertex -0.46 -0.82 0.77
vertex 6.99 -0 -0.07
vertex -0.47 -0.76 0.67
endloop
endfacet
facet normal -0.15 -0.87 0.48
outer loop
vertex -6.68 -0 -4.46
vertex 7.02 -0 -0.06
vertex 6.99 0.05 0.03
endloop
endfacet
facet normal -0.15 -0.87 0.48
outer loop
vertex -6.68 -0 -4.46
vertex 6.99 0.05 0.03
vertex -6.71 0.05 -4.37
endloop
endfacet
facet normal 0.31 -0 -0.95
outer loop
vertex -6.71 0.05 -4.37
vertex 6.99 0.05 0.03
vertex 6.99 -0.05 0.03
endloop
endfacet
facet normal 0.31 -0 -0.95
outer loop
vertex -6.71 0.05 -4.37
vertex 6.99 -0.05 0.03
vertex -6.71 -0.05 -4.37
endloop
endfacet
facet normal -0.15 0.87 0.48
outer loop
vertex -6.71 -0.05 -4.37
vertex 6.99 -0.05 0.03
vertex 7.02 -0 -0.06
endloop
endfacet
facet normal -0.15 0.87 0.48
outer loop
vertex -6.71 -0.05 -4.37
vertex 7.02 -0 -0.06
vertex -6.68 -0 -4.46
endloop
endfacet
facet normal -0.04 -0.86 0.51
outer loop
vertex -0.6 -1.19 -2.57
vertex 7.02 -0 -0.05
vertex 6.99 0.04 0.02
endloop
endfacet
facet normal -0.04 -0.86 0.51
outer loop
vertex -0.6 -1.19 -2.57
vertex 6.99 0.04 0.02
vertex -0.63 -1.14 -2.51
endloop
endfacet
facet normal 0.32 -0 -0.95
outer loop
vertex -0.63 -1.14 -2.51
vertex 6.99 0.04 0.02
vertex 7 -0.04 0.03
endloop
endfacet
facet normal 0.32 -0 -0.95
outer loop
vertex -0.63 -1.14 -2.51
vertex 7 -0.04 0.03
vertex -0.61 -1.23 -2.5
endloop
endfacet
facet normal -0.28 0.86 0.43
outer loop
vertex -0.61 -1.23 -2.5
vertex 7 -0.04 0.03
vertex 7.02 -0 -0.05
endloop
endfacet
facet normal -0.28 0.86 0.43
outer loop
vertex -0.61 -1.23 -2.5
vertex 7.02 -0 -0.05
vertex -0.6 -1.19 -2.57
endloop
endfacet
facet normal -0.02 -0.85 0.52
outer loop
vertex -7.01 -0 0.51
vertex -2.11 -0.89 -0.73
vertex -2.09 -0.86 -0.67
endloop
endfacet
facet normal -0.02 -0.85 0.52
outer loop
vertex -7.01 -0 0.51
vertex -2.09 -0.86 -0.67
vertex -6.99 0.03 0.57
endloop
endfacet
facet normal -0.24 -0 -0.97
outer loop
vertex -6.99 0.03 0.57
vertex -2.09 -0.86 -0.67
vertex -2.1 -0.92 -0.67
endloop
endfacet
facet normal -0.24 -0 -0.97
outer loop
vertex -6.99 0.03 0.57
vertex -2.1 -0.92 -0.67
vertex -7 -0.03 0.57
endloop
endfacet
facet normal 0.27 0.85 0.45
outer loop
vertex -7 -0.03 0.57
vertex -2.1 -0.92 -0.67
vertex -2.11 -0.89 -0.73
endloop
endfacet
facet normal 0.27 0.85 0.45
outer loop
vertex -7 -0.03 0.57
vertex -2.11 -0.89 -0.73
vertex -7.01 -0 0.51
endloop
endfacet
facet normal -0.42 -0.82 0.38
outer loop
vertex -0.72 2.72 -1.87
vertex 7.01 -0 0.86
vertex 7 0.03 0.92
endloop
endfacet
facet normal -0.42 -0.82 0.38
outer loop
vertex -0.72 2.72 -1.87
vertex 7 0.03 0.92
vertex -0.72 2.75 -1.81
endloop
endfacet
facet normal 0.33 -0 -0.94
outer loop
vertex -0.72 2.75 -1.81
vertex 7 0.03 0.92
vertex 6.98 -0.03 0.91
endloop
endfacet
facet normal 0.33 -0 -0.94
outer loop
vertex -0.72 2.75 -1.81
vertex 6.98 -0.03 0.91
vertex -0.74 2.69 -1.82
endloop
endfacet
facet normal 0.09 0.82 0.56
outer loop
vertex -0.74 2.69 -1.82
vertex 6.98 -0.03 0.91
vertex 7.01 -0 0.86
endloop
endfacet
facet normal 0.09 0.82 0.56
outer loop
vertex -0.74 2.69 -1.82
vertex 7.01 -0 0.86
vertex -0.72 2.72 -1.87
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -6.45 -0 -0.59
vertex 7 -0 -0.94
vertex 7 0.03 -0.88
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -6.45 -0 -0.59
vertex 7 0.03 -0.88
vertex -6.45 0.03 -0.53
endloop
endfacet
facet normal -0.03 -0 -1
outer loop
vertex -6.45 0.03 -0.53
vertex 7 0.03 -0.88
vertex 7 -0.03 -0.88
endloop
endfacet
facet normal -0.03 -0 -1
outer loop
vertex -6.45 0.03 -0.53
vertex 7 -0.03 -0.88
vertex -6.45 -0.03 -0.53
endloop
endfacet
facet normal 0.01 0.87 0.5
outer loop
vertex -6.45 -0.03 -0.53
vertex 7 -0.03 -0.88
vertex 7 -0 -0.94
endloop
endfacet
facet normal 0.01 0.87 0.5
outer loop
vertex -6.45 -0.03 -0.53
vertex 7 -0 -0.94
vertex -6.45 -0 -0.59
endloop
endfacet
facet normal 0.15 -0.87 0.48
outer loop
vertex -6.46 -0 3.12
vertex 6.09 -0 -0.93
vertex 6.11 0.03 -0.88
endloop
endfacet
facet normal 0.15 -0.87 0.48
outer loop
vertex -6.46 -0 3.12
vertex 6.11 0.03 -0.88
vertex -6.44 0.03 3.17
endloop
endfacet
facet normal -0.31 -0 -0.95
outer loop
vertex -6.44 0.03 3.17
vertex 6.11 0.03 -0.88
vertex 6.11 -0.03 -0.88
endloop
endfacet
facet normal -0.31 -0 -0.95
outer loop
vertex -6.44 0.03 3.17
vertex 6.11 -0.03 -0.88
vertex -6.44 -0.03 3.17
endloop
endfacet
facet normal 0.15 0.87 0.48
outer loop
vertex -6.44 -0.03 3.17
vertex 6.11 -0.03 -0.88
vertex 6.09 -0 -0.93
endloop
endfacet
facet normal 0.15 0.87 0.48
outer loop
vertex -6.44 -0.03 3.17
vertex 6.09 -0 -0.93
vertex -6.46 -0 3.12
endloop
endfacet
facet normal -0.02 -0.85 0.53
outer loop
vertex -6.46 -0 0.52
vertex -2.11 -0.89 -0.72
vertex -2.09 -0.86 -0.67
endloop
endfacet
facet normal -0.02 -0.85 0.53
outer loop
vertex -6.46 -0 0.52
vertex -2.09 -0.86 -0.67
vertex -6.44 0.03 0.57
endloop
endfacet
facet normal -0.27 -0 -0.96
outer loop
vertex -6.44 0.03 0.57
vertex -2.09 -0.86 -0.67
vertex -2.1 -0.92 -0.67
endloop
endfacet
facet normal -0.27 -0 -0.96
outer loop
vertex -6.44 0.03 0.57
vertex -2.1 -0.92 -0.67
vertex -6.45 -0.03 0.57
endloop
endfacet
facet normal 0.3 0.85 0.44
outer loop
vertex -6.45 -0.03 0.57
vertex -2.1 -0.92 -0.67
vertex -2.11 -0.89 -0.72
endloop
endfacet
facet normal 0.3 0.85 0.44
outer loop
vertex -6.45 -0.03 0.57
vertex -2.11 -0.89 -0.72
vertex -6.46 -0 0.52
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.99 -0 -2.08
vertex 7.01 -0 0.87
vertex 7 0.03 0.92
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.99 -0 -2.08
vertex 7 0.03 0.92
vertex -7 0.03 -2.03
endloop
endfacet
facet normal 0.21 -0 -0.98
outer loop
vertex -7 0.03 -2.03
vertex 7 0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal 0.21 -0 -0.98
outer loop
vertex -7 0.03 -2.03
vertex 7 -0.03 0.92
vertex -7 -0.03 -2.03
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -7 -0.03 -2.03
vertex 7 -0.03 0.92
vertex 7.01 -0 0.87
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -7 -0.03 -2.03
vertex 7.01 -0 0.87
vertex -6.99 -0 -2.08
endloop
endfacet
facet normal 0.45 -0.85 -0.26
outer loop
vertex 1.18 -2.17 0.99
vertex -0.45 -2.55 -0.62
vertex -0.42 -2.52 -0.66
endloop
endfacet
facet normal 0.45 -0.85 -0.26
outer loop
vertex 1.18 -2.17 0.99
vertex -0.42 -2.52 -0.66
vertex 1.21 -2.14 0.95
endloop
endfacet
facet normal -0.7 -0 0.71
outer loop
vertex 1.21 -2.14 0.95
vertex -0.42 -2.52 -0.66
vertex -0.41 -2.58 -0.65
endloop
endfacet
facet normal -0.7 -0 0.71
outer loop
vertex 1.21 -2.14 0.95
vertex -0.41 -2.58 -0.65
vertex 1.22 -2.2 0.95
endloop
endfacet
facet normal 0.25 0.85 -0.46
outer loop
vertex 1.22 -2.2 0.95
vertex -0.41 -2.58 -0.65
vertex -0.45 -2.55 -0.62
endloop
endfacet
facet normal 0.25 0.85 -0.46
outer loop
vertex 1.22 -2.2 0.95
vertex -0.45 -2.55 -0.62
vertex 1.18 -2.17 0.99
endloop
endfacet
facet normal 0.13 -0.86 0.5
outer loop
vertex -7.02 -0 2.02
vertex -2.11 -0.89 -0.72
vertex -2.08 -0.86 -0.67
endloop
endfacet
facet normal 0.13 -0.86 0.5
outer loop
vertex -7.02 -0 2.02
vertex -2.08 -0.86 -0.67
vertex -6.99 0.03 2.06
endloop
endfacet
facet normal -0.49 -0 -0.87
outer loop
vertex -6.99 0.03 2.06
vertex -2.08 -0.86 -0.67
vertex -2.09 -0.92 -0.67
endloop
endfacet
facet normal -0.49 -0 -0.87
outer loop
vertex -6.99 0.03 2.06
vertex -2.09 -0.92 -0.67
vertex -7 -0.03 2.07
endloop
endfacet
facet normal 0.36 0.86 0.37
outer loop
vertex -7 -0.03 2.07
vertex -2.09 -0.92 -0.67
vertex -2.11 -0.89 -0.72
endloop
endfacet
facet normal 0.36 0.86 0.37
outer loop
vertex -7 -0.03 2.07
vertex -2.11 -0.89 -0.72
vertex -7.02 -0 2.02
endloop
endfacet
facet normal 0.8 -0.18 0.57
outer loop
vertex -0.02 2.33 0.35
vertex -0.46 -2.55 -0.63
vertex -0.42 -2.55 -0.67
endloop
endfacet
facet normal 0.8 -0.18 0.57
outer loop
vertex -0.02 2.33 0.35
vertex -0.42 -2.55 -0.67
vertex 0.02 2.33 0.3
endloop
endfacet
facet normal -0.91 -0 0.41
outer loop
vertex 0.02 2.33 0.3
vertex -0.42 -2.55 -0.67
vertex -0.4 -2.56 -0.62
endloop
endfacet
facet normal -0.91 -0 0.41
outer loop
vertex 0.02 2.33 0.3
vertex -0.4 -2.56 -0.62
vertex 0.04 2.32 0.35
endloop
endfacet
facet normal 0.11 0.18 -0.98
outer loop
vertex 0.04 2.32 0.35
vertex -0.4 -2.56 -0.62
vertex -0.46 -2.55 -0.63
endloop
endfacet
facet normal 0.11 0.18 -0.98
outer loop
vertex 0.04 2.32 0.35
vertex -0.46 -2.55 -0.63
vertex -0.02 2.33 0.35
endloop
endfacet
facet normal -0.25 -0.86 0.45
outer loop
vertex -7.54 -0 -0.03
vertex -3.48 -0.58 1.13
vertex -3.49 -0.55 1.18
endloop
endfacet
facet normal -0.25 -0.86 0.45
outer loop
vertex -7.54 -0 -0.03
vertex -3.49 -0.55 1.18
vertex -7.55 0.03 0.02
endloop
endfacet
facet normal 0.28 -0 -0.96
outer loop
vertex -7.55 0.03 0.02
vertex -3.49 -0.55 1.18
vertex -3.5 -0.61 1.18
endloop
endfacet
facet normal 0.28 -0 -0.96
outer loop
vertex -7.55 0.03 0.02
vertex -3.5 -0.61 1.18
vertex -7.56 -0.03 0.02
endloop
endfacet
facet normal -0.02 0.86 0.51
outer loop
vertex -7.56 -0.03 0.02
vertex -3.5 -0.61 1.18
vertex -3.48 -0.58 1.13
endloop
endfacet
facet normal -0.02 0.86 0.51
outer loop
vertex -7.56 -0.03 0.02
vertex -3.48 -0.58 1.13
vertex -7.54 -0 -0.03
endloop
endfacet
facet normal -0.28 -0.85 0.44
outer loop
vertex -6.99 -0 -2.08
vertex -2.09 -0.89 -0.72
vertex -2.1 -0.86 -0.67
endloop
endfacet
facet normal -0.28 -0.85 0.44
outer loop
vertex -6.99 -0 -2.08
vertex -2.1 -0.86 -0.67
vertex -7 0.03 -2.03
endloop
endfacet
facet normal 0.27 -0 -0.96
outer loop
vertex -7 0.03 -2.03
vertex -2.1 -0.86 -0.67
vertex -2.11 -0.92 -0.67
endloop
endfacet
facet normal 0.27 -0 -0.96
outer loop
vertex -7 0.03 -2.03
vertex -2.11 -0.92 -0.67
vertex -7.01 -0.03 -2.03
endloop
endfacet
facet normal 0.01 0.85 0.52
outer loop
vertex -7.01 -0.03 -2.03
vertex -2.11 -0.92 -0.67
vertex -2.09 -0.89 -0.72
endloop
endfacet
facet normal 0.01 0.85 0.52
outer loop
vertex -7.01 -0.03 -2.03
vertex -2.09 -0.89 -0.72
vertex -6.99 -0 -2.08
endloop
endfacet
facet normal 0.07 -0.85 0.51
outer loop
vertex -0.44 1.33 2.35
vertex 6.99 -0 -0.93
vertex 7.01 0.03 -0.89
endloop
endfacet
facet normal 0.07 -0.85 0.51
outer loop
vertex -0.44 1.33 2.35
vertex 7.01 0.03 -0.89
vertex -0.41 1.36 2.4
endloop
endfacet
facet normal -0.4 -0 -0.91
outer loop
vertex -0.41 1.36 2.4
vertex 7.01 0.03 -0.89
vertex 7 -0.03 -0.88
endloop
endfacet
facet normal -0.4 -0 -0.91
outer loop
vertex -0.41 1.36 2.4
vertex 7 -0.03 -0.88
vertex -0.42 1.3 2.4
endloop
endfacet
facet normal 0.33 0.85 0.4
outer loop
vertex -0.42 1.3 2.4
vertex 7 -0.03 -0.88
vertex 6.99 -0 -0.93
endloop
endfacet
facet normal 0.33 0.85 0.4
outer loop
vertex -0.42 1.3 2.4
vertex 6.99 -0 -0.93
vertex -0.44 1.33 2.35
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -4.24 -0.21 -1.05
vertex 6.1 -0 -0.93
vertex 6.1 0.03 -0.88
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -4.24 -0.21 -1.05
vertex 6.1 0.03 -0.88
vertex -4.24 -0.18 -1
endloop
endfacet
facet normal 0.01 -0 -1
outer loop
vertex -4.24 -0.18 -1
vertex 6.1 0.03 -0.88
vertex 6.1 -0.03 -0.88
endloop
endfacet
facet normal 0.01 -0 -1
outer loop
vertex -4.24 -0.18 -1
vertex 6.1 -0.03 -0.88
vertex -4.24 -0.24 -1
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -4.24 -0.24 -1
vertex 6.1 -0.03 -0.88
vertex 6.1 -0 -0.93
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -4.24 -0.24 -1
vertex 6.1 -0 -0.93
vertex -4.24 -0.21 -1.05
endloop
endfacet
facet normal 0.08 -0.86 0.51
outer loop
vertex -7.55 -0 -0.03
vertex -1.78 0.93 0.66
vertex -1.79 0.95 0.71
endloop
endfacet
facet normal 0.08 -0.86 0.51
outer loop
vertex -7.55 -0 -0.03
vertex -1.79 0.95 0.71
vertex -7.56 0.03 0.02
endloop
endfacet
facet normal 0.12 -0 -0.99
outer loop
vertex -7.56 0.03 0.02
vertex -1.79 0.95 0.71
vertex -1.78 0.9 0.71
endloop
endfacet
facet normal 0.12 -0 -0.99
outer loop
vertex -7.56 0.03 0.02
vertex -1.78 0.9 0.71
vertex -7.55 -0.03 0.02
endloop
endfacet
facet normal -0.2 0.86 0.48
outer loop
vertex -7.55 -0.03 0.02
vertex -1.78 0.9 0.71
vertex -1.78 0.93 0.66
endloop
endfacet
facet normal -0.2 0.86 0.48
outer loop
vertex -7.55 -0.03 0.02
vertex -1.78 0.93 0.66
vertex -7.55 -0 -0.03
endloop
endfacet
facet normal -0.85 -0.34 -0.41
outer loop
vertex -2.57 1.31 -1.5
vertex -2.07 -0.89 -0.71
vertex -2.1 -0.88 -0.66
endloop
endfacet
facet normal -0.85 -0.34 -0.41
outer loop
vertex -2.57 1.31 -1.5
vertex -2.1 -0.88 -0.66
vertex -2.6 1.32 -1.44
endloop
endfacet
facet normal 0.84 -0 -0.53
outer loop
vertex -2.6 1.32 -1.44
vertex -2.1 -0.88 -0.66
vertex -2.13 -0.9 -0.7
endloop
endfacet
facet normal 0.84 -0 -0.53
outer loop
vertex -2.6 1.32 -1.44
vertex -2.13 -0.9 -0.7
vertex -2.63 1.3 -1.49
endloop
endfacet
facet normal 0 0.34 0.94
outer loop
vertex -2.63 1.3 -1.49
vertex -2.13 -0.9 -0.7
vertex -2.07 -0.89 -0.71
endloop
endfacet
facet normal 0 0.34 0.94
outer loop
vertex -2.63 1.3 -1.49
vertex -2.07 -0.89 -0.71
vertex -2.57 1.31 -1.5
endloop
endfacet
facet normal 0.17 -0.86 -0.48
outer loop
vertex 6.99 -0 0.93
vertex -2.1 -0.89 -0.65
vertex -2.1 -0.86 -0.7
endloop
endfacet
facet normal 0.17 -0.86 -0.48
outer loop
vertex 6.99 -0 0.93
vertex -2.1 -0.86 -0.7
vertex 7 0.03 0.88
endloop
endfacet
facet normal -0.17 -0 0.99
outer loop
vertex 7 0.03 0.88
vertex -2.1 -0.86 -0.7
vertex -2.09 -0.92 -0.7
endloop
endfacet
facet normal -0.17 -0 0.99
outer loop
vertex 7 0.03 0.88
vertex -2.09 -0.92 -0.7
vertex 7.01 -0.03 0.88
endloop
endfacet
facet normal 0 0.86 -0.51
outer loop
vertex 7.01 -0.03 0.88
vertex -2.09 -0.92 -0.7
vertex -2.1 -0.89 -0.65
endloop
endfacet
facet normal 0 0.86 -0.51
outer loop
vertex 7.01 -0.03 0.88
vertex -2.1 -0.89 -0.65
vertex 6.99 -0 0.93
endloop
endfacet
facet normal 0.08 -0.87 0.49
outer loop
vertex -6.46 -0 2.02
vertex 6.99 -0 -0.03
vertex 7 0.03 0.02
endloop
endfacet
facet normal 0.08 -0.87 0.49
outer loop
vertex -6.46 -0 2.02
vertex 7 0.03 0.02
vertex -6.45 0.03 2.07
endloop
endfacet
facet normal -0.15 -0 -0.99
outer loop
vertex -6.45 0.03 2.07
vertex 7 0.03 0.02
vertex 7 -0.03 0.02
endloop
endfacet
facet normal -0.15 -0 -0.99
outer loop
vertex -6.45 0.03 2.07
vertex 7 -0.03 0.02
vertex -6.45 -0.03 2.07
endloop
endfacet
facet normal 0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 2.07
vertex 7 -0.03 0.02
vertex 6.99 -0 -0.03
endloop
endfacet
facet normal 0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 2.07
vertex 6.99 -0 -0.03
vertex -6.46 -0 2.02
endloop
endfacet
facet normal 0.04 -0.85 0.52
outer loop
vertex -6.69 -0 -4.43
vertex -1.18 0.93 -3.28
vertex -1.19 0.96 -3.24
endloop
endfacet
facet normal 0.04 -0.85 0.52
outer loop
vertex -6.69 -0 -4.43
vertex -1.19 0.96 -3.24
vertex -6.71 0.03 -4.38
endloop
endfacet
facet normal 0.2 -0 -0.98
outer loop
vertex -6.71 0.03 -4.38
vertex -1.19 0.96 -3.24
vertex -1.18 0.9 -3.23
endloop
endfacet
facet normal 0.2 -0 -0.98
outer loop
vertex -6.71 0.03 -4.38
vertex -1.18 0.9 -3.23
vertex -6.7 -0.03 -4.38
endloop
endfacet
facet normal -0.24 0.85 0.46
outer loop
vertex -6.7 -0.03 -4.38
vertex -1.18 0.9 -3.23
vertex -1.18 0.93 -3.28
endloop
endfacet
facet normal -0.24 0.85 0.46
outer loop
vertex -6.7 -0.03 -4.38
vertex -1.18 0.93 -3.28
vertex -6.69 -0 -4.43
endloop
endfacet
facet normal -0.41 -0.82 0.41
outer loop
vertex -7.54 -0 -2.63
vertex -0.42 -2.55 -0.67
vertex -0.42 -2.53 -0.62
endloop
endfacet
facet normal -0.41 -0.82 0.41
outer loop
vertex -7.54 -0 -2.63
vertex -0.42 -2.53 -0.62
vertex -7.55 0.03 -2.58
endloop
endfacet
facet normal 0.27 -0 -0.96
outer loop
vertex -7.55 0.03 -2.58
vertex -0.42 -2.53 -0.62
vertex -0.44 -2.58 -0.63
endloop
endfacet
facet normal 0.27 -0 -0.96
outer loop
vertex -7.55 0.03 -2.58
vertex -0.44 -2.58 -0.63
vertex -7.56 -0.03 -2.59
endloop
endfacet
facet normal 0.14 0.82 0.56
outer loop
vertex -7.56 -0.03 -2.59
vertex -0.44 -2.58 -0.63
vertex -0.42 -2.55 -0.67
endloop
endfacet
facet normal 0.14 0.82 0.56
outer loop
vertex -7.56 -0.03 -2.59
vertex -0.42 -2.55 -0.67
vertex -7.54 -0 -2.63
endloop
endfacet
facet normal -0.33 -0.86 0.39
outer loop
vertex -1.17 0.93 -3.28
vertex 6.12 -0 0.87
vertex 6.09 0.03 0.92
endloop
endfacet
facet normal -0.33 -0.86 0.39
outer loop
vertex -1.17 0.93 -3.28
vertex 6.09 0.03 0.92
vertex -1.19 0.96 -3.24
endloop
endfacet
facet normal 0.5 -0 -0.87
outer loop
vertex -1.19 0.96 -3.24
vertex 6.09 0.03 0.92
vertex 6.09 -0.03 0.91
endloop
endfacet
facet normal 0.5 -0 -0.87
outer loop
vertex -1.19 0.96 -3.24
vertex 6.09 -0.03 0.91
vertex -1.2 0.9 -3.24
endloop
endfacet
facet normal -0.16 0.86 0.48
outer loop
vertex -1.2 0.9 -3.24
vertex 6.09 -0.03 0.91
vertex 6.12 -0 0.87
endloop
endfacet
facet normal -0.16 0.86 0.48
outer loop
vertex -1.2 0.9 -3.24
vertex 6.12 -0 0.87
vertex -1.17 0.93 -3.28
endloop
endfacet
facet normal 0.5 -0.87 0
outer loop
vertex 7.87 -0 0.9
vertex 7.87 -0 0
vertex 7.92 0.03 0
endloop
endfacet
facet normal 0.5 -0.87 0
outer loop
vertex 7.87 -0 0.9
vertex 7.92 0.03 0
vertex 7.92 0.03 0.9
endloop
endfacet
facet normal -1 -0 -0
outer loop
vertex 7.92 0.03 0.9
vertex 7.92 0.03 0
vertex 7.92 -0.03 0
endloop
endfacet
facet normal -1 -0 0
outer loop
vertex 7.92 0.03 0.9
vertex 7.92 -0.03 0
vertex 7.92 -0.03 0.9
endloop
endfacet
facet normal 0.5 0.87 0
outer loop
vertex 7.92 -0.03 0.9
vertex 7.92 -0.03 0
vertex 7.87 -0 0
endloop
endfacet
facet normal 0.5 0.87 0
outer loop
vertex 7.92 -0.03 0.9
vertex 7.87 -0 0
vertex 7.87 -0 0.9
endloop
endfacet
facet normal -0.28 -0.86 0.42
outer loop
vertex -6.99 -0 -2.08
vertex -0.45 -0.76 0.71
vertex -0.47 -0.73 0.75
endloop
endfacet
facet normal -0.28 -0.86 0.42
outer loop
vertex -6.99 -0 -2.08
vertex -0.47 -0.73 0.75
vertex -7 0.03 -2.03
endloop
endfacet
facet normal 0.39 -0 -0.92
outer loop
vertex -7 0.03 -2.03
vertex -0.47 -0.73 0.75
vertex -0.48 -0.79 0.75
endloop
endfacet
facet normal 0.39 -0 -0.92
outer loop
vertex -7 0.03 -2.03
vertex -0.48 -0.79 0.75
vertex -7.01 -0.03 -2.04
endloop
endfacet
facet normal -0.11 0.86 0.5
outer loop
vertex -7.01 -0.03 -2.04
vertex -0.48 -0.79 0.75
vertex -0.45 -0.76 0.71
endloop
endfacet
facet normal -0.11 0.86 0.5
outer loop
vertex -7.01 -0.03 -2.04
vertex -0.45 -0.76 0.71
vertex -6.99 -0 -2.08
endloop
endfacet
facet normal -0.14 -0.84 0.52
outer loop
vertex -7.55 -0 2.02
vertex 1.2 -2.17 0.93
vertex 1.21 -2.14 0.98
endloop
endfacet
facet normal -0.14 -0.84 0.52
outer loop
vertex -7.55 -0 2.02
vertex 1.21 -2.14 0.98
vertex -7.54 0.03 2.07
endloop
endfacet
facet normal -0.12 -0 -0.99
outer loop
vertex -7.54 0.03 2.07
vertex 1.21 -2.14 0.98
vertex 1.2 -2.2 0.98
endloop
endfacet
facet normal -0.12 -0 -0.99
outer loop
vertex -7.54 0.03 2.07
vertex 1.2 -2.2 0.98
vertex -7.55 -0.03 2.07
endloop
endfacet
facet normal 0.27 0.84 0.47
outer loop
vertex -7.55 -0.03 2.07
vertex 1.2 -2.2 0.98
vertex 1.2 -2.17 0.93
endloop
endfacet
facet normal 0.27 0.84 0.47
outer loop
vertex -7.55 -0.03 2.07
vertex 1.2 -2.17 0.93
vertex -7.55 -0 2.02
endloop
endfacet
facet normal 0.33 -0.86 0.4
outer loop
vertex -3.1 1.3 2.52
vertex -0.53 0.59 -1.12
vertex -0.49 0.62 -1.09
endloop
endfacet
facet normal 0.33 -0.86 0.4
outer loop
vertex -3.1 1.3 2.52
vertex -0.49 0.62 -1.09
vertex -3.06 1.33 2.55
endloop
endfacet
facet normal -0.82 -0 -0.58
outer loop
vertex -3.06 1.33 2.55
vertex -0.49 0.62 -1.09
vertex -0.49 0.56 -1.09
endloop
endfacet
facet normal -0.82 -0 -0.58
outer loop
vertex -3.06 1.33 2.55
vertex -0.49 0.56 -1.09
vertex -3.06 1.27 2.55
endloop
endfacet
facet normal 0.49 0.86 0.18
outer loop
vertex -3.06 1.27 2.55
vertex -0.49 0.56 -1.09
vertex -0.53 0.59 -1.12
endloop
endfacet
facet normal 0.49 0.86 0.18
outer loop
vertex -3.06 1.27 2.55
vertex -0.53 0.59 -1.12
vertex -3.1 1.3 2.52
endloop
endfacet
facet normal 0.23 -0.86 0.45
outer loop
vertex -6.47 -0 0.52
vertex -4.26 -0.21 -1.04
vertex -4.23 -0.18 -1
endloop
endfacet
facet normal 0.23 -0.86 0.45
outer loop
vertex -6.47 -0 0.52
vertex -4.23 -0.18 -1
vertex -6.44 0.03 0.56
endloop
endfacet
facet normal -0.58 -0 -0.82
outer loop
vertex -6.44 0.03 0.56
vertex -4.23 -0.18 -1
vertex -4.23 -0.24 -1
endloop
endfacet
facet normal -0.58 -0 -0.82
outer loop
vertex -6.44 0.03 0.56
vertex -4.23 -0.24 -1
vertex -6.44 -0.03 0.56
endloop
endfacet
facet normal 0.34 0.86 0.37
outer loop
vertex -6.44 -0.03 0.56
vertex -4.23 -0.24 -1
vertex -4.26 -0.21 -1.04
endloop
endfacet
facet normal 0.34 0.86 0.37
outer loop
vertex -6.44 -0.03 0.56
vertex -4.26 -0.21 -1.04
vertex -6.47 -0 0.52
endloop
endfacet
facet normal -0.5 -0.87 0
outer loop
vertex 7.93 -0 -0.9
vertex 7.93 -0 0
vertex 7.88 0.03 0
endloop
endfacet
facet normal -0.5 -0.87 0
outer loop
vertex 7.93 -0 -0.9
vertex 7.88 0.03 0
vertex 7.88 0.03 -0.9
endloop
endfacet
facet normal 1 -0 0
outer loop
vertex 7.88 0.03 -0.9
vertex 7.88 0.03 0
vertex 7.88 -0.03 0
endloop
endfacet
facet normal 1 -0 0
outer loop
vertex 7.88 0.03 -0.9
vertex 7.88 -0.03 0
vertex 7.88 -0.03 -0.9
endloop
endfacet
facet normal -0.5 0.87 0
outer loop
vertex 7.88 -0.03 -0.9
vertex 7.88 -0.03 0
vertex 7.93 -0 0
endloop
endfacet
facet normal -0.5 0.87 0
outer loop
vertex 7.88 -0.03 -0.9
vertex 7.93 -0 0
vertex 7.93 -0 -0.9
endloop
endfacet
facet normal 0 -0.86 0.51
outer loop
vertex -7.55 -0 -2.08
vertex -0.5 0.59 -1.13
vertex -0.51 0.62 -1.08
endloop
endfacet
facet normal 0 -0.86 0.51
outer loop
vertex -7.55 -0 -2.08
vertex -0.51 0.62 -1.08
vertex -7.55 0.03 -2.03
endloop
endfacet
facet normal 0.13 -0 -0.99
outer loop
vertex -7.55 0.03 -2.03
vertex -0.51 0.62 -1.08
vertex -0.5 0.56 -1.08
endloop
endfacet
facet normal 0.13 -0 -0.99
outer loop
vertex -7.55 0.03 -2.03
vertex -0.5 0.56 -1.08
vertex -7.55 -0.03 -2.03
endloop
endfacet
facet normal -0.14 0.86 0.49
outer loop
vertex -7.55 -0.03 -2.03
vertex -0.5 0.56 -1.08
vertex -0.5 0.59 -1.13
endloop
endfacet
facet normal -0.14 0.86 0.49
outer loop
vertex -7.55 -0.03 -2.03
vertex -0.5 0.59 -1.13
vertex -7.55 -0 -2.08
endloop
endfacet
facet normal -0.14 -0.86 -0.49
outer loop
vertex 6.1 -0 0.03
vertex -1.78 0.93 0.73
vertex -1.78 0.95 0.68
endloop
endfacet
facet normal -0.14 -0.86 -0.49
outer loop
vertex 6.1 -0 0.03
vertex -1.78 0.95 0.68
vertex 6.1 0.03 -0.02
endloop
endfacet
facet normal 0.09 -0 1
outer loop
vertex 6.1 0.03 -0.02
vertex -1.78 0.95 0.68
vertex -1.79 0.9 0.68
endloop
endfacet
facet normal 0.09 -0 1
outer loop
vertex 6.1 0.03 -0.02
vertex -1.79 0.9 0.68
vertex 6.1 -0.03 -0.02
endloop
endfacet
facet normal 0.06 0.86 -0.51
outer loop
vertex 6.1 -0.03 -0.02
vertex -1.79 0.9 0.68
vertex -1.78 0.93 0.73
endloop
endfacet
facet normal 0.06 0.86 -0.51
outer loop
vertex 6.1 -0.03 -0.02
vertex -1.78 0.93 0.73
vertex 6.1 -0 0.03
endloop
endfacet
facet normal -0.04 -0.87 0.5
outer loop
vertex -7 -0 -2.08
vertex 7.9 -0 -0.93
vertex 7.9 0.03 -0.88
endloop
endfacet
facet normal -0.04 -0.87 0.5
outer loop
vertex -7 -0 -2.08
vertex 7.9 0.03 -0.88
vertex -7 0.03 -2.03
endloop
endfacet
facet normal 0.08 -0 -1
outer loop
vertex -7 0.03 -2.03
vertex 7.9 0.03 -0.88
vertex 7.9 -0.03 -0.88
endloop
endfacet
facet normal 0.08 -0 -1
outer loop
vertex -7 0.03 -2.03
vertex 7.9 -0.03 -0.88
vertex -7 -0.03 -2.03
endloop
endfacet
facet normal -0.04 0.87 0.5
outer loop
vertex -7 -0.03 -2.03
vertex 7.9 -0.03 -0.88
vertex 7.9 -0 -0.93
endloop
endfacet
facet normal -0.04 0.87 0.5
outer loop
vertex -7 -0.03 -2.03
vertex 7.9 -0 -0.93
vertex -7 -0 -2.08
endloop
endfacet
facet normal -0.27 -0.55 0.79
outer loop
vertex -1.15 0.93 -3.26
vertex -0.7 2.72 -1.84
vertex -0.75 2.74 -1.85
endloop
endfacet
facet normal -0.27 -0.55 0.79
outer loop
vertex -1.15 0.93 -3.26
vertex -0.75 2.74 -1.85
vertex -1.21 0.95 -3.27
endloop
endfacet
facet normal 0.95 -0 -0.31
outer loop
vertex -1.21 0.95 -3.27
vertex -0.75 2.74 -1.85
vertex -0.74 2.7 -1.81
endloop
endfacet
facet normal 0.95 -0 -0.31
outer loop
vertex -1.21 0.95 -3.27
vertex -0.74 2.7 -1.81
vertex -1.19 0.91 -3.23
endloop
endfacet
facet normal -0.68 0.55 -0.48
outer loop
vertex -1.19 0.91 -3.23
vertex -0.74 2.7 -1.81
vertex -0.7 2.72 -1.84
endloop
endfacet
facet normal -0.68 0.55 -0.48
outer loop
vertex -1.19 0.91 -3.23
vertex -0.7 2.72 -1.84
vertex -1.15 0.93 -3.26
endloop
endfacet
facet normal 0.92 -0.36 0.17
outer loop
vertex -2.01 -2.39 1.11
vertex 0 2.33 0.3
vertex -0.01 2.34 0.36
endloop
endfacet
facet normal 0.92 -0.36 0.17
outer loop
vertex -2.01 -2.39 1.11
vertex -0.01 2.34 0.36
vertex -2.01 -2.38 1.16
endloop
endfacet
facet normal -0.37 -0 -0.93
outer loop
vertex -2.01 -2.38 1.16
vertex -0.01 2.34 0.36
vertex 0.04 2.32 0.34
endloop
endfacet
facet normal -0.37 -0 -0.93
outer loop
vertex -2.01 -2.38 1.16
vertex 0.04 2.32 0.34
vertex -1.96 -2.4 1.15
endloop
endfacet
facet normal -0.54 0.36 0.76
outer loop
vertex -1.96 -2.4 1.15
vertex 0.04 2.32 0.34
vertex 0 2.33 0.3
endloop
endfacet
facet normal -0.54 0.36 0.76
outer loop
vertex -1.96 -2.4 1.15
vertex 0 2.33 0.3
vertex -2.01 -2.39 1.11
endloop
endfacet
facet normal 0.32 -0.82 -0.47
outer loop
vertex 7 -0 0.03
vertex -0.43 -2.55 -0.61
vertex -0.43 -2.53 -0.66
endloop
endfacet
facet normal 0.32 -0.82 -0.47
outer loop
vertex 7 -0 0.03
vertex -0.43 -2.53 -0.66
vertex 6.99 0.03 -0.02
endloop
endfacet
facet normal -0.09 -0 1
outer loop
vertex 6.99 0.03 -0.02
vertex -0.43 -2.53 -0.66
vertex -0.42 -2.58 -0.66
endloop
endfacet
facet normal -0.09 -0 1
outer loop
vertex 6.99 0.03 -0.02
vertex -0.42 -2.58 -0.66
vertex 7.01 -0.03 -0.02
endloop
endfacet
facet normal -0.24 0.82 -0.52
outer loop
vertex 7.01 -0.03 -0.02
vertex -0.42 -2.58 -0.66
vertex -0.43 -2.55 -0.61
endloop
endfacet
facet normal -0.24 0.82 -0.52
outer loop
vertex 7.01 -0.03 -0.02
vertex -0.43 -2.55 -0.61
vertex 7 -0 0.03
endloop
endfacet
facet normal -0.02 -0.87 0.5
outer loop
vertex -7 -0 -0.58
vertex 7.9 -0 -0.03
vertex 7.9 0.03 0.02
endloop
endfacet
facet normal -0.02 -0.87 0.5
outer loop
vertex -7 -0 -0.58
vertex 7.9 0.03 0.02
vertex -7 0.03 -0.53
endloop
endfacet
facet normal 0.04 -0 -1
outer loop
vertex -7 0.03 -0.53
vertex 7.9 0.03 0.02
vertex 7.9 -0.03 0.02
endloop
endfacet
facet normal 0.04 -0 -1
outer loop
vertex -7 0.03 -0.53
vertex 7.9 -0.03 0.02
vertex -7 -0.03 -0.53
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -7 -0.03 -0.53
vertex 7.9 -0.03 0.02
vertex 7.9 -0 -0.03
endloop
endfacet
facet normal -0.02 0.87 0.5
outer loop
vertex -7 -0.03 -0.53
vertex 7.9 -0 -0.03
vertex -7 -0 -0.58
endloop
endfacet
facet normal -0.04 -0.87 -0.5
outer loop
vertex 1.15 -0.89 -0.94
vertex -2.09 -0.89 -0.65
vertex -2.1 -0.86 -0.7
endloop
endfacet
facet normal -0.04 -0.87 -0.5
outer loop
vertex 1.15 -0.89 -0.94
vertex -2.1 -0.86 -0.7
vertex 1.14 -0.86 -0.99
endloop
endfacet
facet normal 0.09 -0 1
outer loop
vertex 1.14 -0.86 -0.99
vertex -2.1 -0.86 -0.7
vertex -2.1 -0.92 -0.7
endloop
endfacet
facet normal 0.09 -0 1
outer loop
vertex 1.14 -0.86 -0.99
vertex -2.1 -0.92 -0.7
vertex 1.14 -0.91 -0.99
endloop
endfacet
facet normal -0.05 0.87 -0.5
outer loop
vertex 1.14 -0.91 -0.99
vertex -2.1 -0.92 -0.7
vertex -2.09 -0.89 -0.65
endloop
endfacet
facet normal -0.05 0.87 -0.5
outer loop
vertex 1.14 -0.91 -0.99
vertex -2.09 -0.89 -0.65
vertex 1.15 -0.89 -0.94
endloop
endfacet
facet normal -0.14 -0.78 0.61
outer loop
vertex -6.46 -0 3.12
vertex -2.01 -2.39 1.11
vertex -1.98 -2.37 1.15
endloop
endfacet
facet normal -0.14 -0.78 0.61
outer loop
vertex -6.46 -0 3.12
vertex -1.98 -2.37 1.15
vertex -6.43 0.03 3.16
endloop
endfacet
facet normal -0.41 -0 -0.91
outer loop
vertex -6.43 0.03 3.16
vertex -1.98 -2.37 1.15
vertex -2 -2.42 1.16
endloop
endfacet
facet normal -0.41 -0 -0.91
outer loop
vertex -6.43 0.03 3.16
vertex -2 -2.42 1.16
vertex -6.45 -0.03 3.17
endloop
endfacet
facet normal 0.55 0.78 0.3
outer loop
vertex -6.45 -0.03 3.17
vertex -2 -2.42 1.16
vertex -2.01 -2.39 1.11
endloop
endfacet
facet normal 0.55 0.78 0.3
outer loop
vertex -6.45 -0.03 3.17
vertex -2.01 -2.39 1.11
vertex -6.46 -0 3.12
endloop
endfacet
facet normal 0.21 -0.86 0.47
outer loop
vertex -7.02 -0 3.12
vertex -2.12 -0.89 -0.71
vertex -2.08 -0.86 -0.68
endloop
endfacet
facet normal 0.21 -0.86 0.47
outer loop
vertex -7.02 -0 3.12
vertex -2.08 -0.86 -0.68
vertex -6.99 0.03 3.16
endloop
endfacet
facet normal -0.62 -0 -0.79
outer loop
vertex -6.99 0.03 3.16
vertex -2.08 -0.86 -0.68
vertex -2.09 -0.92 -0.67
endloop
endfacet
facet normal -0.62 -0 -0.79
outer loop
vertex -6.99 0.03 3.16
vertex -2.09 -0.92 -0.67
vertex -6.99 -0.03 3.17
endloop
endfacet
facet normal 0.4 0.86 0.32
outer loop
vertex -6.99 -0.03 3.17
vertex -2.09 -0.92 -0.67
vertex -2.12 -0.89 -0.71
endloop
endfacet
facet normal 0.4 0.86 0.32
outer loop
vertex -6.99 -0.03 3.17
vertex -2.12 -0.89 -0.71
vertex -7.02 -0 3.12
endloop
endfacet
facet normal 0.61 -0.77 0.2
outer loop
vertex -7.02 -0 3.12
vertex -2.14 3.06 -0.01
vertex -2.13 3.09 0.03
endloop
endfacet
facet normal 0.61 -0.77 0.2
outer loop
vertex -7.02 -0 3.12
vertex -2.13 3.09 0.03
vertex -7 0.03 3.17
endloop
endfacet
facet normal -0.54 -0 -0.84
outer loop
vertex -7 0.03 3.17
vertex -2.13 3.09 0.03
vertex -2.11 3.04 0.02
endloop
endfacet
facet normal -0.54 -0 -0.84
outer loop
vertex -7 0.03 3.17
vertex -2.11 3.04 0.02
vertex -6.98 -0.02 3.16
endloop
endfacet
facet normal -0.07 0.77 0.64
outer loop
vertex -6.98 -0.02 3.16
vertex -2.11 3.04 0.02
vertex -2.14 3.06 -0.01
endloop
endfacet
facet normal -0.07 0.77 0.64
outer loop
vertex -6.98 -0.02 3.16
vertex -2.14 3.06 -0.01
vertex -7.02 -0 3.12
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -6.45 -0 -0.58
vertex 7.9 -0 -0.93
vertex 7.9 0.03 -0.88
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -6.45 -0 -0.58
vertex 7.9 0.03 -0.88
vertex -6.45 0.03 -0.53
endloop
endfacet
facet normal -0.02 -0 -1
outer loop
vertex -6.45 0.03 -0.53
vertex 7.9 0.03 -0.88
vertex 7.9 -0.03 -0.88
endloop
endfacet
facet normal -0.02 -0 -1
outer loop
vertex -6.45 0.03 -0.53
vertex 7.9 -0.03 -0.88
vertex -6.45 -0.03 -0.53
endloop
endfacet
facet normal 0.01 0.87 0.5
outer loop
vertex -6.45 -0.03 -0.53
vertex 7.9 -0.03 -0.88
vertex 7.9 -0 -0.93
endloop
endfacet
facet normal 0.01 0.87 0.5
outer loop
vertex -6.45 -0.03 -0.53
vertex 7.9 -0 -0.93
vertex -6.45 -0 -0.58
endloop
endfacet
facet normal 0.07 -0.86 -0.5
outer loop
vertex 7 -0 -0.87
vertex -2.1 -0.89 -0.65
vertex -2.1 -0.86 -0.7
endloop
endfacet
facet normal 0.07 -0.86 -0.5
outer loop
vertex 7 -0 -0.87
vertex -2.1 -0.86 -0.7
vertex 7 0.03 -0.92
endloop
endfacet
facet normal 0.02 -0 1
outer loop
vertex 7 0.03 -0.92
vertex -2.1 -0.86 -0.7
vertex -2.09 -0.92 -0.7
endloop
endfacet
facet normal 0.02 -0 1
outer loop
vertex 7 0.03 -0.92
vertex -2.09 -0.92 -0.7
vertex 7 -0.03 -0.92
endloop
endfacet
facet normal -0.1 0.86 -0.5
outer loop
vertex 7 -0.03 -0.92
vertex -2.09 -0.92 -0.7
vertex -2.1 -0.89 -0.65
endloop
endfacet
facet normal -0.1 0.86 -0.5
outer loop
vertex 7 -0.03 -0.92
vertex -2.1 -0.89 -0.65
vertex 7 -0 -0.87
endloop
endfacet
facet normal -0.52 -0.68 -0.52
outer loop
vertex -1.75 0.93 0.69
vertex -1.69 -0.58 2.61
vertex -1.74 -0.56 2.62
endloop
endfacet
facet normal -0.52 -0.68 -0.52
outer loop
vertex -1.75 0.93 0.69
vertex -1.74 -0.56 2.62
vertex -1.8 0.95 0.71
endloop
endfacet
facet normal 1 -0 -0.03
outer loop
vertex -1.8 0.95 0.71
vertex -1.74 -0.56 2.62
vertex -1.74 -0.61 2.59
endloop
endfacet
facet normal 1 -0 -0.03
outer loop
vertex -1.8 0.95 0.71
vertex -1.74 -0.61 2.59
vertex -1.8 0.9 0.68
endloop
endfacet
facet normal -0.48 0.68 0.55
outer loop
vertex -1.8 0.9 0.68
vertex -1.74 -0.61 2.59
vertex -1.69 -0.58 2.61
endloop
endfacet
facet normal -0.48 0.68 0.55
outer loop
vertex -1.8 0.9 0.68
vertex -1.69 -0.58 2.61
vertex -1.75 0.93 0.69
endloop
endfacet
facet normal -0.38 -0.85 0.36
outer loop
vertex -6.43 -0 -3.18
vertex -2.08 -0.89 -0.72
vertex -2.1 -0.86 -0.67
endloop
endfacet
facet normal -0.38 -0.85 0.36
outer loop
vertex -6.43 -0 -3.18
vertex -2.1 -0.86 -0.67
vertex -6.45 0.03 -3.13
endloop
endfacet
facet normal 0.49 -0 -0.87
outer loop
vertex -6.45 0.03 -3.13
vertex -2.1 -0.86 -0.67
vertex -2.11 -0.92 -0.68
endloop
endfacet
facet normal 0.49 -0 -0.87
outer loop
vertex -6.45 0.03 -3.13
vertex -2.11 -0.92 -0.68
vertex -6.46 -0.03 -3.14
endloop
endfacet
facet normal -0.11 0.85 0.51
outer loop
vertex -6.46 -0.03 -3.14
vertex -2.11 -0.92 -0.68
vertex -2.08 -0.89 -0.72
endloop
endfacet
facet normal -0.11 0.85 0.51
outer loop
vertex -6.46 -0.03 -3.14
vertex -2.08 -0.89 -0.72
vertex -6.43 -0 -3.18
endloop
endfacet
facet normal -0 -0.84 0.55
outer loop
vertex -7.54 -0 0.52
vertex -3.06 1.3 2.51
vertex -3.09 1.33 2.55
endloop
endfacet
facet normal -0 -0.84 0.55
outer loop
vertex -7.54 -0 0.52
vertex -3.09 1.33 2.55
vertex -7.56 0.03 0.56
endloop
endfacet
facet normal 0.41 -0 -0.91
outer loop
vertex -7.56 0.03 0.56
vertex -3.09 1.33 2.55
vertex -3.07 1.27 2.56
endloop
endfacet
facet normal 0.41 -0 -0.91
outer loop
vertex -7.56 0.03 0.56
vertex -3.07 1.27 2.56
vertex -7.55 -0.03 0.57
endloop
endfacet
facet normal -0.41 0.84 0.37
outer loop
vertex -7.55 -0.03 0.57
vertex -3.07 1.27 2.56
vertex -3.06 1.3 2.51
endloop
endfacet
facet normal -0.41 0.84 0.37
outer loop
vertex -7.55 -0.03 0.57
vertex -3.06 1.3 2.51
vertex -7.54 -0 0.52
endloop
endfacet
facet normal 0.11 -0.86 -0.5
outer loop
vertex -0.48 -0.76 0.77
vertex -4.25 -0.21 -0.98
vertex -4.23 -0.18 -1.03
endloop
endfacet
facet normal 0.11 -0.86 -0.5
outer loop
vertex -0.48 -0.76 0.77
vertex -4.23 -0.18 -1.03
vertex -0.46 -0.73 0.72
endloop
endfacet
facet normal -0.42 -0 0.91
outer loop
vertex -0.46 -0.73 0.72
vertex -4.23 -0.18 -1.03
vertex -4.24 -0.24 -1.03
endloop
endfacet
facet normal -0.42 -0 0.91
outer loop
vertex -0.46 -0.73 0.72
vertex -4.24 -0.24 -1.03
vertex -0.46 -0.79 0.72
endloop
endfacet
facet normal 0.31 0.86 -0.41
outer loop
vertex -0.46 -0.79 0.72
vertex -4.24 -0.24 -1.03
vertex -4.25 -0.21 -0.98
endloop
endfacet
facet normal 0.31 0.86 -0.41
outer loop
vertex -0.46 -0.79 0.72
vertex -4.25 -0.21 -0.98
vertex -0.48 -0.76 0.77
endloop
endfacet
facet normal -0.03 -0.87 0.5
outer loop
vertex -6.45 -0 -0.03
vertex 7 -0 0.87
vertex 7 0.03 0.92
endloop
endfacet
facet normal -0.03 -0.87 0.5
outer loop
vertex -6.45 -0 -0.03
vertex 7 0.03 0.92
vertex -6.45 0.03 0.02
endloop
endfacet
facet normal 0.07 -0 -1
outer loop
vertex -6.45 0.03 0.02
vertex 7 0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal 0.07 -0 -1
outer loop
vertex -6.45 0.03 0.02
vertex 7 -0.03 0.92
vertex -6.45 -0.03 0.02
endloop
endfacet
facet normal -0.03 0.87 0.5
outer loop
vertex -6.45 -0.03 0.02
vertex 7 -0.03 0.92
vertex 7 -0 0.87
endloop
endfacet
facet normal -0.03 0.87 0.5
outer loop
vertex -6.45 -0.03 0.02
vertex 7 -0 0.87
vertex -6.45 -0 -0.03
endloop
endfacet
facet normal -0.4 -0.86 0.31
outer loop
vertex -7.28 -0 -4.42
vertex -4.22 -0.21 -1.03
vertex -4.25 -0.18 -1
endloop
endfacet
facet normal -0.4 -0.86 0.31
outer loop
vertex -7.28 -0 -4.42
vertex -4.25 -0.18 -1
vertex -7.31 0.03 -4.39
endloop
endfacet
facet normal 0.74 -0 -0.67
outer loop
vertex -7.31 0.03 -4.39
vertex -4.25 -0.18 -1
vertex -4.25 -0.24 -1
endloop
endfacet
facet normal 0.74 -0 -0.67
outer loop
vertex -7.31 0.03 -4.39
vertex -4.25 -0.24 -1
vertex -7.31 -0.03 -4.39
endloop
endfacet
facet normal -0.34 0.87 0.36
outer loop
vertex -7.31 -0.03 -4.39
vertex -4.25 -0.24 -1
vertex -4.22 -0.21 -1.03
endloop
endfacet
facet normal -0.34 0.87 0.36
outer loop
vertex -7.31 -0.03 -4.39
vertex -4.22 -0.21 -1.03
vertex -7.28 -0 -4.42
endloop
endfacet
facet normal 0.39 -0.86 -0.32
outer loop
vertex 1.28 0.54 0.94
vertex -1.21 0.93 -3.23
vertex -1.17 0.96 -3.26
endloop
endfacet
facet normal 0.39 -0.86 -0.32
outer loop
vertex 1.28 0.54 0.94
vertex -1.17 0.96 -3.26
vertex 1.32 0.57 0.92
endloop
endfacet
facet normal -0.86 -0 0.51
outer loop
vertex 1.32 0.57 0.92
vertex -1.17 0.96 -3.26
vertex -1.17 0.91 -3.26
endloop
endfacet
facet normal -0.86 -0 0.51
outer loop
vertex 1.32 0.57 0.92
vertex -1.17 0.91 -3.26
vertex 1.32 0.51 0.91
endloop
endfacet
facet normal 0.47 0.86 -0.2
outer loop
vertex 1.32 0.51 0.91
vertex -1.17 0.91 -3.26
vertex -1.21 0.93 -3.23
endloop
endfacet
facet normal 0.47 0.86 -0.2
outer loop
vertex 1.32 0.51 0.91
vertex -1.21 0.93 -3.23
vertex 1.28 0.54 0.94
endloop
endfacet
facet normal 0.3 -0.86 0.42
outer loop
vertex -7.56 -0 -0.58
vertex -1.2 0.93 -3.28
vertex -1.18 0.96 -3.23
endloop
endfacet
facet normal 0.3 -0.86 0.42
outer loop
vertex -7.56 -0 -0.58
vertex -1.18 0.96 -3.23
vertex -7.55 0.03 -0.53
endloop
endfacet
facet normal -0.39 -0 -0.92
outer loop
vertex -7.55 0.03 -0.53
vertex -1.18 0.96 -3.23
vertex -1.18 0.91 -3.24
endloop
endfacet
facet normal -0.39 -0 -0.92
outer loop
vertex -7.55 0.03 -0.53
vertex -1.18 0.91 -3.24
vertex -7.54 -0.03 -0.54
endloop
endfacet
facet normal 0.09 0.86 0.51
outer loop
vertex -7.54 -0.03 -0.54
vertex -1.18 0.91 -3.24
vertex -1.2 0.93 -3.28
endloop
endfacet
facet normal 0.09 0.86 0.51
outer loop
vertex -7.54 -0.03 -0.54
vertex -1.2 0.93 -3.28
vertex -7.56 -0 -0.58
endloop
endfacet
facet normal 0.35 -0.86 0.37
outer loop
vertex -6.48 -0 2.03
vertex -2.87 -0.6 -2.85
vertex -2.83 -0.58 -2.82
endloop
endfacet
facet normal 0.35 -0.86 0.37
outer loop
vertex -6.48 -0 2.03
vertex -2.83 -0.58 -2.82
vertex -6.44 0.03 2.06
endloop
endfacet
facet normal -0.8 -0 -0.59
outer loop
vertex -6.44 0.03 2.06
vertex -2.83 -0.58 -2.82
vertex -2.84 -0.63 -2.82
endloop
endfacet
facet normal -0.8 -0 -0.59
outer loop
vertex -6.44 0.03 2.06
vertex -2.84 -0.63 -2.82
vertex -6.44 -0.03 2.06
endloop
endfacet
facet normal 0.45 0.86 0.23
outer loop
vertex -6.44 -0.03 2.06
vertex -2.84 -0.63 -2.82
vertex -2.87 -0.6 -2.85
endloop
endfacet
facet normal 0.45 0.86 0.23
outer loop
vertex -6.44 -0.03 2.06
vertex -2.87 -0.6 -2.85
vertex -6.48 -0 2.03
endloop
endfacet
facet normal -0.08 -0.87 0.49
outer loop
vertex -6.45 -0 -3.18
vertex 7.9 -0 -0.93
vertex 7.9 0.03 -0.88
endloop
endfacet
facet normal -0.08 -0.87 0.49
outer loop
vertex -6.45 -0 -3.18
vertex 7.9 0.03 -0.88
vertex -6.45 0.03 -3.13
endloop
endfacet
facet normal 0.15 -0 -0.99
outer loop
vertex -6.45 0.03 -3.13
vertex 7.9 0.03 -0.88
vertex 7.9 -0.03 -0.88
endloop
endfacet
facet normal 0.15 -0 -0.99
outer loop
vertex -6.45 0.03 -3.13
vertex 7.9 -0.03 -0.88
vertex -6.45 -0.03 -3.13
endloop
endfacet
facet normal -0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 -3.13
vertex 7.9 -0.03 -0.88
vertex 7.9 -0 -0.93
endloop
endfacet
facet normal -0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 -3.13
vertex 7.9 -0 -0.93
vertex -6.45 -0 -3.18
endloop
endfacet
facet normal -0.08 -0.87 0.49
outer loop
vertex -7.54 -0 -3.18
vertex 6.11 -0 -0.93
vertex 6.1 0.03 -0.88
endloop
endfacet
facet normal -0.08 -0.87 0.49
outer loop
vertex -7.54 -0 -3.18
vertex 6.1 0.03 -0.88
vertex -7.55 0.03 -3.13
endloop
endfacet
facet normal 0.16 -0 -0.99
outer loop
vertex -7.55 0.03 -3.13
vertex 6.1 0.03 -0.88
vertex 6.1 -0.03 -0.88
endloop
endfacet
facet normal 0.16 -0 -0.99
outer loop
vertex -7.55 0.03 -3.13
vertex 6.1 -0.03 -0.88
vertex -7.55 -0.03 -3.13
endloop
endfacet
facet normal -0.08 0.87 0.49
outer loop
vertex -7.55 -0.03 -3.13
vertex 6.1 -0.03 -0.88
vertex 6.11 -0 -0.93
endloop
endfacet
facet normal -0.08 0.87 0.49
outer loop
vertex -7.55 -0.03 -3.13
vertex 6.11 -0 -0.93
vertex -7.54 -0 -3.18
endloop
endfacet
facet normal 0.11 -0.83 -0.55
outer loop
vertex 7.91 -0 -0.87
vertex 1.21 -2.17 0.99
vertex 1.19 -2.15 0.95
endloop
endfacet
facet normal 0.11 -0.83 -0.55
outer loop
vertex 7.91 -0 -0.87
vertex 1.19 -2.15 0.95
vertex 7.89 0.03 -0.91
endloop
endfacet
facet normal 0.27 -0 0.96
outer loop
vertex 7.89 0.03 -0.91
vertex 1.19 -2.15 0.95
vertex 1.2 -2.2 0.95
endloop
endfacet
facet normal 0.27 -0 0.96
outer loop
vertex 7.89 0.03 -0.91
vertex 1.2 -2.2 0.95
vertex 7.9 -0.03 -0.92
endloop
endfacet
facet normal -0.38 0.83 -0.41
outer loop
vertex 7.9 -0.03 -0.92
vertex 1.2 -2.2 0.95
vertex 1.21 -2.17 0.99
endloop
endfacet
facet normal -0.38 0.83 -0.41
outer loop
vertex 7.9 -0.03 -0.92
vertex 1.21 -2.17 0.99
vertex 7.91 -0 -0.87
endloop
endfacet
facet normal 0.1 -0.87 0.49
outer loop
vertex -6.46 -0 2.57
vertex 6.09 -0 -0.03
vertex 6.1 0.03 0.02
endloop
endfacet
facet normal 0.1 -0.87 0.49
outer loop
vertex -6.46 -0 2.57
vertex 6.1 0.03 0.02
vertex -6.45 0.03 2.62
endloop
endfacet
facet normal -0.2 -0 -0.98
outer loop
vertex -6.45 0.03 2.62
vertex 6.1 0.03 0.02
vertex 6.1 -0.03 0.02
endloop
endfacet
facet normal -0.2 -0 -0.98
outer loop
vertex -6.45 0.03 2.62
vertex 6.1 -0.03 0.02
vertex -6.45 -0.03 2.62
endloop
endfacet
facet normal 0.1 0.87 0.49
outer loop
vertex -6.45 -0.03 2.62
vertex 6.1 -0.03 0.02
vertex 6.09 -0 -0.03
endloop
endfacet
facet normal 0.1 0.87 0.49
outer loop
vertex -6.45 -0.03 2.62
vertex 6.09 -0 -0.03
vertex -6.46 -0 2.57
endloop
endfacet
facet normal -0.12 -0.86 0.51
outer loop
vertex -3.91 1.61 0.35
vertex 6.1 -0 -0.03
vertex 6.11 0.03 0.02
endloop
endfacet
facet normal -0.12 -0.86 0.51
outer loop
vertex -3.91 1.61 0.35
vertex 6.11 0.03 0.02
vertex -3.91 1.63 0.4
endloop
endfacet
facet normal -0.04 -0 -1
outer loop
vertex -3.91 1.63 0.4
vertex 6.11 0.03 0.02
vertex 6.1 -0.03 0.02
endloop
endfacet
facet normal -0.04 -0 -1
outer loop
vertex -3.91 1.63 0.4
vertex 6.1 -0.03 0.02
vertex -3.92 1.58 0.4
endloop
endfacet
facet normal 0.16 0.86 0.49
outer loop
vertex -3.92 1.58 0.4
vertex 6.1 -0.03 0.02
vertex 6.1 -0 -0.03
endloop
endfacet
facet normal 0.16 0.86 0.49
outer loop
vertex -3.92 1.58 0.4
vertex 6.1 -0 -0.03
vertex -3.91 1.61 0.35
endloop
endfacet
facet normal 0.12 -0.87 0.49
outer loop
vertex -7.56 -0 2.57
vertex 6.99 -0 -0.93
vertex 7 0.03 -0.88
endloop
endfacet
facet normal 0.12 -0.87 0.49
outer loop
vertex -7.56 -0 2.57
vertex 7 0.03 -0.88
vertex -7.55 0.03 2.62
endloop
endfacet
facet normal -0.23 -0 -0.97
outer loop
vertex -7.55 0.03 2.62
vertex 7 0.03 -0.88
vertex 7 -0.03 -0.88
endloop
endfacet
facet normal -0.23 -0 -0.97
outer loop
vertex -7.55 0.03 2.62
vertex 7 -0.03 -0.88
vertex -7.55 -0.03 2.62
endloop
endfacet
facet normal 0.12 0.87 0.49
outer loop
vertex -7.55 -0.03 2.62
vertex 7 -0.03 -0.88
vertex 6.99 -0 -0.93
endloop
endfacet
facet normal 0.12 0.87 0.49
outer loop
vertex -7.55 -0.03 2.62
vertex 6.99 -0 -0.93
vertex -7.56 -0 2.57
endloop
endfacet
facet normal -0.42 -0.86 0.3
outer loop
vertex -6.98 -0 -2.07
vertex -3.47 -0.58 1.14
vertex -3.5 -0.55 1.18
endloop
endfacet
facet normal -0.42 -0.86 0.3
outer loop
vertex -6.98 -0 -2.07
vertex -3.5 -0.55 1.18
vertex -7.01 0.03 -2.04
endloop
endfacet
facet normal 0.68 -0 -0.74
outer loop
vertex -7.01 0.03 -2.04
vertex -3.5 -0.55 1.18
vertex -3.51 -0.61 1.17
endloop
endfacet
facet normal 0.68 -0 -0.74
outer loop
vertex -7.01 0.03 -2.04
vertex -3.51 -0.61 1.17
vertex -7.01 -0.03 -2.04
endloop
endfacet
facet normal -0.26 0.86 0.44
outer loop
vertex -7.01 -0.03 -2.04
vertex -3.51 -0.61 1.17
vertex -3.47 -0.58 1.14
endloop
endfacet
facet normal -0.26 0.86 0.44
outer loop
vertex -7.01 -0.03 -2.04
vertex -3.47 -0.58 1.14
vertex -6.98 -0 -2.07
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -6.44 -0 -3.18
vertex 6.11 -0 -0.93
vertex 6.1 0.03 -0.88
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -6.44 -0 -3.18
vertex 6.1 0.03 -0.88
vertex -6.45 0.03 -3.13
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -6.45 0.03 -3.13
vertex 6.1 0.03 -0.88
vertex 6.1 -0.03 -0.88
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -6.45 0.03 -3.13
vertex 6.1 -0.03 -0.88
vertex -6.45 -0.03 -3.13
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -6.45 -0.03 -3.13
vertex 6.1 -0.03 -0.88
vertex 6.11 -0 -0.93
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -6.45 -0.03 -3.13
vertex 6.11 -0 -0.93
vertex -6.44 -0 -3.18
endloop
endfacet
facet normal -0.06 -0.84 0.54
outer loop
vertex -7.53 -0 -0.03
vertex -3.06 1.3 2.51
vertex -3.09 1.33 2.55
endloop
endfacet
facet normal -0.06 -0.84 0.54
outer loop
vertex -7.53 -0 -0.03
vertex -3.09 1.33 2.55
vertex -7.56 0.03 0.01
endloop
endfacet
facet normal 0.49 -0 -0.87
outer loop
vertex -7.56 0.03 0.01
vertex -3.09 1.33 2.55
vertex -3.08 1.27 2.56
endloop
endfacet
facet normal 0.49 -0 -0.87
outer loop
vertex -7.56 0.03 0.01
vertex -3.08 1.27 2.56
vertex -7.55 -0.03 0.02
endloop
endfacet
facet normal -0.43 0.84 0.33
outer loop
vertex -7.55 -0.03 0.02
vertex -3.08 1.27 2.56
vertex -3.06 1.3 2.51
endloop
endfacet
facet normal -0.43 0.84 0.33
outer loop
vertex -7.55 -0.03 0.02
vertex -3.06 1.3 2.51
vertex -7.53 -0 -0.03
endloop
endfacet
facet normal 0.48 -0.84 0.27
outer loop
vertex -3.09 1.3 2.51
vertex -0.01 2.33 0.31
vertex 0.02 2.35 0.35
endloop
endfacet
facet normal 0.48 -0.84 0.27
outer loop
vertex -3.09 1.3 2.51
vertex 0.02 2.35 0.35
vertex -3.07 1.33 2.56
endloop
endfacet
facet normal -0.58 -0 -0.81
outer loop
vertex -3.07 1.33 2.56
vertex 0.02 2.35 0.35
vertex 0.03 2.3 0.34
endloop
endfacet
facet normal -0.58 -0 -0.81
outer loop
vertex -3.07 1.33 2.56
vertex 0.03 2.3 0.34
vertex -3.06 1.27 2.55
endloop
endfacet
facet normal 0.11 0.84 0.54
outer loop
vertex -3.06 1.27 2.55
vertex 0.03 2.3 0.34
vertex -0.01 2.33 0.31
endloop
endfacet
facet normal 0.11 0.84 0.54
outer loop
vertex -3.06 1.27 2.55
vertex -0.01 2.33 0.31
vertex -3.09 1.3 2.51
endloop
endfacet
facet normal -0.35 -0.86 0.37
outer loop
vertex -6.43 -0 -3.18
vertex -0.45 -0.76 0.71
vertex -0.47 -0.73 0.75
endloop
endfacet
facet normal -0.35 -0.86 0.37
outer loop
vertex -6.43 -0 -3.18
vertex -0.47 -0.73 0.75
vertex -6.46 0.03 -3.14
endloop
endfacet
facet normal 0.54 -0 -0.84
outer loop
vertex -6.46 0.03 -3.14
vertex -0.47 -0.73 0.75
vertex -0.48 -0.79 0.75
endloop
endfacet
facet normal 0.54 -0 -0.84
outer loop
vertex -6.46 0.03 -3.14
vertex -0.48 -0.79 0.75
vertex -6.46 -0.03 -3.14
endloop
endfacet
facet normal -0.2 0.86 0.47
outer loop
vertex -6.46 -0.03 -3.14
vertex -0.48 -0.79 0.75
vertex -0.45 -0.76 0.71
endloop
endfacet
facet normal -0.2 0.86 0.47
outer loop
vertex -6.46 -0.03 -3.14
vertex -0.45 -0.76 0.71
vertex -6.43 -0 -3.18
endloop
endfacet
facet normal -0.3 -0.72 -0.62
outer loop
vertex -2.1 3.06 0.02
vertex -3.04 1.3 2.55
vertex -3.09 1.32 2.55
endloop
endfacet
facet normal -0.3 -0.72 -0.62
outer loop
vertex -2.1 3.06 0.02
vertex -3.09 1.32 2.55
vertex -2.15 3.09 0.02
endloop
endfacet
facet normal 0.94 -0 0.35
outer loop
vertex -2.15 3.09 0.02
vertex -3.09 1.32 2.55
vertex -3.08 1.28 2.52
endloop
endfacet
facet normal 0.94 -0 0.35
outer loop
vertex -2.15 3.09 0.02
vertex -3.08 1.28 2.52
vertex -2.14 3.04 -0.01
endloop
endfacet
facet normal -0.63 0.73 0.27
outer loop
vertex -2.14 3.04 -0.01
vertex -3.08 1.28 2.52
vertex -3.04 1.3 2.55
endloop
endfacet
facet normal -0.63 0.73 0.27
outer loop
vertex -2.14 3.04 -0.01
vertex -3.04 1.3 2.55
vertex -2.1 3.06 0.02
endloop
endfacet
facet normal 0.34 -0.86 0.37
outer loop
vertex -2.02 -2.39 1.12
vertex -0.45 -2.55 -0.66
vertex -0.41 -2.53 -0.63
endloop
endfacet
facet normal 0.34 -0.86 0.37
outer loop
vertex -2.02 -2.39 1.12
vertex -0.41 -2.53 -0.63
vertex -1.98 -2.36 1.15
endloop
endfacet
facet normal -0.75 -0 -0.66
outer loop
vertex -1.98 -2.36 1.15
vertex -0.41 -2.53 -0.63
vertex -0.42 -2.58 -0.63
endloop
endfacet
facet normal -0.75 -0 -0.66
outer loop
vertex -1.98 -2.36 1.15
vertex -0.42 -2.58 -0.63
vertex -1.98 -2.42 1.15
endloop
endfacet
facet normal 0.41 0.86 0.29
outer loop
vertex -1.98 -2.42 1.15
vertex -0.42 -2.58 -0.63
vertex -0.45 -2.55 -0.66
endloop
endfacet
facet normal 0.41 0.86 0.29
outer loop
vertex -1.98 -2.42 1.15
vertex -0.45 -2.55 -0.66
vertex -2.02 -2.39 1.12
endloop
endfacet
facet normal -0.41 -0.84 0.36
outer loop
vertex -6.44 -0 -2.63
vertex 1.21 -2.17 0.94
vertex 1.2 -2.15 0.98
endloop
endfacet
facet normal -0.41 -0.84 0.36
outer loop
vertex -6.44 -0 -2.63
vertex 1.2 -2.15 0.98
vertex -6.45 0.03 -2.58
endloop
endfacet
facet normal 0.42 -0 -0.91
outer loop
vertex -6.45 0.03 -2.58
vertex 1.2 -2.15 0.98
vertex 1.19 -2.2 0.98
endloop
endfacet
facet normal 0.42 -0 -0.91
outer loop
vertex -6.45 0.03 -2.58
vertex 1.19 -2.2 0.98
vertex -6.46 -0.03 -2.59
endloop
endfacet
facet normal -0.02 0.84 0.54
outer loop
vertex -6.46 -0.03 -2.59
vertex 1.19 -2.2 0.98
vertex 1.21 -2.17 0.94
endloop
endfacet
facet normal -0.02 0.84 0.54
outer loop
vertex -6.46 -0.03 -2.59
vertex 1.21 -2.17 0.94
vertex -6.44 -0 -2.63
endloop
endfacet
facet normal 0.11 -0.8 0.59
outer loop
vertex -6.69 -0 -4.43
vertex -0.72 2.72 -1.86
vertex -0.74 2.74 -1.82
endloop
endfacet
facet normal 0.11 -0.8 0.59
outer loop
vertex -6.69 -0 -4.43
vertex -0.74 2.74 -1.82
vertex -6.72 0.03 -4.39
endloop
endfacet
facet normal 0.39 -0 -0.92
outer loop
vertex -6.72 0.03 -4.39
vertex -0.74 2.74 -1.82
vertex -0.72 2.69 -1.81
endloop
endfacet
facet normal 0.39 -0 -0.92
outer loop
vertex -6.72 0.03 -4.39
vertex -0.72 2.69 -1.81
vertex -6.7 -0.03 -4.38
endloop
endfacet
facet normal -0.5 0.8 0.33
outer loop
vertex -6.7 -0.03 -4.38
vertex -0.72 2.69 -1.81
vertex -0.72 2.72 -1.86
endloop
endfacet
facet normal -0.5 0.8 0.33
outer loop
vertex -6.7 -0.03 -4.38
vertex -0.72 2.72 -1.86
vertex -6.69 -0 -4.43
endloop
endfacet
facet normal -0.07 -0.86 -0.5
outer loop
vertex 7.9 -0 0.93
vertex -1.79 0.93 0.73
vertex -1.78 0.95 0.68
endloop
endfacet
facet normal -0.07 -0.86 -0.5
outer loop
vertex 7.9 -0 0.93
vertex -1.78 0.95 0.68
vertex 7.9 0.03 0.88
endloop
endfacet
facet normal -0.02 -0 1
outer loop
vertex 7.9 0.03 0.88
vertex -1.78 0.95 0.68
vertex -1.79 0.9 0.68
endloop
endfacet
facet normal -0.02 -0 1
outer loop
vertex 7.9 0.03 0.88
vertex -1.79 0.9 0.68
vertex 7.9 -0.03 0.88
endloop
endfacet
facet normal 0.09 0.86 -0.5
outer loop
vertex 7.9 -0.03 0.88
vertex -1.79 0.9 0.68
vertex -1.79 0.93 0.73
endloop
endfacet
facet normal 0.09 0.86 -0.5
outer loop
vertex 7.9 -0.03 0.88
vertex -1.79 0.93 0.73
vertex 7.9 -0 0.93
endloop
endfacet
facet normal -0.15 -0.86 0.49
outer loop
vertex -3.91 1.61 0.35
vertex 7 -0 0.87
vertex 7 0.03 0.92
endloop
endfacet
facet normal -0.15 -0.86 0.49
outer loop
vertex -3.91 1.61 0.35
vertex 7 0.03 0.92
vertex -3.91 1.63 0.4
endloop
endfacet
facet normal 0.05 -0 -1
outer loop
vertex -3.91 1.63 0.4
vertex 7 0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal 0.05 -0 -1
outer loop
vertex -3.91 1.63 0.4
vertex 7 -0.03 0.92
vertex -3.92 1.58 0.4
endloop
endfacet
facet normal 0.1 0.86 0.51
outer loop
vertex -3.92 1.58 0.4
vertex 7 -0.03 0.92
vertex 7 -0 0.87
endloop
endfacet
facet normal 0.1 0.86 0.51
outer loop
vertex -3.92 1.58 0.4
vertex 7 -0 0.87
vertex -3.91 1.61 0.35
endloop
endfacet
facet normal 0.37 -0.86 -0.35
outer loop
vertex -1.75 -0.58 2.62
vertex -4.27 -0.21 -0.99
vertex -4.23 -0.18 -1.02
endloop
endfacet
facet normal 0.37 -0.86 -0.35
outer loop
vertex -1.75 -0.58 2.62
vertex -4.23 -0.18 -1.02
vertex -1.71 -0.56 2.6
endloop
endfacet
facet normal -0.82 -0 0.57
outer loop
vertex -1.71 -0.56 2.6
vertex -4.23 -0.18 -1.02
vertex -4.23 -0.24 -1.02
endloop
endfacet
facet normal -0.82 -0 0.57
outer loop
vertex -1.71 -0.56 2.6
vertex -4.23 -0.24 -1.02
vertex -1.71 -0.61 2.6
endloop
endfacet
facet normal 0.45 0.86 -0.23
outer loop
vertex -1.71 -0.61 2.6
vertex -4.23 -0.24 -1.02
vertex -4.27 -0.21 -0.99
endloop
endfacet
facet normal 0.45 0.86 -0.23
outer loop
vertex -1.71 -0.61 2.6
vertex -4.27 -0.21 -0.99
vertex -1.75 -0.58 2.62
endloop
endfacet
facet normal -0.09 -0.83 -0.55
outer loop
vertex -1.17 0.93 -3.23
vertex -4.22 -0.21 -0.99
vertex -4.26 -0.19 -1.02
endloop
endfacet
facet normal -0.09 -0.83 -0.55
outer loop
vertex -1.17 0.93 -3.23
vertex -4.26 -0.19 -1.02
vertex -1.2 0.96 -3.26
endloop
endfacet
facet normal 0.59 -0 0.81
outer loop
vertex -1.2 0.96 -3.26
vertex -4.26 -0.19 -1.02
vertex -4.24 -0.24 -1.03
endloop
endfacet
facet normal 0.59 -0 0.81
outer loop
vertex -1.2 0.96 -3.26
vertex -4.24 -0.24 -1.03
vertex -1.19 0.91 -3.27
endloop
endfacet
facet normal -0.5 0.83 -0.26
outer loop
vertex -1.19 0.91 -3.27
vertex -4.24 -0.24 -1.03
vertex -4.22 -0.21 -0.99
endloop
endfacet
facet normal -0.5 0.83 -0.26
outer loop
vertex -1.19 0.91 -3.27
vertex -4.22 -0.21 -0.99
vertex -1.17 0.93 -3.23
endloop
endfacet
facet normal -0.88 -0.14 0.45
outer loop
vertex -2.22 -2.73 -2.01
vertex -2.57 1.31 -1.46
vertex -2.6 1.31 -1.51
endloop
endfacet
facet normal -0.88 -0.14 0.45
outer loop
vertex -2.22 -2.73 -2.01
vertex -2.6 1.31 -1.51
vertex -2.24 -2.73 -2.06
endloop
endfacet
facet normal 0.84 -0 0.54
outer loop
vertex -2.24 -2.73 -2.06
vertex -2.6 1.31 -1.51
vertex -2.62 1.31 -1.46
endloop
endfacet
facet normal 0.84 -0 0.54
outer loop
vertex -2.24 -2.73 -2.06
vertex -2.62 1.31 -1.46
vertex -2.27 -2.74 -2.02
endloop
endfacet
facet normal 0.04 0.14 -0.99
outer loop
vertex -2.27 -2.74 -2.02
vertex -2.62 1.31 -1.46
vertex -2.57 1.31 -1.46
endloop
endfacet
facet normal 0.04 0.14 -0.99
outer loop
vertex -2.27 -2.74 -2.02
vertex -2.57 1.31 -1.46
vertex -2.22 -2.73 -2.01
endloop
endfacet
facet normal 0.13 -0.74 0.66
outer loop
vertex -6.43 -0 -2.63
vertex -2.11 3.06 -0.01
vertex -2.15 3.09 0.02
endloop
endfacet
facet normal 0.13 -0.74 0.66
outer loop
vertex -6.43 -0 -2.63
vertex -2.15 3.09 0.02
vertex -6.47 0.02 -2.59
endloop
endfacet
facet normal 0.52 -0 -0.86
outer loop
vertex -6.47 0.02 -2.59
vertex -2.15 3.09 0.02
vertex -2.12 3.04 0.03
endloop
endfacet
facet normal 0.52 -0 -0.86
outer loop
vertex -6.47 0.02 -2.59
vertex -2.12 3.04 0.03
vertex -6.45 -0.02 -2.58
endloop
endfacet
facet normal -0.64 0.74 0.2
outer loop
vertex -6.45 -0.02 -2.58
vertex -2.12 3.04 0.03
vertex -2.11 3.06 -0.01
endloop
endfacet
facet normal -0.64 0.74 0.2
outer loop
vertex -6.45 -0.02 -2.58
vertex -2.11 3.06 -0.01
vertex -6.43 -0 -2.63
endloop
endfacet
facet normal 0.47 -0.67 -0.58
outer loop
vertex -2.03 -2.39 1.14
vertex -2.13 -0.89 -0.69
vertex -2.08 -0.87 -0.67
endloop
endfacet
facet normal 0.47 -0.67 -0.58
outer loop
vertex -2.03 -2.39 1.14
vertex -2.08 -0.87 -0.67
vertex -1.98 -2.37 1.16
endloop
endfacet
facet normal -1 -0 0.06
outer loop
vertex -1.98 -2.37 1.16
vertex -2.08 -0.87 -0.67
vertex -2.08 -0.91 -0.71
endloop
endfacet
facet normal -1 -0 0.06
outer loop
vertex -1.98 -2.37 1.16
vertex -2.08 -0.91 -0.71
vertex -1.98 -2.41 1.12
endloop
endfacet
facet normal 0.53 0.67 0.52
outer loop
vertex -1.98 -2.41 1.12
vertex -2.08 -0.91 -0.71
vertex -2.13 -0.89 -0.69
endloop
endfacet
facet normal 0.53 0.67 0.52
outer loop
vertex -1.98 -2.41 1.12
vertex -2.13 -0.89 -0.69
vertex -2.03 -2.39 1.14
endloop
endfacet
facet normal -0.15 -0.86 0.49
outer loop
vertex -7 -0 -0.03
vertex -0.46 -0.76 0.71
vertex -0.47 -0.73 0.75
endloop
endfacet
facet normal -0.15 -0.86 0.49
outer loop
vertex -7 -0 -0.03
vertex -0.47 -0.73 0.75
vertex -7 0.03 0.02
endloop
endfacet
facet normal 0.11 -0 -0.99
outer loop
vertex -7 0.03 0.02
vertex -0.47 -0.73 0.75
vertex -0.47 -0.79 0.75
endloop
endfacet
facet normal 0.11 -0 -0.99
outer loop
vertex -7 0.03 0.02
vertex -0.47 -0.79 0.75
vertex -7 -0.03 0.02
endloop
endfacet
facet normal 0.04 0.86 0.51
outer loop
vertex -7 -0.03 0.02
vertex -0.47 -0.79 0.75
vertex -0.46 -0.76 0.71
endloop
endfacet
facet normal 0.04 0.86 0.51
outer loop
vertex -7 -0.03 0.02
vertex -0.46 -0.76 0.71
vertex -7 -0 -0.03
endloop
endfacet
facet normal 0.49 -0.81 -0.32
outer loop
vertex 1.29 0.54 0.95
vertex -2.11 -0.89 -0.66
vertex -2.1 -0.87 -0.71
endloop
endfacet
facet normal 0.49 -0.81 -0.32
outer loop
vertex 1.29 0.54 0.95
vertex -2.1 -0.87 -0.71
vertex 1.31 0.56 0.91
endloop
endfacet
facet normal -0.43 -0 0.9
outer loop
vertex 1.31 0.56 0.91
vertex -2.1 -0.87 -0.71
vertex -2.08 -0.92 -0.7
endloop
endfacet
facet normal -0.43 -0 0.9
outer loop
vertex 1.31 0.56 0.91
vertex -2.08 -0.92 -0.7
vertex 1.32 0.51 0.91
endloop
endfacet
facet normal -0.06 0.81 -0.58
outer loop
vertex 1.32 0.51 0.91
vertex -2.08 -0.92 -0.7
vertex -2.11 -0.89 -0.66
endloop
endfacet
facet normal -0.06 0.81 -0.58
outer loop
vertex 1.32 0.51 0.91
vertex -2.11 -0.89 -0.66
vertex 1.29 0.54 0.95
endloop
endfacet
facet normal 0.04 -0.87 0.5
outer loop
vertex -6.45 -0 -0.03
vertex 6.1 -0 -0.93
vertex 6.1 0.03 -0.88
endloop
endfacet
facet normal 0.04 -0.87 0.5
outer loop
vertex -6.45 -0 -0.03
vertex 6.1 0.03 -0.88
vertex -6.45 0.03 0.02
endloop
endfacet
facet normal -0.07 -0 -1
outer loop
vertex -6.45 0.03 0.02
vertex 6.1 0.03 -0.88
vertex 6.1 -0.03 -0.88
endloop
endfacet
facet normal -0.07 -0 -1
outer loop
vertex -6.45 0.03 0.02
vertex 6.1 -0.03 -0.88
vertex -6.45 -0.03 0.02
endloop
endfacet
facet normal 0.04 0.87 0.5
outer loop
vertex -6.45 -0.03 0.02
vertex 6.1 -0.03 -0.88
vertex 6.1 -0 -0.93
endloop
endfacet
facet normal 0.04 0.87 0.5
outer loop
vertex -6.45 -0.03 0.02
vertex 6.1 -0 -0.93
vertex -6.45 -0 -0.03
endloop
endfacet
facet normal -0.14 -0.86 0.49
outer loop
vertex -7.55 -0 -3.18
vertex -2.85 -0.6 -2.86
vertex -2.85 -0.58 -2.81
endloop
endfacet
facet normal -0.14 -0.86 0.49
outer loop
vertex -7.55 -0 -3.18
vertex -2.85 -0.58 -2.81
vertex -7.55 0.03 -3.13
endloop
endfacet
facet normal 0.07 -0 -1
outer loop
vertex -7.55 0.03 -3.13
vertex -2.85 -0.58 -2.81
vertex -2.85 -0.63 -2.81
endloop
endfacet
facet normal 0.07 -0 -1
outer loop
vertex -7.55 0.03 -3.13
vertex -2.85 -0.63 -2.81
vertex -7.55 -0.03 -3.13
endloop
endfacet
facet normal 0.08 0.86 0.51
outer loop
vertex -7.55 -0.03 -3.13
vertex -2.85 -0.63 -2.81
vertex -2.85 -0.6 -2.86
endloop
endfacet
facet normal 0.08 0.86 0.51
outer loop
vertex -7.55 -0.03 -3.13
vertex -2.85 -0.6 -2.86
vertex -7.55 -0 -3.18
endloop
endfacet
facet normal -0.29 -0.82 0.5
outer loop
vertex -7.55 -0 -0.58
vertex -0.43 -2.55 -0.67
vertex -0.42 -2.53 -0.62
endloop
endfacet
facet normal -0.29 -0.82 0.5
outer loop
vertex -7.55 -0 -0.58
vertex -0.42 -2.53 -0.62
vertex -7.54 0.03 -0.53
endloop
endfacet
facet normal -0.01 -0 -1
outer loop
vertex -7.54 0.03 -0.53
vertex -0.42 -2.53 -0.62
vertex -0.44 -2.58 -0.62
endloop
endfacet
facet normal -0.01 -0 -1
outer loop
vertex -7.54 0.03 -0.53
vertex -0.44 -2.58 -0.62
vertex -7.56 -0.03 -0.53
endloop
endfacet
facet normal 0.3 0.82 0.5
outer loop
vertex -7.56 -0.03 -0.53
vertex -0.44 -2.58 -0.62
vertex -0.43 -2.55 -0.67
endloop
endfacet
facet normal 0.3 0.82 0.5
outer loop
vertex -7.56 -0.03 -0.53
vertex -0.43 -2.55 -0.67
vertex -7.55 -0 -0.58
endloop
endfacet
facet normal 0.14 -0.87 0.48
outer loop
vertex -7.01 -0 3.12
vertex 6.99 -0 -0.93
vertex 7 0.03 -0.88
endloop
endfacet
facet normal 0.14 -0.87 0.48
outer loop
vertex -7.01 -0 3.12
vertex 7 0.03 -0.88
vertex -7 0.03 3.17
endloop
endfacet
facet normal -0.28 -0 -0.96
outer loop
vertex -7 0.03 3.17
vertex 7 0.03 -0.88
vertex 7 -0.03 -0.88
endloop
endfacet
facet normal -0.28 -0 -0.96
outer loop
vertex -7 0.03 3.17
vertex 7 -0.03 -0.88
vertex -7 -0.03 3.17
endloop
endfacet
facet normal 0.14 0.87 0.48
outer loop
vertex -7 -0.03 3.17
vertex 7 -0.03 -0.88
vertex 6.99 -0 -0.93
endloop
endfacet
facet normal 0.14 0.87 0.48
outer loop
vertex -7 -0.03 3.17
vertex 6.99 -0 -0.93
vertex -7.01 -0 3.12
endloop
endfacet
facet normal 0.02 -0.85 0.52
outer loop
vertex -7.54 -0 0.52
vertex -0.41 1.33 2.35
vertex -0.43 1.36 2.4
endloop
endfacet
facet normal 0.02 -0.85 0.52
outer loop
vertex -7.54 -0 0.52
vertex -0.43 1.36 2.4
vertex -7.56 0.03 0.56
endloop
endfacet
facet normal 0.25 -0 -0.97
outer loop
vertex -7.56 0.03 0.56
vertex -0.43 1.36 2.4
vertex -0.42 1.31 2.4
endloop
endfacet
facet normal 0.25 -0 -0.97
outer loop
vertex -7.56 0.03 0.56
vertex -0.42 1.31 2.4
vertex -7.55 -0.03 0.57
endloop
endfacet
facet normal -0.27 0.85 0.45
outer loop
vertex -7.55 -0.03 0.57
vertex -0.42 1.31 2.4
vertex -0.41 1.33 2.35
endloop
endfacet
facet normal -0.27 0.85 0.45
outer loop
vertex -7.55 -0.03 0.57
vertex -0.41 1.33 2.35
vertex -7.54 -0 0.52
endloop
endfacet
facet normal -0.14 -0.85 0.5
outer loop
vertex -7 -0 -0.58
vertex -2.1 -0.89 -0.72
vertex -2.09 -0.86 -0.67
endloop
endfacet
facet normal -0.14 -0.85 0.5
outer loop
vertex -7 -0 -0.58
vertex -2.09 -0.86 -0.67
vertex -6.99 0.03 -0.53
endloop
endfacet
facet normal -0.03 -0 -1
outer loop
vertex -6.99 0.03 -0.53
vertex -2.09 -0.86 -0.67
vertex -2.1 -0.92 -0.67
endloop
endfacet
facet normal -0.03 -0 -1
outer loop
vertex -6.99 0.03 -0.53
vertex -2.1 -0.92 -0.67
vertex -7 -0.03 -0.53
endloop
endfacet
facet normal 0.17 0.85 0.5
outer loop
vertex -7 -0.03 -0.53
vertex -2.1 -0.92 -0.67
vertex -2.1 -0.89 -0.72
endloop
endfacet
facet normal 0.17 0.85 0.5
outer loop
vertex -7 -0.03 -0.53
vertex -2.1 -0.89 -0.72
vertex -7 -0 -0.58
endloop
endfacet
facet normal -0.01 -0.87 0.5
outer loop
vertex -7.55 -0 0.52
vertex 6.1 -0 0.87
vertex 6.1 0.03 0.92
endloop
endfacet
facet normal -0.01 -0.87 0.5
outer loop
vertex -7.55 -0 0.52
vertex 6.1 0.03 0.92
vertex -7.55 0.03 0.57
endloop
endfacet
facet normal 0.03 -0 -1
outer loop
vertex -7.55 0.03 0.57
vertex 6.1 0.03 0.92
vertex 6.1 -0.03 0.92
endloop
endfacet
facet normal 0.03 -0 -1
outer loop
vertex -7.55 0.03 0.57
vertex 6.1 -0.03 0.92
vertex -7.55 -0.03 0.57
endloop
endfacet
facet normal -0.01 0.87 0.5
outer loop
vertex -7.55 -0.03 0.57
vertex 6.1 -0.03 0.92
vertex 6.1 -0 0.87
endloop
endfacet
facet normal -0.01 0.87 0.5
outer loop
vertex -7.55 -0.03 0.57
vertex 6.1 -0 0.87
vertex -7.55 -0 0.52
endloop
endfacet
facet normal 0.27 -0.81 0.51
outer loop
vertex -6.45 -0 -0.03
vertex 0.01 2.33 0.3
vertex 0 2.35 0.35
endloop
endfacet
facet normal 0.27 -0.81 0.51
outer loop
vertex -6.45 -0 -0.03
vertex 0 2.35 0.35
vertex -6.46 0.03 0.02
endloop
endfacet
facet normal 0.05 -0 -1
outer loop
vertex -6.46 0.03 0.02
vertex 0 2.35 0.35
vertex 0.02 2.3 0.35
endloop
endfacet
facet normal 0.05 -0 -1
outer loop
vertex -6.46 0.03 0.02
vertex 0.02 2.3 0.35
vertex -6.44 -0.03 0.02
endloop
endfacet
facet normal -0.32 0.81 0.48
outer loop
vertex -6.44 -0.03 0.02
vertex 0.02 2.3 0.35
vertex 0.01 2.33 0.3
endloop
endfacet
facet normal -0.32 0.81 0.48
outer loop
vertex -6.44 -0.03 0.02
vertex 0.01 2.33 0.3
vertex -6.45 -0 -0.03
endloop
endfacet
facet normal -0.67 -0.55 -0.49
outer loop
vertex -1.99 -2.39 1.17
vertex -3.49 -0.58 1.19
vertex -3.47 -0.56 1.15
endloop
endfacet
facet normal -0.67 -0.55 -0.49
outer loop
vertex -1.99 -2.39 1.17
vertex -3.47 -0.56 1.15
vertex -1.97 -2.37 1.12
endloop
endfacet
facet normal 0.01 -0 1
outer loop
vertex -1.97 -2.37 1.12
vertex -3.47 -0.56 1.15
vertex -3.51 -0.6 1.15
endloop
endfacet
facet normal 0.01 -0 1
outer loop
vertex -1.97 -2.37 1.12
vertex -3.51 -0.6 1.15
vertex -2.01 -2.41 1.12
endloop
endfacet
facet normal 0.66 0.55 -0.51
outer loop
vertex -2.01 -2.41 1.12
vertex -3.51 -0.6 1.15
vertex -3.49 -0.58 1.19
endloop
endfacet
facet normal 0.66 0.55 -0.51
outer loop
vertex -2.01 -2.41 1.12
vertex -3.49 -0.58 1.19
vertex -1.99 -2.39 1.17
endloop
endfacet
facet normal 0.52 -0.83 -0.21
outer loop
vertex -0.44 1.33 2.41
vertex -4.26 -0.21 -0.99
vertex -4.24 -0.19 -1.03
endloop
endfacet
facet normal 0.52 -0.83 -0.21
outer loop
vertex -0.44 1.33 2.41
vertex -4.24 -0.19 -1.03
vertex -0.42 1.36 2.37
endloop
endfacet
facet normal -0.66 -0 0.75
outer loop
vertex -0.42 1.36 2.37
vertex -4.24 -0.19 -1.03
vertex -4.22 -0.24 -1.02
endloop
endfacet
facet normal -0.66 -0 0.75
outer loop
vertex -0.42 1.36 2.37
vertex -4.22 -0.24 -1.02
vertex -0.41 1.31 2.38
endloop
endfacet
facet normal 0.14 0.83 -0.54
outer loop
vertex -0.41 1.31 2.38
vertex -4.22 -0.24 -1.02
vertex -4.26 -0.21 -0.99
endloop
endfacet
facet normal 0.14 0.83 -0.54
outer loop
vertex -0.41 1.31 2.38
vertex -4.26 -0.21 -0.99
vertex -0.44 1.33 2.41
endloop
endfacet
facet normal 0.85 -0.5 -0.15
outer loop
vertex -0.74 2.72 -1.81
vertex -2.86 -0.6 -2.8
vertex -2.86 -0.59 -2.85
endloop
endfacet
facet normal 0.85 -0.5 -0.15
outer loop
vertex -0.74 2.72 -1.81
vertex -2.86 -0.59 -2.85
vertex -0.74 2.73 -1.86
endloop
endfacet
facet normal -0.43 -0 0.91
outer loop
vertex -0.74 2.73 -1.86
vertex -2.86 -0.59 -2.85
vertex -2.82 -0.62 -2.83
endloop
endfacet
facet normal -0.43 -0 0.91
outer loop
vertex -0.74 2.73 -1.86
vertex -2.82 -0.62 -2.83
vertex -0.7 2.7 -1.84
endloop
endfacet
facet normal -0.43 0.5 -0.75
outer loop
vertex -0.7 2.7 -1.84
vertex -2.82 -0.62 -2.83
vertex -2.86 -0.6 -2.8
endloop
endfacet
facet normal -0.43 0.5 -0.75
outer loop
vertex -0.7 2.7 -1.84
vertex -2.86 -0.6 -2.8
vertex -0.74 2.72 -1.81
endloop
endfacet
facet normal 0.27 -0.38 -0.88
outer loop
vertex -2.11 3.06 0.04
vertex -3.47 -0.58 1.19
vertex -3.52 -0.57 1.17
endloop
endfacet
facet normal 0.27 -0.38 -0.88
outer loop
vertex -2.11 3.06 0.04
vertex -3.52 -0.57 1.17
vertex -2.15 3.07 0.02
endloop
endfacet
facet normal 0.64 -0 0.77
outer loop
vertex -2.15 3.07 0.02
vertex -3.52 -0.57 1.17
vertex -3.48 -0.59 1.13
endloop
endfacet
facet normal 0.64 -0 0.77
outer loop
vertex -2.15 3.07 0.02
vertex -3.48 -0.59 1.13
vertex -2.12 3.05 -0.01
endloop
endfacet
facet normal -0.92 0.38 0.12
outer loop
vertex -2.12 3.05 -0.01
vertex -3.48 -0.59 1.13
vertex -3.47 -0.58 1.19
endloop
endfacet
facet normal -0.92 0.38 0.12
outer loop
vertex -2.12 3.05 -0.01
vertex -3.47 -0.58 1.19
vertex -2.11 3.06 0.04
endloop
endfacet
facet normal -0.47 -0.7 0.55
outer loop
vertex -7 -0 -0.03
vertex -3.79 -2.39 -0.34
vertex -3.77 -2.37 -0.3
endloop
endfacet
facet normal -0.47 -0.7 0.55
outer loop
vertex -7 -0 -0.03
vertex -3.77 -2.37 -0.3
vertex -6.98 0.02 0.01
endloop
endfacet
facet normal -0.1 -0 -1
outer loop
vertex -6.98 0.02 0.01
vertex -3.77 -2.37 -0.3
vertex -3.81 -2.42 -0.29
endloop
endfacet
facet normal -0.1 -0 -1
outer loop
vertex -6.98 0.02 0.01
vertex -3.81 -2.42 -0.29
vertex -7.01 -0.02 0.02
endloop
endfacet
facet normal 0.56 0.7 0.45
outer loop
vertex -7.01 -0.02 0.02
vertex -3.81 -2.42 -0.29
vertex -3.79 -2.39 -0.34
endloop
endfacet
facet normal 0.56 0.7 0.45
outer loop
vertex -7.01 -0.02 0.02
vertex -3.79 -2.39 -0.34
vertex -7 -0 -0.03
endloop
endfacet
facet normal -0.38 -0.82 0.43
outer loop
vertex -7.54 -0 -2.08
vertex -0.42 -2.55 -0.67
vertex -0.42 -2.53 -0.62
endloop
endfacet
facet normal -0.38 -0.82 0.43
outer loop
vertex -7.54 -0 -2.08
vertex -0.42 -2.53 -0.62
vertex -7.54 0.03 -2.03
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -7.54 0.03 -2.03
vertex -0.42 -2.53 -0.62
vertex -0.44 -2.58 -0.63
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -7.54 0.03 -2.03
vertex -0.44 -2.58 -0.63
vertex -7.56 -0.03 -2.04
endloop
endfacet
facet normal 0.18 0.82 0.55
outer loop
vertex -7.56 -0.03 -2.04
vertex -0.44 -2.58 -0.63
vertex -0.42 -2.55 -0.67
endloop
endfacet
facet normal 0.18 0.82 0.55
outer loop
vertex -7.56 -0.03 -2.04
vertex -0.42 -2.55 -0.67
vertex -7.54 -0 -2.08
endloop
endfacet
facet normal 0.04 -0.87 0.5
outer loop
vertex -7.55 -0 2.02
vertex 7.9 -0 0.87
vertex 7.9 0.03 0.92
endloop
endfacet
facet normal 0.04 -0.87 0.5
outer loop
vertex -7.55 -0 2.02
vertex 7.9 0.03 0.92
vertex -7.55 0.03 2.07
endloop
endfacet
facet normal -0.07 -0 -1
outer loop
vertex -7.55 0.03 2.07
vertex 7.9 0.03 0.92
vertex 7.9 -0.03 0.92
endloop
endfacet
facet normal -0.07 -0 -1
outer loop
vertex -7.55 0.03 2.07
vertex 7.9 -0.03 0.92
vertex -7.55 -0.03 2.07
endloop
endfacet
facet normal 0.04 0.87 0.5
outer loop
vertex -7.55 -0.03 2.07
vertex 7.9 -0.03 0.92
vertex 7.9 -0 0.87
endloop
endfacet
facet normal 0.04 0.87 0.5
outer loop
vertex -7.55 -0.03 2.07
vertex 7.9 -0 0.87
vertex -7.55 -0 2.02
endloop
endfacet
facet normal 0.71 -0.6 0.36
outer loop
vertex -0.43 -2.55 -0.67
vertex 1.14 -0.89 -1
vertex 1.13 -0.87 -0.95
endloop
endfacet
facet normal 0.71 -0.6 0.36
outer loop
vertex -0.43 -2.55 -0.67
vertex 1.13 -0.87 -0.95
vertex -0.44 -2.53 -0.62
endloop
endfacet
facet normal -0.21 -0 -0.98
outer loop
vertex -0.44 -2.53 -0.62
vertex 1.13 -0.87 -0.95
vertex 1.17 -0.9 -0.96
endloop
endfacet
facet normal -0.21 -0 -0.98
outer loop
vertex -0.44 -2.53 -0.62
vertex 1.17 -0.9 -0.96
vertex -0.4 -2.57 -0.63
endloop
endfacet
facet normal -0.51 0.6 0.62
outer loop
vertex -0.4 -2.57 -0.63
vertex 1.17 -0.9 -0.96
vertex 1.14 -0.89 -1
endloop
endfacet
facet normal -0.51 0.6 0.62
outer loop
vertex -0.4 -2.57 -0.63
vertex 1.14 -0.89 -1
vertex -0.43 -2.55 -0.67
endloop
endfacet
facet normal -0.15 -0.83 -0.54
outer loop
vertex 7.89 -0 0.03
vertex -0.73 2.72 -1.8
vertex -0.72 2.74 -1.85
endloop
endfacet
facet normal -0.15 -0.83 -0.54
outer loop
vertex 7.89 -0 0.03
vertex -0.72 2.74 -1.85
vertex 7.91 0.03 -0.01
endloop
endfacet
facet normal -0.21 -0 0.98
outer loop
vertex 7.91 0.03 -0.01
vertex -0.72 2.74 -1.85
vertex -0.73 2.69 -1.85
endloop
endfacet
facet normal -0.21 -0 0.98
outer loop
vertex 7.91 0.03 -0.01
vertex -0.73 2.69 -1.85
vertex 7.9 -0.03 -0.02
endloop
endfacet
facet normal 0.35 0.83 -0.44
outer loop
vertex 7.9 -0.03 -0.02
vertex -0.73 2.69 -1.85
vertex -0.73 2.72 -1.8
endloop
endfacet
facet normal 0.35 0.83 -0.44
outer loop
vertex 7.9 -0.03 -0.02
vertex -0.73 2.72 -1.8
vertex 7.89 -0 0.03
endloop
endfacet
facet normal 0.63 -0.72 -0.27
outer loop
vertex -3.1 1.3 2.53
vertex -2.15 3.06 0
vertex -2.12 3.09 0.03
endloop
endfacet
facet normal 0.63 -0.72 -0.27
outer loop
vertex -3.1 1.3 2.53
vertex -2.12 3.09 0.03
vertex -3.06 1.32 2.56
endloop
endfacet
facet normal -0.94 -0 -0.35
outer loop
vertex -3.06 1.32 2.56
vertex -2.12 3.09 0.03
vertex -2.11 3.04 0
endloop
endfacet
facet normal -0.94 -0 -0.35
outer loop
vertex -3.06 1.32 2.56
vertex -2.11 3.04 0
vertex -3.05 1.28 2.53
endloop
endfacet
facet normal 0.3 0.73 0.62
outer loop
vertex -3.05 1.28 2.53
vertex -2.11 3.04 0
vertex -2.15 3.06 0
endloop
endfacet
facet normal 0.3 0.73 0.62
outer loop
vertex -3.05 1.28 2.53
vertex -2.15 3.06 0
vertex -3.1 1.3 2.53
endloop
endfacet
facet normal 0.24 -0.74 0.62
outer loop
vertex -6.44 -0 -0.58
vertex -3.9 1.61 0.35
vertex -3.93 1.63 0.39
endloop
endfacet
facet normal 0.24 -0.74 0.62
outer loop
vertex -6.44 -0 -0.58
vertex -3.93 1.63 0.39
vertex -6.47 0.02 -0.54
endloop
endfacet
facet normal 0.34 -0 -0.94
outer loop
vertex -6.47 0.02 -0.54
vertex -3.93 1.63 0.39
vertex -3.9 1.58 0.4
endloop
endfacet
facet normal 0.34 -0 -0.94
outer loop
vertex -6.47 0.02 -0.54
vertex -3.9 1.58 0.4
vertex -6.44 -0.02 -0.53
endloop
endfacet
facet normal -0.59 0.74 0.32
outer loop
vertex -6.44 -0.02 -0.53
vertex -3.9 1.58 0.4
vertex -3.9 1.61 0.35
endloop
endfacet
facet normal -0.59 0.74 0.32
outer loop
vertex -6.44 -0.02 -0.53
vertex -3.9 1.61 0.35
vertex -6.44 -0 -0.58
endloop
endfacet
facet normal -0.14 -0.86 0.49
outer loop
vertex -6.45 -0 2.02
vertex 0.58 -0.77 2.71
vertex 0.58 -0.75 2.75
endloop
endfacet
facet normal -0.14 -0.86 0.49
outer loop
vertex -6.45 -0 2.02
vertex 0.58 -0.75 2.75
vertex -6.45 0.03 2.07
endloop
endfacet
facet normal 0.1 -0 -1
outer loop
vertex -6.45 0.03 2.07
vertex 0.58 -0.75 2.75
vertex 0.57 -0.8 2.75
endloop
endfacet
facet normal 0.1 -0 -1
outer loop
vertex -6.45 0.03 2.07
vertex 0.57 -0.8 2.75
vertex -6.45 -0.03 2.07
endloop
endfacet
facet normal 0.04 0.86 0.51
outer loop
vertex -6.45 -0.03 2.07
vertex 0.57 -0.8 2.75
vertex 0.58 -0.77 2.71
endloop
endfacet
facet normal 0.04 0.86 0.51
outer loop
vertex -6.45 -0.03 2.07
vertex 0.58 -0.77 2.71
vertex -6.45 -0 2.02
endloop
endfacet
facet normal -0.03 -0.81 0.59
outer loop
vertex -4.22 -0.21 -1.04
vertex -1.77 0.93 0.67
vertex -1.8 0.95 0.7
endloop
endfacet
facet normal -0.03 -0.81 0.59
outer loop
vertex -4.22 -0.21 -1.04
vertex -1.8 0.95 0.7
vertex -4.26 -0.19 -1.01
endloop
endfacet
facet normal 0.57 -0 -0.82
outer loop
vertex -4.26 -0.19 -1.01
vertex -1.8 0.95 0.7
vertex -1.79 0.9 0.71
endloop
endfacet
facet normal 0.57 -0 -0.82
outer loop
vertex -4.26 -0.19 -1.01
vertex -1.79 0.9 0.71
vertex -4.24 -0.24 -0.99
endloop
endfacet
facet normal -0.54 0.81 0.23
outer loop
vertex -4.24 -0.24 -0.99
vertex -1.79 0.9 0.71
vertex -1.77 0.93 0.67
endloop
endfacet
facet normal -0.54 0.81 0.23
outer loop
vertex -4.24 -0.24 -0.99
vertex -1.77 0.93 0.67
vertex -4.22 -0.21 -1.04
endloop
endfacet
facet normal 0.41 -0.86 0.3
outer loop
vertex -7.02 -0 2.03
vertex -1.21 0.93 -3.27
vertex -1.18 0.96 -3.24
endloop
endfacet
facet normal 0.41 -0.86 0.3
outer loop
vertex -7.02 -0 2.03
vertex -1.18 0.96 -3.24
vertex -6.99 0.03 2.06
endloop
endfacet
facet normal -0.67 -0 -0.74
outer loop
vertex -6.99 0.03 2.06
vertex -1.18 0.96 -3.24
vertex -1.17 0.91 -3.24
endloop
endfacet
facet normal -0.67 -0 -0.74
outer loop
vertex -6.99 0.03 2.06
vertex -1.17 0.91 -3.24
vertex -6.99 -0.03 2.06
endloop
endfacet
facet normal 0.26 0.86 0.44
outer loop
vertex -6.99 -0.03 2.06
vertex -1.17 0.91 -3.24
vertex -1.21 0.93 -3.27
endloop
endfacet
facet normal 0.26 0.86 0.44
outer loop
vertex -6.99 -0.03 2.06
vertex -1.21 0.93 -3.27
vertex -7.02 -0 2.03
endloop
endfacet
facet normal 0.43 -0.82 0.39
outer loop
vertex 1.19 -2.17 0.93
vertex 6.99 -0 -0.93
vertex 7 0.03 -0.88
endloop
endfacet
facet normal 0.43 -0.82 0.39
outer loop
vertex 1.19 -2.17 0.93
vertex 7 0.03 -0.88
vertex 1.2 -2.15 0.98
endloop
endfacet
facet normal -0.31 -0 -0.95
outer loop
vertex 1.2 -2.15 0.98
vertex 7 0.03 -0.88
vertex 7.01 -0.03 -0.89
endloop
endfacet
facet normal -0.31 -0 -0.95
outer loop
vertex 1.2 -2.15 0.98
vertex 7.01 -0.03 -0.89
vertex 1.21 -2.2 0.98
endloop
endfacet
facet normal -0.12 0.82 0.57
outer loop
vertex 1.21 -2.2 0.98
vertex 7.01 -0.03 -0.89
vertex 6.99 -0 -0.93
endloop
endfacet
facet normal -0.12 0.82 0.57
outer loop
vertex 1.21 -2.2 0.98
vertex 6.99 -0 -0.93
vertex 1.19 -2.17 0.93
endloop
endfacet
facet normal 0.01 -0.86 -0.51
outer loop
vertex 6.99 -0 0.93
vertex -2.6 1.31 -1.45
vertex -2.59 1.34 -1.49
endloop
endfacet
facet normal 0.01 -0.86 -0.51
outer loop
vertex 6.99 -0 0.93
vertex -2.59 1.34 -1.49
vertex 7.01 0.03 0.89
endloop
endfacet
facet normal -0.24 -0 0.97
outer loop
vertex 7.01 0.03 0.89
vertex -2.59 1.34 -1.49
vertex -2.6 1.28 -1.49
endloop
endfacet
facet normal -0.24 -0 0.97
outer loop
vertex 7.01 0.03 0.89
vertex -2.6 1.28 -1.49
vertex 7 -0.03 0.88
endloop
endfacet
facet normal 0.23 0.86 -0.46
outer loop
vertex 7 -0.03 0.88
vertex -2.6 1.28 -1.49
vertex -2.6 1.31 -1.45
endloop
endfacet
facet normal 0.23 0.86 -0.46
outer loop
vertex 7 -0.03 0.88
vertex -2.6 1.31 -1.45
vertex 6.99 -0 0.93
endloop
endfacet
facet normal 0.35 -0.86 0.38
outer loop
vertex -1.8 0.93 0.67
vertex 1.55 1.43 -1.25
vertex 1.57 1.46 -1.21
endloop
endfacet
facet normal 0.35 -0.86 0.38
outer loop
vertex -1.8 0.93 0.67
vertex 1.57 1.46 -1.21
vertex -1.78 0.95 0.71
endloop
endfacet
facet normal -0.5 -0 -0.87
outer loop
vertex -1.78 0.95 0.71
vertex 1.57 1.46 -1.21
vertex 1.58 1.4 -1.21
endloop
endfacet
facet normal -0.5 -0 -0.87
outer loop
vertex -1.78 0.95 0.71
vertex 1.58 1.4 -1.21
vertex -1.78 0.9 0.71
endloop
endfacet
facet normal 0.15 0.86 0.49
outer loop
vertex -1.78 0.9 0.71
vertex 1.58 1.4 -1.21
vertex 1.55 1.43 -1.25
endloop
endfacet
facet normal 0.15 0.86 0.49
outer loop
vertex -1.78 0.9 0.71
vertex 1.55 1.43 -1.25
vertex -1.8 0.93 0.67
endloop
endfacet
facet normal 0.04 -0.87 0.5
outer loop
vertex -7 -0 2.02
vertex 7 -0 0.87
vertex 7 0.03 0.92
endloop
endfacet
facet normal 0.04 -0.87 0.5
outer loop
vertex -7 -0 2.02
vertex 7 0.03 0.92
vertex -7 0.03 2.07
endloop
endfacet
facet normal -0.08 -0 -1
outer loop
vertex -7 0.03 2.07
vertex 7 0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal -0.08 -0 -1
outer loop
vertex -7 0.03 2.07
vertex 7 -0.03 0.92
vertex -7 -0.03 2.07
endloop
endfacet
facet normal 0.04 0.87 0.5
outer loop
vertex -7 -0.03 2.07
vertex 7 -0.03 0.92
vertex 7 -0 0.87
endloop
endfacet
facet normal 0.04 0.87 0.5
outer loop
vertex -7 -0.03 2.07
vertex 7 -0 0.87
vertex -7 -0 2.02
endloop
endfacet
facet normal -0.3 -0.86 0.4
outer loop
vertex -3.9 1.61 0.35
vertex -0.41 1.33 2.36
vertex -0.43 1.36 2.4
endloop
endfacet
facet normal -0.3 -0.86 0.4
outer loop
vertex -3.9 1.61 0.35
vertex -0.43 1.36 2.4
vertex -3.92 1.63 0.4
endloop
endfacet
facet normal 0.5 -0 -0.87
outer loop
vertex -3.92 1.63 0.4
vertex -0.43 1.36 2.4
vertex -0.43 1.31 2.4
endloop
endfacet
facet normal 0.5 -0 -0.87
outer loop
vertex -3.92 1.63 0.4
vertex -0.43 1.31 2.4
vertex -3.92 1.58 0.39
endloop
endfacet
facet normal -0.2 0.86 0.46
outer loop
vertex -3.92 1.58 0.39
vertex -0.43 1.31 2.4
vertex -0.41 1.33 2.36
endloop
endfacet
facet normal -0.2 0.86 0.46
outer loop
vertex -3.92 1.58 0.39
vertex -0.41 1.33 2.36
vertex -3.9 1.61 0.35
endloop
endfacet
facet normal -0.07 -0.87 0.5
outer loop
vertex -4.24 -0.21 -1.04
vertex 6.11 -0 0.87
vertex 6.1 0.03 0.92
endloop
endfacet
facet normal -0.07 -0.87 0.5
outer loop
vertex -4.24 -0.21 -1.04
vertex 6.1 0.03 0.92
vertex -4.24 -0.19 -1
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -4.24 -0.19 -1
vertex 6.1 0.03 0.92
vertex 6.1 -0.03 0.92
endloop
endfacet
facet normal 0.18 -0 -0.98
outer loop
vertex -4.24 -0.19 -1
vertex 6.1 -0.03 0.92
vertex -4.24 -0.24 -1
endloop
endfacet
facet normal -0.11 0.87 0.49
outer loop
vertex -4.24 -0.24 -1
vertex 6.1 -0.03 0.92
vertex 6.11 -0 0.87
endloop
endfacet
facet normal -0.11 0.87 0.49
outer loop
vertex -4.24 -0.24 -1
vertex 6.11 -0 0.87
vertex -4.24 -0.21 -1.04
endloop
endfacet
facet normal 0.61 -0.79 0.06
outer loop
vertex 1.28 0.54 0.94
vertex -0.64 -1.19 -2.51
vertex -0.6 -1.16 -2.54
endloop
endfacet
facet normal 0.61 -0.79 0.06
outer loop
vertex 1.28 0.54 0.94
vertex -0.6 -1.16 -2.54
vertex 1.32 0.56 0.91
endloop
endfacet
facet normal -0.87 -0 0.49
outer loop
vertex 1.32 0.56 0.91
vertex -0.6 -1.16 -2.54
vertex -0.59 -1.21 -2.53
endloop
endfacet
facet normal -0.87 -0 0.49
outer loop
vertex 1.32 0.56 0.91
vertex -0.59 -1.21 -2.53
vertex 1.33 0.51 0.93
endloop
endfacet
facet normal 0.27 0.79 -0.55
outer loop
vertex 1.33 0.51 0.93
vertex -0.59 -1.21 -2.53
vertex -0.64 -1.19 -2.51
endloop
endfacet
facet normal 0.27 0.79 -0.55
outer loop
vertex 1.33 0.51 0.93
vertex -0.64 -1.19 -2.51
vertex 1.28 0.54 0.94
endloop
endfacet
facet normal -0.04 -0.86 0.5
outer loop
vertex -6.99 -0 -2.63
vertex -0.49 0.59 -1.13
vertex -0.51 0.62 -1.08
endloop
endfacet
facet normal -0.04 -0.86 0.5
outer loop
vertex -6.99 -0 -2.63
vertex -0.51 0.62 -1.08
vertex -7.01 0.03 -2.59
endloop
endfacet
facet normal 0.23 -0 -0.97
outer loop
vertex -7.01 0.03 -2.59
vertex -0.51 0.62 -1.08
vertex -0.5 0.57 -1.08
endloop
endfacet
facet normal 0.23 -0 -0.97
outer loop
vertex -7.01 0.03 -2.59
vertex -0.5 0.57 -1.08
vertex -7 -0.03 -2.58
endloop
endfacet
facet normal -0.19 0.86 0.47
outer loop
vertex -7 -0.03 -2.58
vertex -0.5 0.57 -1.08
vertex -0.49 0.59 -1.13
endloop
endfacet
facet normal -0.19 0.86 0.47
outer loop
vertex -7 -0.03 -2.58
vertex -0.49 0.59 -1.13
vertex -6.99 -0 -2.63
endloop
endfacet
facet normal -0.09 -0.86 0.5
outer loop
vertex 1.31 0.54 0.89
vertex 6.1 -0 0.87
vertex 6.1 0.03 0.92
endloop
endfacet
facet normal -0.09 -0.86 0.5
outer loop
vertex 1.31 0.54 0.89
vertex 6.1 0.03 0.92
vertex 1.31 0.57 0.94
endloop
endfacet
facet normal -0 -0 -1
outer loop
vertex 1.31 0.57 0.94
vertex 6.1 0.03 0.92
vertex 6.1 -0.03 0.92
endloop
endfacet
facet normal -0 -0 -1
outer loop
vertex 1.31 0.57 0.94
vertex 6.1 -0.03 0.92
vertex 1.3 0.51 0.94
endloop
endfacet
facet normal 0.1 0.86 0.5
outer loop
vertex 1.3 0.51 0.94
vertex 6.1 -0.03 0.92
vertex 6.1 -0 0.87
endloop
endfacet
facet normal 0.1 0.86 0.5
outer loop
vertex 1.3 0.51 0.94
vertex 6.1 -0 0.87
vertex 1.31 0.54 0.89
endloop
endfacet
facet normal -0.1 -0.86 -0.51
outer loop
vertex 6.1 -0 -0.87
vertex -2.6 1.31 -1.45
vertex -2.59 1.34 -1.49
endloop
endfacet
facet normal -0.1 -0.86 -0.51
outer loop
vertex 6.1 -0 -0.87
vertex -2.59 1.34 -1.49
vertex 6.11 0.03 -0.92
endloop
endfacet
facet normal -0.07 -0 1
outer loop
vertex 6.11 0.03 -0.92
vertex -2.59 1.34 -1.49
vertex -2.6 1.28 -1.49
endloop
endfacet
facet normal -0.07 -0 1
outer loop
vertex 6.11 0.03 -0.92
vertex -2.6 1.28 -1.49
vertex 6.1 -0.03 -0.92
endloop
endfacet
facet normal 0.16 0.86 -0.49
outer loop
vertex 6.1 -0.03 -0.92
vertex -2.6 1.28 -1.49
vertex -2.6 1.31 -1.45
endloop
endfacet
facet normal 0.16 0.86 -0.49
outer loop
vertex 6.1 -0.03 -0.92
vertex -2.6 1.31 -1.45
vertex 6.1 -0 -0.87
endloop
endfacet
facet normal 0.74 -0.67 -0.06
outer loop
vertex -3.52 -0.58 1.14
vertex -0.75 2.72 -1.85
vertex -0.73 2.74 -1.81
endloop
endfacet
facet normal 0.74 -0.67 -0.06
outer loop
vertex -3.52 -0.58 1.14
vertex -0.73 2.74 -1.81
vertex -3.49 -0.56 1.18
endloop
endfacet
facet normal -0.73 -0 -0.68
outer loop
vertex -3.49 -0.56 1.18
vertex -0.73 2.74 -1.81
vertex -0.7 2.7 -1.84
endloop
endfacet
facet normal -0.73 -0 -0.68
outer loop
vertex -3.49 -0.56 1.18
vertex -0.7 2.7 -1.84
vertex -3.47 -0.6 1.16
endloop
endfacet
facet normal -0 0.67 0.74
outer loop
vertex -3.47 -0.6 1.16
vertex -0.7 2.7 -1.84
vertex -0.75 2.72 -1.85
endloop
endfacet
facet normal -0 0.67 0.74
outer loop
vertex -3.47 -0.6 1.16
vertex -0.75 2.72 -1.85
vertex -3.52 -0.58 1.14
endloop
endfacet
facet normal -0.01 -0.87 0.5
outer loop
vertex -7 -0 0.52
vertex 7 -0 0.87
vertex 7 0.03 0.92
endloop
endfacet
facet normal -0.01 -0.87 0.5
outer loop
vertex -7 -0 0.52
vertex 7 0.03 0.92
vertex -7 0.03 0.57
endloop
endfacet
facet normal 0.02 -0 -1
outer loop
vertex -7 0.03 0.57
vertex 7 0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal 0.02 -0 -1
outer loop
vertex -7 0.03 0.57
vertex 7 -0.03 0.92
vertex -7 -0.03 0.57
endloop
endfacet
facet normal -0.01 0.87 0.5
outer loop
vertex -7 -0.03 0.57
vertex 7 -0.03 0.92
vertex 7 -0 0.87
endloop
endfacet
facet normal -0.01 0.87 0.5
outer loop
vertex -7 -0.03 0.57
vertex 7 -0 0.87
vertex -7 -0 0.52
endloop
endfacet
facet normal -0.35 -0.86 0.36
outer loop
vertex -6.98 -0 -3.17
vertex -4.22 -0.21 -1.04
vertex -4.25 -0.19 -1
endloop
endfacet
facet normal -0.35 -0.86 0.36
outer loop
vertex -6.98 -0 -3.17
vertex -4.25 -0.19 -1
vertex -7.01 0.03 -3.14
endloop
endfacet
facet normal 0.61 -0 -0.79
outer loop
vertex -7.01 0.03 -3.14
vertex -4.25 -0.19 -1
vertex -4.25 -0.24 -1
endloop
endfacet
facet normal 0.61 -0 -0.79
outer loop
vertex -7.01 0.03 -3.14
vertex -4.25 -0.24 -1
vertex -7.01 -0.03 -3.14
endloop
endfacet
facet normal -0.26 0.86 0.43
outer loop
vertex -7.01 -0.03 -3.14
vertex -4.25 -0.24 -1
vertex -4.22 -0.21 -1.04
endloop
endfacet
facet normal -0.26 0.86 0.43
outer loop
vertex -7.01 -0.03 -3.14
vertex -4.22 -0.21 -1.04
vertex -6.98 -0 -3.17
endloop
endfacet
facet normal -0.42 -0.73 0.54
outer loop
vertex -7.55 -0 -0.03
vertex -3.79 -2.39 -0.34
vertex -3.78 -2.37 -0.29
endloop
endfacet
facet normal -0.42 -0.73 0.54
outer loop
vertex -7.55 -0 -0.03
vertex -3.78 -2.37 -0.29
vertex -7.53 0.02 0.01
endloop
endfacet
facet normal -0.08 -0 -1
outer loop
vertex -7.53 0.02 0.01
vertex -3.78 -2.37 -0.29
vertex -3.8 -2.42 -0.29
endloop
endfacet
facet normal -0.08 -0 -1
outer loop
vertex -7.53 0.02 0.01
vertex -3.8 -2.42 -0.29
vertex -7.56 -0.02 0.02
endloop
endfacet
facet normal 0.5 0.73 0.46
outer loop
vertex -7.56 -0.02 0.02
vertex -3.8 -2.42 -0.29
vertex -3.79 -2.39 -0.34
endloop
endfacet
facet normal 0.5 0.73 0.46
outer loop
vertex -7.56 -0.02 0.02
vertex -3.79 -2.39 -0.34
vertex -7.55 -0 -0.03
endloop
endfacet
facet normal -0.05 -0.86 0.5
outer loop
vertex -1.79 0.93 0.66
vertex 7.9 -0 -0.03
vertex 7.9 0.03 0.02
endloop
endfacet
facet normal -0.05 -0.86 0.5
outer loop
vertex -1.79 0.93 0.66
vertex 7.9 0.03 0.02
vertex -1.78 0.95 0.71
endloop
endfacet
facet normal -0.07 -0 -1
outer loop
vertex -1.78 0.95 0.71
vertex 7.9 0.03 0.02
vertex 7.9 -0.03 0.02
endloop
endfacet
facet normal -0.07 -0 -1
outer loop
vertex -1.78 0.95 0.71
vertex 7.9 -0.03 0.02
vertex -1.79 0.9 0.71
endloop
endfacet
facet normal 0.12 0.86 0.49
outer loop
vertex -1.79 0.9 0.71
vertex 7.9 -0.03 0.02
vertex 7.9 -0 -0.03
endloop
endfacet
facet normal 0.12 0.86 0.49
outer loop
vertex -1.79 0.9 0.71
vertex 7.9 -0 -0.03
vertex -1.79 0.93 0.66
endloop
endfacet
facet normal 0.43 -0.5 0.75
outer loop
vertex -2.84 -0.6 -2.86
vertex -0.71 2.72 -1.86
vertex -0.75 2.73 -1.83
endloop
endfacet
facet normal 0.43 -0.5 0.75
outer loop
vertex -2.84 -0.6 -2.86
vertex -0.75 2.73 -1.83
vertex -2.87 -0.59 -2.83
endloop
endfacet
facet normal 0.43 -0 -0.91
outer loop
vertex -2.87 -0.59 -2.83
vertex -0.75 2.73 -1.83
vertex -0.71 2.7 -1.81
endloop
endfacet
facet normal 0.43 -0 -0.91
outer loop
vertex -2.87 -0.59 -2.83
vertex -0.71 2.7 -1.81
vertex -2.83 -0.62 -2.81
endloop
endfacet
facet normal -0.85 0.5 0.15
outer loop
vertex -2.83 -0.62 -2.81
vertex -0.71 2.7 -1.81
vertex -0.71 2.72 -1.86
endloop
endfacet
facet normal -0.85 0.5 0.15
outer loop
vertex -2.83 -0.62 -2.81
vertex -0.71 2.72 -1.86
vertex -2.84 -0.6 -2.86
endloop
endfacet
facet normal -0.33 -0.83 0.46
outer loop
vertex -4.24 -0.21 -1.04
vertex -2.09 -0.89 -0.72
vertex -2.09 -0.87 -0.67
endloop
endfacet
facet normal -0.33 -0.83 0.46
outer loop
vertex -4.24 -0.21 -1.04
vertex -2.09 -0.87 -0.67
vertex -4.24 -0.19 -1
endloop
endfacet
facet normal 0.15 -0 -0.99
outer loop
vertex -4.24 -0.19 -1
vertex -2.09 -0.87 -0.67
vertex -2.11 -0.92 -0.67
endloop
endfacet
facet normal 0.15 -0 -0.99
outer loop
vertex -4.24 -0.19 -1
vertex -2.11 -0.92 -0.67
vertex -4.25 -0.24 -1
endloop
endfacet
facet normal 0.18 0.83 0.53
outer loop
vertex -4.25 -0.24 -1
vertex -2.11 -0.92 -0.67
vertex -2.09 -0.89 -0.72
endloop
endfacet
facet normal 0.18 0.83 0.53
outer loop
vertex -4.25 -0.24 -1
vertex -2.09 -0.89 -0.72
vertex -4.24 -0.21 -1.04
endloop
endfacet
facet normal -0.08 -0.83 0.55
outer loop
vertex -6.98 -0 -0.03
vertex -3.06 1.3 2.51
vertex -3.09 1.33 2.55
endloop
endfacet
facet normal -0.08 -0.83 0.55
outer loop
vertex -6.98 -0 -0.03
vertex -3.09 1.33 2.55
vertex -7.01 0.03 0.01
endloop
endfacet
facet normal 0.54 -0 -0.84
outer loop
vertex -7.01 0.03 0.01
vertex -3.09 1.33 2.55
vertex -3.08 1.27 2.56
endloop
endfacet
facet normal 0.54 -0 -0.84
outer loop
vertex -7.01 0.03 0.01
vertex -3.08 1.27 2.56
vertex -7 -0.03 0.02
endloop
endfacet
facet normal -0.47 0.83 0.29
outer loop
vertex -7 -0.03 0.02
vertex -3.08 1.27 2.56
vertex -3.06 1.3 2.51
endloop
endfacet
facet normal -0.47 0.83 0.29
outer loop
vertex -7 -0.03 0.02
vertex -3.06 1.3 2.51
vertex -6.98 -0 -0.03
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -6.99 -0 -2.63
vertex 7.91 -0 -0.03
vertex 7.9 0.03 0.02
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -6.99 -0 -2.63
vertex 7.9 0.03 0.02
vertex -7 0.03 -2.58
endloop
endfacet
facet normal 0.17 -0 -0.99
outer loop
vertex -7 0.03 -2.58
vertex 7.9 0.03 0.02
vertex 7.9 -0.03 0.02
endloop
endfacet
facet normal 0.17 -0 -0.99
outer loop
vertex -7 0.03 -2.58
vertex 7.9 -0.03 0.02
vertex -7 -0.03 -2.58
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7 -0.03 -2.58
vertex 7.9 -0.03 0.02
vertex 7.91 -0 -0.03
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7 -0.03 -2.58
vertex 7.91 -0 -0.03
vertex -6.99 -0 -2.63
endloop
endfacet
facet normal -0.2 -0.86 -0.48
outer loop
vertex 6.1 -0 -0.87
vertex -3.91 1.61 0.41
vertex -3.91 1.63 0.36
endloop
endfacet
facet normal -0.2 -0.86 -0.48
outer loop
vertex 6.1 -0 -0.87
vertex -3.91 1.63 0.36
vertex 6.1 0.03 -0.92
endloop
endfacet
facet normal 0.13 -0 0.99
outer loop
vertex 6.1 0.03 -0.92
vertex -3.91 1.63 0.36
vertex -3.92 1.58 0.37
endloop
endfacet
facet normal 0.13 -0 0.99
outer loop
vertex 6.1 0.03 -0.92
vertex -3.92 1.58 0.37
vertex 6.09 -0.03 -0.91
endloop
endfacet
facet normal 0.07 0.86 -0.51
outer loop
vertex 6.09 -0.03 -0.91
vertex -3.92 1.58 0.37
vertex -3.91 1.61 0.41
endloop
endfacet
facet normal 0.07 0.86 -0.51
outer loop
vertex 6.09 -0.03 -0.91
vertex -3.91 1.61 0.41
vertex 6.1 -0 -0.87
endloop
endfacet
facet normal -0.22 -0.82 0.52
outer loop
vertex -2.13 3.06 -0.02
vertex 7 -0 -0.93
vertex 7.01 0.03 -0.89
endloop
endfacet
facet normal -0.22 -0.82 0.52
outer loop
vertex -2.13 3.06 -0.02
vertex 7.01 0.03 -0.89
vertex -2.12 3.09 0.03
endloop
endfacet
facet normal -0.1 -0 -1
outer loop
vertex -2.12 3.09 0.03
vertex 7.01 0.03 -0.89
vertex 6.99 -0.03 -0.88
endloop
endfacet
facet normal -0.1 -0 -1
outer loop
vertex -2.12 3.09 0.03
vertex 6.99 -0.03 -0.88
vertex -2.13 3.04 0.03
endloop
endfacet
facet normal 0.32 0.82 0.47
outer loop
vertex -2.13 3.04 0.03
vertex 6.99 -0.03 -0.88
vertex 7 -0 -0.93
endloop
endfacet
facet normal 0.32 0.82 0.47
outer loop
vertex -2.13 3.04 0.03
vertex 7 -0 -0.93
vertex -2.13 3.06 -0.02
endloop
endfacet
facet normal -0.43 -0.86 0.26
outer loop
vertex -6.68 -0 -4.42
vertex -4.22 -0.21 -1.03
vertex -4.25 -0.19 -1
endloop
endfacet
facet normal -0.43 -0.86 0.26
outer loop
vertex -6.68 -0 -4.42
vertex -4.25 -0.19 -1
vertex -6.71 0.03 -4.39
endloop
endfacet
facet normal 0.81 -0 -0.59
outer loop
vertex -6.71 0.03 -4.39
vertex -4.25 -0.19 -1
vertex -4.25 -0.24 -1
endloop
endfacet
facet normal 0.81 -0 -0.59
outer loop
vertex -6.71 0.03 -4.39
vertex -4.25 -0.24 -1
vertex -6.71 -0.03 -4.39
endloop
endfacet
facet normal -0.38 0.86 0.33
outer loop
vertex -6.71 -0.03 -4.39
vertex -4.25 -0.24 -1
vertex -4.22 -0.21 -1.03
endloop
endfacet
facet normal -0.38 0.86 0.33
outer loop
vertex -6.71 -0.03 -4.39
vertex -4.22 -0.21 -1.03
vertex -6.68 -0 -4.42
endloop
endfacet
facet normal -0.33 -0.85 -0.41
outer loop
vertex 6.11 -0 0.03
vertex -0.41 1.33 2.41
vertex -0.42 1.36 2.37
endloop
endfacet
facet normal -0.33 -0.85 -0.41
outer loop
vertex 6.11 -0 0.03
vertex -0.42 1.36 2.37
vertex 6.1 0.03 -0.02
endloop
endfacet
facet normal 0.34 -0 0.94
outer loop
vertex 6.1 0.03 -0.02
vertex -0.42 1.36 2.37
vertex -0.43 1.31 2.37
endloop
endfacet
facet normal 0.34 -0 0.94
outer loop
vertex 6.1 0.03 -0.02
vertex -0.43 1.31 2.37
vertex 6.09 -0.03 -0.01
endloop
endfacet
facet normal -0.02 0.85 -0.53
outer loop
vertex 6.09 -0.03 -0.01
vertex -0.43 1.31 2.37
vertex -0.41 1.33 2.41
endloop
endfacet
facet normal -0.02 0.85 -0.53
outer loop
vertex 6.09 -0.03 -0.01
vertex -0.41 1.33 2.41
vertex 6.11 -0 0.03
endloop
endfacet
facet normal -0.08 -0.78 0.62
outer loop
vertex -1.74 -0.58 2.58
vertex 1.19 -2.17 0.94
vertex 1.22 -2.15 0.97
endloop
endfacet
facet normal -0.08 -0.78 0.62
outer loop
vertex -1.74 -0.58 2.58
vertex 1.22 -2.15 0.97
vertex -1.7 -0.56 2.61
endloop
endfacet
facet normal -0.49 -0 -0.87
outer loop
vertex -1.7 -0.56 2.61
vertex 1.22 -2.15 0.97
vertex 1.2 -2.2 0.98
endloop
endfacet
facet normal -0.49 -0 -0.87
outer loop
vertex -1.7 -0.56 2.61
vertex 1.2 -2.2 0.98
vertex -1.72 -0.61 2.63
endloop
endfacet
facet normal 0.57 0.78 0.25
outer loop
vertex -1.72 -0.61 2.63
vertex 1.2 -2.2 0.98
vertex 1.19 -2.17 0.94
endloop
endfacet
facet normal 0.57 0.78 0.25
outer loop
vertex -1.72 -0.61 2.63
vertex 1.19 -2.17 0.94
vertex -1.74 -0.58 2.58
endloop
endfacet
facet normal 0.57 -0.8 0.17
outer loop
vertex -1.82 0.93 0.7
vertex -2.88 -0.6 -2.82
vertex -2.84 -0.58 -2.84
endloop
endfacet
facet normal 0.57 -0.8 0.17
outer loop
vertex -1.82 0.93 0.7
vertex -2.84 -0.58 -2.84
vertex -1.77 0.95 0.68
endloop
endfacet
facet normal -0.96 -0 0.29
outer loop
vertex -1.77 0.95 0.68
vertex -2.84 -0.58 -2.84
vertex -2.83 -0.63 -2.82
endloop
endfacet
facet normal -0.96 -0 0.29
outer loop
vertex -1.77 0.95 0.68
vertex -2.83 -0.63 -2.82
vertex -1.77 0.9 0.7
endloop
endfacet
facet normal 0.38 0.8 -0.46
outer loop
vertex -1.77 0.9 0.7
vertex -2.83 -0.63 -2.82
vertex -2.88 -0.6 -2.82
endloop
endfacet
facet normal 0.38 0.8 -0.46
outer loop
vertex -1.77 0.9 0.7
vertex -2.88 -0.6 -2.82
vertex -1.82 0.93 0.7
endloop
endfacet
facet normal -0.22 -0.86 0.46
outer loop
vertex -6.99 -0 -3.18
vertex 1.15 -0.89 -1
vertex 1.14 -0.86 -0.96
endloop
endfacet
facet normal -0.22 -0.86 0.46
outer loop
vertex -6.99 -0 -3.18
vertex 1.14 -0.86 -0.96
vertex -7 0.03 -3.13
endloop
endfacet
facet normal 0.26 -0 -0.97
outer loop
vertex -7 0.03 -3.13
vertex 1.14 -0.86 -0.96
vertex 1.14 -0.91 -0.96
endloop
endfacet
facet normal 0.26 -0 -0.97
outer loop
vertex -7 0.03 -3.13
vertex 1.14 -0.91 -0.96
vertex -7.01 -0.03 -3.14
endloop
endfacet
facet normal -0.04 0.86 0.51
outer loop
vertex -7.01 -0.03 -3.14
vertex 1.14 -0.91 -0.96
vertex 1.15 -0.89 -1
endloop
endfacet
facet normal -0.04 0.86 0.51
outer loop
vertex -7.01 -0.03 -3.14
vertex 1.15 -0.89 -1
vertex -6.99 -0 -3.18
endloop
endfacet
facet normal -0.29 -0.86 0.43
outer loop
vertex -7.54 -0 -2.63
vertex -2.09 -0.89 -0.72
vertex -2.1 -0.86 -0.67
endloop
endfacet
facet normal -0.29 -0.86 0.43
outer loop
vertex -7.54 -0 -2.63
vertex -2.1 -0.86 -0.67
vertex -7.55 0.03 -2.58
endloop
endfacet
facet normal 0.33 -0 -0.94
outer loop
vertex -7.55 0.03 -2.58
vertex -2.1 -0.86 -0.67
vertex -2.11 -0.92 -0.67
endloop
endfacet
facet normal 0.33 -0 -0.94
outer loop
vertex -7.55 0.03 -2.58
vertex -2.11 -0.92 -0.67
vertex -7.56 -0.03 -2.59
endloop
endfacet
facet normal -0.04 0.86 0.52
outer loop
vertex -7.56 -0.03 -2.59
vertex -2.11 -0.92 -0.67
vertex -2.09 -0.89 -0.72
endloop
endfacet
facet normal -0.04 0.86 0.52
outer loop
vertex -7.56 -0.03 -2.59
vertex -2.09 -0.89 -0.72
vertex -7.54 -0 -2.63
endloop
endfacet
facet normal -0.05 -0.87 0.5
outer loop
vertex -6.45 -0 -0.58
vertex 7 -0 0.87
vertex 7 0.03 0.92
endloop
endfacet
facet normal -0.05 -0.87 0.5
outer loop
vertex -6.45 -0 -0.58
vertex 7 0.03 0.92
vertex -6.45 0.03 -0.53
endloop
endfacet
facet normal 0.11 -0 -0.99
outer loop
vertex -6.45 0.03 -0.53
vertex 7 0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal 0.11 -0 -0.99
outer loop
vertex -6.45 0.03 -0.53
vertex 7 -0.03 0.92
vertex -6.45 -0.03 -0.53
endloop
endfacet
facet normal -0.05 0.87 0.5
outer loop
vertex -6.45 -0.03 -0.53
vertex 7 -0.03 0.92
vertex 7 -0 0.87
endloop
endfacet
facet normal -0.05 0.87 0.5
outer loop
vertex -6.45 -0.03 -0.53
vertex 7 -0 0.87
vertex -6.45 -0 -0.58
endloop
endfacet
facet normal -0.8 -0.58 -0.15
outer loop
vertex -2.23 -2.73 -2
vertex -4.23 -0.21 -0.99
vertex -4.23 -0.19 -1.04
endloop
endfacet
facet normal -0.8 -0.58 -0.15
outer loop
vertex -2.23 -2.73 -2
vertex -4.23 -0.19 -1.04
vertex -2.23 -2.71 -2.05
endloop
endfacet
facet normal 0.45 -0 0.89
outer loop
vertex -2.23 -2.71 -2.05
vertex -4.23 -0.19 -1.04
vertex -4.27 -0.23 -1.02
endloop
endfacet
facet normal 0.45 -0 0.89
outer loop
vertex -2.23 -2.71 -2.05
vertex -4.27 -0.23 -1.02
vertex -2.27 -2.75 -2.03
endloop
endfacet
facet normal 0.35 0.58 -0.74
outer loop
vertex -2.27 -2.75 -2.03
vertex -4.27 -0.23 -1.02
vertex -4.23 -0.21 -0.99
endloop
endfacet
facet normal 0.35 0.58 -0.74
outer loop
vertex -2.27 -2.75 -2.03
vertex -4.23 -0.21 -0.99
vertex -2.23 -2.73 -2
endloop
endfacet
facet normal 0.11 -0.86 0.49
outer loop
vertex -7.55 -0 2.02
vertex 1.3 0.54 0.89
vertex 1.31 0.57 0.94
endloop
endfacet
facet normal 0.11 -0.86 0.49
outer loop
vertex -7.55 -0 2.02
vertex 1.31 0.57 0.94
vertex -7.55 0.03 2.07
endloop
endfacet
facet normal -0.13 -0 -0.99
outer loop
vertex -7.55 0.03 2.07
vertex 1.31 0.57 0.94
vertex 1.31 0.51 0.94
endloop
endfacet
facet normal -0.13 -0 -0.99
outer loop
vertex -7.55 0.03 2.07
vertex 1.31 0.51 0.94
vertex -7.55 -0.03 2.06
endloop
endfacet
facet normal 0.01 0.86 0.5
outer loop
vertex -7.55 -0.03 2.06
vertex 1.31 0.51 0.94
vertex 1.3 0.54 0.89
endloop
endfacet
facet normal 0.01 0.86 0.5
outer loop
vertex -7.55 -0.03 2.06
vertex 1.3 0.54 0.89
vertex -7.55 -0 2.02
endloop
endfacet
facet normal -0.87 -0.37 -0.32
outer loop
vertex -2.82 -0.6 -2.85
vertex -2.22 -2.73 -2.05
vertex -2.24 -2.72 -2
endloop
endfacet
facet normal -0.87 -0.37 -0.32
outer loop
vertex -2.82 -0.6 -2.85
vertex -2.24 -2.72 -2
vertex -2.85 -0.59 -2.8
endloop
endfacet
facet normal 0.8 -0 -0.6
outer loop
vertex -2.85 -0.59 -2.8
vertex -2.24 -2.72 -2
vertex -2.27 -2.74 -2.04
endloop
endfacet
facet normal 0.8 -0 -0.6
outer loop
vertex -2.85 -0.59 -2.8
vertex -2.27 -2.74 -2.04
vertex -2.87 -0.61 -2.84
endloop
endfacet
facet normal 0.07 0.37 0.93
outer loop
vertex -2.87 -0.61 -2.84
vertex -2.27 -2.74 -2.04
vertex -2.22 -2.73 -2.05
endloop
endfacet
facet normal 0.07 0.37 0.93
outer loop
vertex -2.87 -0.61 -2.84
vertex -2.22 -2.73 -2.05
vertex -2.82 -0.6 -2.85
endloop
endfacet
facet normal 0.55 -0.79 0.27
outer loop
vertex -7.01 -0 2.02
vertex -3.93 1.61 0.35
vertex -3.91 1.63 0.4
endloop
endfacet
facet normal 0.55 -0.79 0.27
outer loop
vertex -7.01 -0 2.02
vertex -3.91 1.63 0.4
vertex -7 0.02 2.07
endloop
endfacet
facet normal -0.48 -0 -0.88
outer loop
vertex -7 0.02 2.07
vertex -3.91 1.63 0.4
vertex -3.9 1.58 0.39
endloop
endfacet
facet normal -0.48 -0 -0.88
outer loop
vertex -7 0.02 2.07
vertex -3.9 1.58 0.39
vertex -6.98 -0.02 2.06
endloop
endfacet
facet normal -0.08 0.79 0.61
outer loop
vertex -6.98 -0.02 2.06
vertex -3.9 1.58 0.39
vertex -3.93 1.61 0.35
endloop
endfacet
facet normal -0.08 0.79 0.61
outer loop
vertex -6.98 -0.02 2.06
vertex -3.93 1.61 0.35
vertex -7.01 -0 2.02
endloop
endfacet
facet normal 0.45 -0.87 0.22
outer loop
vertex -0.45 1.33 2.37
vertex 1.54 1.43 -1.24
vertex 1.58 1.46 -1.21
endloop
endfacet
facet normal 0.45 -0.87 0.22
outer loop
vertex -0.45 1.33 2.37
vertex 1.58 1.46 -1.21
vertex -0.41 1.36 2.39
endloop
endfacet
facet normal -0.88 -0 -0.48
outer loop
vertex -0.41 1.36 2.39
vertex 1.58 1.46 -1.21
vertex 1.58 1.4 -1.22
endloop
endfacet
facet normal -0.88 -0 -0.48
outer loop
vertex -0.41 1.36 2.39
vertex 1.58 1.4 -1.22
vertex -0.41 1.31 2.39
endloop
endfacet
facet normal 0.43 0.87 0.26
outer loop
vertex -0.41 1.31 2.39
vertex 1.58 1.4 -1.22
vertex 1.54 1.43 -1.24
endloop
endfacet
facet normal 0.43 0.87 0.26
outer loop
vertex -0.41 1.31 2.39
vertex 1.54 1.43 -1.24
vertex -0.45 1.33 2.37
endloop
endfacet
facet normal -0.28 -0.84 0.48
outer loop
vertex -7.28 -0 -4.42
vertex -3.89 1.61 0.36
vertex -3.93 1.63 0.38
endloop
endfacet
facet normal -0.28 -0.84 0.48
outer loop
vertex -7.28 -0 -4.42
vertex -3.93 1.63 0.38
vertex -7.32 0.03 -4.4
endloop
endfacet
facet normal 0.82 -0 -0.58
outer loop
vertex -7.32 0.03 -4.4
vertex -3.93 1.63 0.38
vertex -3.92 1.58 0.4
endloop
endfacet
facet normal 0.82 -0 -0.58
outer loop
vertex -7.32 0.03 -4.4
vertex -3.92 1.58 0.4
vertex -7.31 -0.03 -4.39
endloop
endfacet
facet normal -0.54 0.84 0.1
outer loop
vertex -7.31 -0.03 -4.39
vertex -3.92 1.58 0.4
vertex -3.89 1.61 0.36
endloop
endfacet
facet normal -0.54 0.84 0.1
outer loop
vertex -7.31 -0.03 -4.39
vertex -3.89 1.61 0.36
vertex -7.28 -0 -4.42
endloop
endfacet
facet normal -0.35 -0.55 0.76
outer loop
vertex -0.58 -1.19 -2.53
vertex 0.04 2.33 0.33
vertex -0.01 2.34 0.32
endloop
endfacet
facet normal -0.35 -0.55 0.76
outer loop
vertex -0.58 -1.19 -2.53
vertex -0.01 2.34 0.32
vertex -0.63 -1.17 -2.54
endloop
endfacet
facet normal 0.98 -0 -0.21
outer loop
vertex -0.63 -1.17 -2.54
vertex -0.01 2.34 0.32
vertex 0 2.31 0.36
endloop
endfacet
facet normal 0.98 -0 -0.21
outer loop
vertex -0.63 -1.17 -2.54
vertex 0 2.31 0.36
vertex -0.62 -1.2 -2.5
endloop
endfacet
facet normal -0.63 0.55 -0.54
outer loop
vertex -0.62 -1.2 -2.5
vertex 0 2.31 0.36
vertex 0.04 2.33 0.33
endloop
endfacet
facet normal -0.63 0.55 -0.54
outer loop
vertex -0.62 -1.2 -2.5
vertex 0.04 2.33 0.33
vertex -0.58 -1.19 -2.53
endloop
endfacet
facet normal 0.32 -0.87 -0.39
outer loop
vertex -1.74 -0.58 2.63
vertex -3.51 -0.58 1.19
vertex -3.48 -0.56 1.15
endloop
endfacet
facet normal 0.32 -0.87 -0.39
outer loop
vertex -1.74 -0.58 2.63
vertex -3.48 -0.56 1.15
vertex -1.71 -0.56 2.59
endloop
endfacet
facet normal -0.63 -0 0.78
outer loop
vertex -1.71 -0.56 2.59
vertex -3.48 -0.56 1.15
vertex -3.48 -0.61 1.15
endloop
endfacet
facet normal -0.63 -0 0.78
outer loop
vertex -1.71 -0.56 2.59
vertex -3.48 -0.61 1.15
vertex -1.71 -0.61 2.59
endloop
endfacet
facet normal 0.32 0.87 -0.39
outer loop
vertex -1.71 -0.61 2.59
vertex -3.48 -0.61 1.15
vertex -3.51 -0.58 1.19
endloop
endfacet
facet normal 0.32 0.87 -0.39
outer loop
vertex -1.71 -0.61 2.59
vertex -3.51 -0.58 1.19
vertex -1.74 -0.58 2.63
endloop
endfacet
facet normal 0.57 -0.51 0.64
outer loop
vertex -2.13 -0.89 -0.68
vertex -2.27 -2.73 -2.03
vertex -2.23 -2.72 -2.05
endloop
endfacet
facet normal 0.57 -0.51 0.64
outer loop
vertex -2.13 -0.89 -0.68
vertex -2.23 -2.72 -2.05
vertex -2.08 -0.88 -0.71
endloop
endfacet
facet normal -0.99 -0 0.11
outer loop
vertex -2.08 -0.88 -0.71
vertex -2.23 -2.72 -2.05
vertex -2.23 -2.75 -2.01
endloop
endfacet
facet normal -0.99 -0 0.11
outer loop
vertex -2.08 -0.88 -0.71
vertex -2.23 -2.75 -2.01
vertex -2.08 -0.91 -0.67
endloop
endfacet
facet normal 0.42 0.51 -0.75
outer loop
vertex -2.08 -0.91 -0.67
vertex -2.23 -2.75 -2.01
vertex -2.27 -2.73 -2.03
endloop
endfacet
facet normal 0.42 0.51 -0.75
outer loop
vertex -2.08 -0.91 -0.67
vertex -2.27 -2.73 -2.03
vertex -2.13 -0.89 -0.68
endloop
endfacet
facet normal -0.12 -0.87 0.49
outer loop
vertex -7.29 -0 -4.43
vertex 7.01 -0 -0.93
vertex 7 0.03 -0.89
endloop
endfacet
facet normal -0.12 -0.87 0.49
outer loop
vertex -7.29 -0 -4.43
vertex 7 0.03 -0.89
vertex -7.3 0.03 -4.39
endloop
endfacet
facet normal 0.24 -0 -0.97
outer loop
vertex -7.3 0.03 -4.39
vertex 7 0.03 -0.89
vertex 7 -0.03 -0.89
endloop
endfacet
facet normal 0.24 -0 -0.97
outer loop
vertex -7.3 0.03 -4.39
vertex 7 -0.03 -0.89
vertex -7.3 -0.03 -4.39
endloop
endfacet
facet normal -0.12 0.87 0.49
outer loop
vertex -7.3 -0.03 -4.39
vertex 7 -0.03 -0.89
vertex 7.01 -0 -0.93
endloop
endfacet
facet normal -0.12 0.87 0.49
outer loop
vertex -7.3 -0.03 -4.39
vertex 7.01 -0 -0.93
vertex -7.29 -0 -4.43
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -7.54 -0 -2.08
vertex 7.91 -0 0.87
vertex 7.9 0.03 0.91
endloop
endfacet
facet normal -0.09 -0.87 0.49
outer loop
vertex -7.54 -0 -2.08
vertex 7.9 0.03 0.91
vertex -7.55 0.03 -2.04
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -7.55 0.03 -2.04
vertex 7.9 0.03 0.91
vertex 7.9 -0.03 0.91
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -7.55 0.03 -2.04
vertex 7.9 -0.03 0.91
vertex -7.55 -0.03 -2.04
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7.55 -0.03 -2.04
vertex 7.9 -0.03 0.91
vertex 7.91 -0 0.87
endloop
endfacet
facet normal -0.09 0.87 0.49
outer loop
vertex -7.55 -0.03 -2.04
vertex 7.91 -0 0.87
vertex -7.54 -0 -2.08
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex 7 -0 0.87
vertex 7.9 -0 0.87
vertex 7.9 0.03 0.92
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex 7 -0 0.87
vertex 7.9 0.03 0.92
vertex 7 0.03 0.92
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex 7 0.03 0.92
vertex 7.9 0.03 0.92
vertex 7.9 -0.03 0.92
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex 7 0.03 0.92
vertex 7.9 -0.03 0.92
vertex 7 -0.03 0.92
endloop
endfacet
facet normal -0 0.87 0.5
outer loop
vertex 7 -0.03 0.92
vertex 7.9 -0.03 0.92
vertex 7.9 -0 0.87
endloop
endfacet
facet normal 0 0.87 0.5
outer loop
vertex 7 -0.03 0.92
vertex 7.9 -0 0.87
vertex 7 -0 0.87
endloop
endfacet
facet normal 0.24 -0.86 -0.44
outer loop
vertex 6.09 -0 0.93
vertex -2.86 -0.6 -2.8
vertex -2.84 -0.58 -2.84
endloop
endfacet
facet normal 0.24 -0.86 -0.44
outer loop
vertex 6.09 -0 0.93
vertex -2.84 -0.58 -2.84
vertex 6.1 0.03 0.89
endloop
endfacet
facet normal -0.38 -0 0.92
outer loop
vertex 6.1 0.03 0.89
vertex -2.84 -0.58 -2.84
vertex -2.84 -0.63 -2.84
endloop
endfacet
facet normal -0.38 -0 0.92
outer loop
vertex 6.1 0.03 0.89
vertex -2.84 -0.63 -2.84
vertex 6.11 -0.03 0.89
endloop
endfacet
facet normal 0.14 0.86 -0.48
outer loop
vertex 6.11 -0.03 0.89
vertex -2.84 -0.63 -2.84
vertex -2.86 -0.6 -2.8
endloop
endfacet
facet normal 0.14 0.86 -0.48
outer loop
vertex 6.11 -0.03 0.89
vertex -2.86 -0.6 -2.8
vertex 6.09 -0 0.93
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.99 -0 -2.63
vertex 6.11 -0 -0.03
vertex 6.1 0.03 0.01
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.99 -0 -2.63
vertex 6.1 0.03 0.01
vertex -7 0.03 -2.59
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -7 0.03 -2.59
vertex 6.1 0.03 0.01
vertex 6.1 -0.03 0.01
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -7 0.03 -2.59
vertex 6.1 -0.03 0.01
vertex -7 -0.03 -2.59
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -7 -0.03 -2.59
vertex 6.1 -0.03 0.01
vertex 6.11 -0 -0.03
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -7 -0.03 -2.59
vertex 6.11 -0 -0.03
vertex -6.99 -0 -2.63
endloop
endfacet
facet normal -0.27 -0.78 0.56
outer loop
vertex -7.01 -0 2.02
vertex -2 -2.39 1.11
vertex -1.98 -2.37 1.15
endloop
endfacet
facet normal -0.27 -0.78 0.56
outer loop
vertex -7.01 -0 2.02
vertex -1.98 -2.37 1.15
vertex -6.99 0.02 2.06
endloop
endfacet
facet normal -0.18 -0 -0.98
outer loop
vertex -6.99 0.02 2.06
vertex -1.98 -2.37 1.15
vertex -2 -2.42 1.16
endloop
endfacet
facet normal -0.18 -0 -0.98
outer loop
vertex -6.99 0.02 2.06
vertex -2 -2.42 1.16
vertex -7.01 -0.02 2.07
endloop
endfacet
facet normal 0.45 0.78 0.43
outer loop
vertex -7.01 -0.02 2.07
vertex -2 -2.42 1.16
vertex -2 -2.39 1.11
endloop
endfacet
facet normal 0.45 0.78 0.43
outer loop
vertex -7.01 -0.02 2.07
vertex -2 -2.39 1.11
vertex -7.01 -0 2.02
endloop
endfacet
facet normal 0.52 -0.81 0.27
outer loop
vertex -7.02 -0 2.02
vertex -0.74 2.72 -1.86
vertex -0.73 2.74 -1.82
endloop
endfacet
facet normal 0.52 -0.81 0.27
outer loop
vertex -7.02 -0 2.02
vertex -0.73 2.74 -1.82
vertex -7 0.02 2.07
endloop
endfacet
facet normal -0.53 -0 -0.85
outer loop
vertex -7 0.02 2.07
vertex -0.73 2.74 -1.82
vertex -0.71 2.69 -1.83
endloop
endfacet
facet normal -0.53 -0 -0.85
outer loop
vertex -7 0.02 2.07
vertex -0.71 2.69 -1.83
vertex -6.98 -0.02 2.06
endloop
endfacet
facet normal 0.01 0.81 0.58
outer loop
vertex -6.98 -0.02 2.06
vertex -0.71 2.69 -1.83
vertex -0.74 2.72 -1.86
endloop
endfacet
facet normal 0.01 0.81 0.58
outer loop
vertex -6.98 -0.02 2.06
vertex -0.74 2.72 -1.86
vertex -7.02 -0 2.02
endloop
endfacet
facet normal 0.23 -0.86 0.46
outer loop
vertex -6.47 -0 3.13
vertex -2.12 -0.89 -0.71
vertex -2.08 -0.87 -0.68
endloop
endfacet
facet normal 0.23 -0.86 0.46
outer loop
vertex -6.47 -0 3.13
vertex -2.08 -0.87 -0.68
vertex -6.44 0.03 3.16
endloop
endfacet
facet normal -0.66 -0 -0.75
outer loop
vertex -6.44 0.03 3.16
vertex -2.08 -0.87 -0.68
vertex -2.09 -0.92 -0.67
endloop
endfacet
facet normal -0.66 -0 -0.75
outer loop
vertex -6.44 0.03 3.16
vertex -2.09 -0.92 -0.67
vertex -6.44 -0.03 3.16
endloop
endfacet
facet normal 0.43 0.86 0.29
outer loop
vertex -6.44 -0.03 3.16
vertex -2.09 -0.92 -0.67
vertex -2.12 -0.89 -0.71
endloop
endfacet
facet normal 0.43 0.86 0.29
outer loop
vertex -6.44 -0.03 3.16
vertex -2.12 -0.89 -0.71
vertex -6.47 -0 3.13
endloop
endfacet
facet normal 0.12 -0.87 0.49
outer loop
vertex -6.46 -0 2.57
vertex 7.89 -0 -0.93
vertex 7.9 0.03 -0.89
endloop
endfacet
facet normal 0.12 -0.87 0.49
outer loop
vertex -6.46 -0 2.57
vertex 7.9 0.03 -0.89
vertex -6.45 0.03 2.61
endloop
endfacet
facet normal -0.24 -0 -0.97
outer loop
vertex -6.45 0.03 2.61
vertex 7.9 0.03 -0.89
vertex 7.9 -0.03 -0.89
endloop
endfacet
facet normal -0.24 -0 -0.97
outer loop
vertex -6.45 0.03 2.61
vertex 7.9 -0.03 -0.89
vertex -6.45 -0.03 2.61
endloop
endfacet
facet normal 0.12 0.87 0.49
outer loop
vertex -6.45 -0.03 2.61
vertex 7.9 -0.03 -0.89
vertex 7.89 -0 -0.93
endloop
endfacet
facet normal 0.12 0.87 0.49
outer loop
vertex -6.45 -0.03 2.61
vertex 7.89 -0 -0.93
vertex -6.46 -0 2.57
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -7 -0 -0.58
vertex 7 -0 -0.93
vertex 7 0.03 -0.88
endloop
endfacet
facet normal 0.01 -0.87 0.5
outer loop
vertex -7 -0 -0.58
vertex 7 0.03 -0.88
vertex -7 0.03 -0.53
endloop
endfacet
facet normal -0.02 -0 -1
outer loop
vertex -7 0.03 -0.53
vertex 7 0.03 -0.88
vertex 7 -0.03 -0.88
endloop
endfacet
facet normal -0.02 -0 -1
outer loop
vertex -7 0.03 -0.53
vertex 7 -0.03 -0.88
vertex -7 -0.03 -0.53
endloop
endfacet
facet normal 0.01 0.87 0.5
outer loop
vertex -7 -0.03 -0.53
vertex 7 -0.03 -0.88
vertex 7 -0 -0.93
endloop
endfacet
facet normal 0.01 0.87 0.5
outer loop
vertex -7 -0.03 -0.53
vertex 7 -0 -0.93
vertex -7 -0 -0.58
endloop
endfacet
facet normal -0.29 -0.85 -0.43
outer loop
vertex 7.01 -0 0.03
vertex -0.41 1.33 2.41
vertex -0.42 1.36 2.37
endloop
endfacet
facet normal -0.29 -0.85 -0.43
outer loop
vertex 7.01 -0 0.03
vertex -0.42 1.36 2.37
vertex 7 0.03 -0.02
endloop
endfacet
facet normal 0.31 -0 0.95
outer loop
vertex 7 0.03 -0.02
vertex -0.42 1.36 2.37
vertex -0.43 1.31 2.37
endloop
endfacet
facet normal 0.31 -0 0.95
outer loop
vertex 7 0.03 -0.02
vertex -0.43 1.31 2.37
vertex 6.99 -0.03 -0.01
endloop
endfacet
facet normal -0.01 0.85 -0.52
outer loop
vertex 6.99 -0.03 -0.01
vertex -0.43 1.31 2.37
vertex -0.41 1.33 2.41
endloop
endfacet
facet normal -0.01 0.85 -0.52
outer loop
vertex 6.99 -0.03 -0.01
vertex -0.41 1.33 2.41
vertex 7.01 -0 0.03
endloop
endfacet
facet normal -0.05 -0.86 0.5
outer loop
vertex -2.84 -0.6 -2.86
vertex 7.01 -0 -0.93
vertex 7 0.03 -0.89
endloop
endfacet
facet normal -0.05 -0.86 0.5
outer loop
vertex -2.84 -0.6 -2.86
vertex 7 0.03 -0.89
vertex -2.85 -0.58 -2.82
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -2.85 -0.58 -2.82
vertex 7 0.03 -0.89
vertex 7 -0.03 -0.88
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex -2.85 -0.58 -2.82
vertex 7 -0.03 -0.88
vertex -2.85 -0.63 -2.81
endloop
endfacet
facet normal -0.15 0.86 0.48
outer loop
vertex -2.85 -0.63 -2.81
vertex 7 -0.03 -0.88
vertex 7.01 -0 -0.93
endloop
endfacet
facet normal -0.15 0.86 0.48
outer loop
vertex -2.85 -0.63 -2.81
vertex 7.01 -0 -0.93
vertex -2.84 -0.6 -2.86
endloop
endfacet
facet normal 0.06 -0.86 -0.5
outer loop
vertex 7.89 -0 0.93
vertex -0.51 0.59 -1.07
vertex -0.5 0.62 -1.11
endloop
endfacet
facet normal 0.06 -0.86 -0.5
outer loop
vertex 7.89 -0 0.93
vertex -0.5 0.62 -1.11
vertex 7.91 0.03 0.89
endloop
endfacet
facet normal -0.23 -0 0.97
outer loop
vertex 7.91 0.03 0.89
vertex -0.5 0.62 -1.11
vertex -0.5 0.57 -1.11
endloop
endfacet
facet normal -0.23 -0 0.97
outer loop
vertex 7.91 0.03 0.89
vertex -0.5 0.57 -1.11
vertex 7.9 -0.03 0.88
endloop
endfacet
facet normal 0.17 0.86 -0.47
outer loop
vertex 7.9 -0.03 0.88
vertex -0.5 0.57 -1.11
vertex -0.51 0.59 -1.07
endloop
endfacet
facet normal 0.17 0.86 -0.47
outer loop
vertex 7.9 -0.03 0.88
vertex -0.51 0.59 -1.07
vertex 7.89 -0 0.93
endloop
endfacet
facet normal -0.43 -0.81 0.4
outer loop
vertex -0.72 2.72 -1.86
vertex 6.11 -0 -0.03
vertex 6.11 0.02 0.02
endloop
endfacet
facet normal -0.43 -0.81 0.4
outer loop
vertex -0.72 2.72 -1.86
vertex 6.11 0.02 0.02
vertex -0.72 2.74 -1.82
endloop
endfacet
facet normal 0.26 -0 -0.97
outer loop
vertex -0.72 2.74 -1.82
vertex 6.11 0.02 0.02
vertex 6.09 -0.02 0.01
endloop
endfacet
facet normal 0.26 -0 -0.97
outer loop
vertex -0.72 2.74 -1.82
vertex 6.09 -0.02 0.01
vertex -0.74 2.69 -1.82
endloop
endfacet
facet normal 0.17 0.81 0.56
outer loop
vertex -0.74 2.69 -1.82
vertex 6.09 -0.02 0.01
vertex 6.11 -0 -0.03
endloop
endfacet
facet normal 0.17 0.81 0.56
outer loop
vertex -0.74 2.69 -1.82
vertex 6.11 -0 -0.03
vertex -0.72 2.72 -1.86
endloop
endfacet
facet normal -0.01 -0.31 -0.95
outer loop
vertex -3.82 -2.39 -0.29
vertex -4.27 -0.21 -1
vertex -4.22 -0.2 -1
endloop
endfacet
facet normal -0.01 -0.31 -0.95
outer loop
vertex -3.82 -2.39 -0.29
vertex -4.22 -0.2 -1
vertex -3.77 -2.39 -0.3
endloop
endfacet
facet normal -0.84 -0 0.54
outer loop
vertex -3.77 -2.39 -0.3
vertex -4.22 -0.2 -1
vertex -4.24 -0.22 -1.04
endloop
endfacet
facet normal -0.84 -0 0.54
outer loop
vertex -3.77 -2.39 -0.3
vertex -4.24 -0.22 -1.04
vertex -3.79 -2.4 -0.34
endloop
endfacet
facet normal 0.86 0.31 0.41
outer loop
vertex -3.79 -2.4 -0.34
vertex -4.24 -0.22 -1.04
vertex -4.27 -0.21 -1
endloop
endfacet
facet normal 0.86 0.31 0.41
outer loop
vertex -3.79 -2.4 -0.34
vertex -4.27 -0.21 -1
vertex -3.82 -2.39 -0.29
endloop
endfacet
facet normal 0.2 -0.86 0.46
outer loop
vertex -7.01 -0 0.52
vertex -4.26 -0.21 -1.04
vertex -4.23 -0.19 -1
endloop
endfacet
facet normal 0.2 -0.86 0.46
outer loop
vertex -7.01 -0 0.52
vertex -4.23 -0.19 -1
vertex -6.99 0.03 0.56
endloop
endfacet
facet normal -0.49 -0 -0.87
outer loop
vertex -6.99 0.03 0.56
vertex -4.23 -0.19 -1
vertex -4.24 -0.24 -1
endloop
endfacet
facet normal -0.49 -0 -0.87
outer loop
vertex -6.99 0.03 0.56
vertex -4.24 -0.24 -1
vertex -6.99 -0.03 0.56
endloop
endfacet
facet normal 0.3 0.86 0.41
outer loop
vertex -6.99 -0.03 0.56
vertex -4.24 -0.24 -1
vertex -4.26 -0.21 -1.04
endloop
endfacet
facet normal 0.3 0.86 0.41
outer loop
vertex -6.99 -0.03 0.56
vertex -4.26 -0.21 -1.04
vertex -7.01 -0 0.52
endloop
endfacet
facet normal 0.32 -0.86 0.4
outer loop
vertex -3.94 1.61 0.36
vertex -1.21 0.93 -3.27
vertex -1.17 0.96 -3.25
endloop
endfacet
facet normal 0.32 -0.86 0.4
outer loop
vertex -3.94 1.61 0.36
vertex -1.17 0.96 -3.25
vertex -3.9 1.63 0.39
endloop
endfacet
facet normal -0.8 -0 -0.6
outer loop
vertex -3.9 1.63 0.39
vertex -1.17 0.96 -3.25
vertex -1.18 0.91 -3.24
endloop
endfacet
facet normal -0.8 -0 -0.6
outer loop
vertex -3.9 1.63 0.39
vertex -1.18 0.91 -3.24
vertex -3.9 1.58 0.39
endloop
endfacet
facet normal 0.48 0.86 0.2
outer loop
vertex -3.9 1.58 0.39
vertex -1.18 0.91 -3.24
vertex -1.21 0.93 -3.27
endloop
endfacet
facet normal 0.48 0.86 0.2
outer loop
vertex -3.9 1.58 0.39
vertex -1.21 0.93 -3.27
vertex -3.94 1.61 0.36
endloop
endfacet
facet normal -0.38 -0.86 0.33
outer loop
vertex -7.28 -0 -4.42
vertex 0.6 -0.77 2.72
vertex 0.57 -0.75 2.75
endloop
endfacet
facet normal -0.38 -0.86 0.33
outer loop
vertex -7.28 -0 -4.42
vertex 0.57 -0.75 2.75
vertex -7.31 0.03 -4.39
endloop
endfacet
facet normal 0.67 -0 -0.74
outer loop
vertex -7.31 0.03 -4.39
vertex 0.57 -0.75 2.75
vertex 0.57 -0.8 2.75
endloop
endfacet
facet normal 0.67 -0 -0.74
outer loop
vertex -7.31 0.03 -4.39
vertex 0.57 -0.8 2.75
vertex -7.31 -0.03 -4.39
endloop
endfacet
facet normal -0.29 0.86 0.41
outer loop
vertex -7.31 -0.03 -4.39
vertex 0.57 -0.8 2.75
vertex 0.6 -0.77 2.72
endloop
endfacet
facet normal -0.29 0.86 0.41
outer loop
vertex -7.31 -0.03 -4.39
vertex 0.6 -0.77 2.72
vertex -7.28 -0 -4.42
endloop
endfacet
facet normal -0.55 -0.7 -0.45
outer loop
vertex -0.58 -1.19 -2.53
vertex -0.4 -2.55 -0.64
vertex -0.44 -2.53 -0.62
endloop
endfacet
facet normal -0.55 -0.7 -0.45
outer loop
vertex -0.58 -1.19 -2.53
vertex -0.44 -2.53 -0.62
vertex -0.63 -1.17 -2.51
endloop
endfacet
facet normal 1 -0 -0.1
outer loop
vertex -0.63 -1.17 -2.51
vertex -0.44 -2.53 -0.62
vertex -0.44 -2.57 -0.65
endloop
endfacet
facet normal 1 -0 -0.1
outer loop
vertex -0.63 -1.17 -2.51
vertex -0.44 -2.57 -0.65
vertex -0.63 -1.21 -2.54
endloop
endfacet
facet normal -0.45 0.7 0.55
outer loop
vertex -0.63 -1.21 -2.54
vertex -0.44 -2.57 -0.65
vertex -0.4 -2.55 -0.64
endloop
endfacet
facet normal -0.45 0.7 0.55
outer loop
vertex -0.63 -1.21 -2.54
vertex -0.4 -2.55 -0.64
vertex -0.58 -1.19 -2.53
endloop
endfacet
facet normal -0.08 -0.87 0.49
outer loop
vertex -6.45 -0 -3.18
vertex 7 -0 -0.93
vertex 7 0.03 -0.89
endloop
endfacet
facet normal -0.08 -0.87 0.49
outer loop
vertex -6.45 -0 -3.18
vertex 7 0.03 -0.89
vertex -6.45 0.03 -3.14
endloop
endfacet
facet normal 0.16 -0 -0.99
outer loop
vertex -6.45 0.03 -3.14
vertex 7 0.03 -0.89
vertex 7 -0.03 -0.89
endloop
endfacet
facet normal 0.16 -0 -0.99
outer loop
vertex -6.45 0.03 -3.14
vertex 7 -0.03 -0.89
vertex -6.45 -0.03 -3.14
endloop
endfacet
facet normal -0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 -3.14
vertex 7 -0.03 -0.89
vertex 7 -0 -0.93
endloop
endfacet
facet normal -0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 -3.14
vertex 7 -0 -0.93
vertex -6.45 -0 -3.18
endloop
endfacet
facet normal 0.4 -0.87 0.31
outer loop
vertex -7.03 -0 3.13
vertex -4.27 -0.21 -1.03
vertex -4.23 -0.19 -1.01
endloop
endfacet
facet normal 0.4 -0.87 0.31
outer loop
vertex -7.03 -0 3.13
vertex -4.23 -0.19 -1.01
vertex -6.99 0.03 3.16
endloop
endfacet
facet normal -0.83 -0 -0.55
outer loop
vertex -6.99 0.03 3.16
vertex -4.23 -0.19 -1.01
vertex -4.23 -0.24 -1
endloop
endfacet
facet normal -0.83 -0 -0.55
outer loop
vertex -6.99 0.03 3.16
vertex -4.23 -0.24 -1
vertex -6.99 -0.03 3.16
endloop
endfacet
facet normal 0.44 0.87 0.25
outer loop
vertex -6.99 -0.03 3.16
vertex -4.23 -0.24 -1
vertex -4.27 -0.21 -1.03
endloop
endfacet
facet normal 0.44 0.87 0.25
outer loop
vertex -6.99 -0.03 3.16
vertex -4.27 -0.21 -1.03
vertex -7.03 -0 3.13
endloop
endfacet
facet normal 0.76 -0.65 0.1
outer loop
vertex -2.01 -2.39 1.11
vertex 1.55 1.43 -1.25
vertex 1.56 1.45 -1.2
endloop
endfacet
facet normal 0.76 -0.65 0.1
outer loop
vertex -2.01 -2.39 1.11
vertex 1.56 1.45 -1.2
vertex -2 -2.37 1.16
endloop
endfacet
facet normal -0.55 -0 -0.83
outer loop
vertex -2 -2.37 1.16
vertex 1.56 1.45 -1.2
vertex 1.59 1.41 -1.22
endloop
endfacet
facet normal -0.55 -0 -0.83
outer loop
vertex -2 -2.37 1.16
vertex 1.59 1.41 -1.22
vertex -1.97 -2.41 1.14
endloop
endfacet
facet normal -0.2 0.65 0.74
outer loop
vertex -1.97 -2.41 1.14
vertex 1.59 1.41 -1.22
vertex 1.55 1.43 -1.25
endloop
endfacet
facet normal -0.2 0.65 0.74
outer loop
vertex -1.97 -2.41 1.14
vertex 1.55 1.43 -1.25
vertex -2.01 -2.39 1.11
endloop
endfacet
facet normal 0.08 -0.83 0.56
outer loop
vertex -6.99 -0 -2.08
vertex 0.02 2.33 0.3
vertex 0 2.35 0.34
endloop
endfacet
facet normal 0.08 -0.83 0.56
outer loop
vertex -6.99 -0 -2.08
vertex 0 2.35 0.34
vertex -7.01 0.02 -2.04
endloop
endfacet
facet normal 0.32 -0 -0.95
outer loop
vertex -7.01 0.02 -2.04
vertex 0 2.35 0.34
vertex 0.02 2.3 0.35
endloop
endfacet
facet normal 0.32 -0 -0.95
outer loop
vertex -7.01 0.02 -2.04
vertex 0.02 2.3 0.35
vertex -7 -0.02 -2.03
endloop
endfacet
facet normal -0.41 0.83 0.39
outer loop
vertex -7 -0.02 -2.03
vertex 0.02 2.3 0.35
vertex 0.02 2.33 0.3
endloop
endfacet
facet normal -0.41 0.83 0.39
outer loop
vertex -7 -0.02 -2.03
vertex 0.02 2.33 0.3
vertex -6.99 -0 -2.08
endloop
endfacet
facet normal -0.09 -0.78 -0.61
outer loop
vertex 0.56 -0.77 2.77
vertex -3.93 1.61 0.41
vertex -3.9 1.63 0.37
endloop
endfacet
facet normal -0.09 -0.78 -0.61
outer loop
vertex 0.56 -0.77 2.77
vertex -3.9 1.63 0.37
vertex 0.59 -0.75 2.73
endloop
endfacet
facet normal -0.46 -0 0.89
outer loop
vertex 0.59 -0.75 2.73
vertex -3.9 1.63 0.37
vertex -3.92 1.58 0.36
endloop
endfacet
facet normal -0.46 -0 0.89
outer loop
vertex 0.59 -0.75 2.73
vertex -3.92 1.58 0.36
vertex 0.57 -0.8 2.72
endloop
endfacet
facet normal 0.56 0.78 -0.27
outer loop
vertex 0.57 -0.8 2.72
vertex -3.92 1.58 0.36
vertex -3.93 1.61 0.41
endloop
endfacet
facet normal 0.56 0.78 -0.27
outer loop
vertex 0.57 -0.8 2.72
vertex -3.93 1.61 0.41
vertex 0.56 -0.77 2.77
endloop
endfacet
facet normal 0.55 -0.54 0.63
outer loop
vertex -0.53 0.59 -1.1
vertex -0.64 -1.19 -2.53
vertex -0.6 -1.17 -2.55
endloop
endfacet
facet normal 0.55 -0.54 0.63
outer loop
vertex -0.53 0.59 -1.1
vertex -0.6 -1.17 -2.55
vertex -0.49 0.61 -1.12
endloop
endfacet
facet normal -1 -0 0.08
outer loop
vertex -0.49 0.61 -1.12
vertex -0.6 -1.17 -2.55
vertex -0.6 -1.2 -2.51
endloop
endfacet
facet normal -1 -0 0.08
outer loop
vertex -0.49 0.61 -1.12
vertex -0.6 -1.2 -2.51
vertex -0.49 0.58 -1.08
endloop
endfacet
facet normal 0.45 0.54 -0.71
outer loop
vertex -0.49 0.58 -1.08
vertex -0.6 -1.2 -2.51
vertex -0.64 -1.19 -2.53
endloop
endfacet
facet normal 0.45 0.54 -0.71
outer loop
vertex -0.49 0.58 -1.08
vertex -0.64 -1.19 -2.53
vertex -0.53 0.59 -1.1
endloop
endfacet
facet normal -0.9 -0.39 0.21
outer loop
vertex -0.41 1.33 2.35
vertex 0.59 -0.77 2.71
vertex 0.59 -0.76 2.76
endloop
endfacet
facet normal -0.9 -0.39 0.21
outer loop
vertex -0.41 1.33 2.35
vertex 0.59 -0.76 2.76
vertex -0.4 1.35 2.4
endloop
endfacet
facet normal 0.34 -0 -0.94
outer loop
vertex -0.4 1.35 2.4
vertex 0.59 -0.76 2.76
vertex 0.55 -0.78 2.74
endloop
endfacet
facet normal 0.34 -0 -0.94
outer loop
vertex -0.4 1.35 2.4
vertex 0.55 -0.78 2.74
vertex -0.45 1.32 2.39
endloop
endfacet
facet normal 0.56 0.39 0.73
outer loop
vertex -0.45 1.32 2.39
vertex 0.55 -0.78 2.74
vertex 0.59 -0.77 2.71
endloop
endfacet
facet normal 0.56 0.39 0.73
outer loop
vertex -0.45 1.32 2.39
vertex 0.59 -0.77 2.71
vertex -0.41 1.33 2.35
endloop
endfacet
facet normal 0.08 -0.87 0.49
outer loop
vertex -6.45 -0 2.02
vertex 6.1 -0 -0.03
vertex 6.1 0.03 0.01
endloop
endfacet
facet normal 0.08 -0.87 0.49
outer loop
vertex -6.45 -0 2.02
vertex 6.1 0.03 0.01
vertex -6.45 0.03 2.06
endloop
endfacet
facet normal -0.16 -0 -0.99
outer loop
vertex -6.45 0.03 2.06
vertex 6.1 0.03 0.01
vertex 6.1 -0.03 0.01
endloop
endfacet
facet normal -0.16 -0 -0.99
outer loop
vertex -6.45 0.03 2.06
vertex 6.1 -0.03 0.01
vertex -6.45 -0.03 2.06
endloop
endfacet
facet normal 0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 2.06
vertex 6.1 -0.03 0.01
vertex 6.1 -0 -0.03
endloop
endfacet
facet normal 0.08 0.87 0.49
outer loop
vertex -6.45 -0.03 2.06
vertex 6.1 -0 -0.03
vertex -6.45 -0 2.02
endloop
endfacet
facet normal -0.01 -0.6 -0.8
outer loop
vertex 1.18 -2.17 0.98
vertex -0.53 0.59 -1.08
vertex -0.48 0.61 -1.09
endloop
endfacet
facet normal -0.01 -0.6 -0.8
outer loop
vertex 1.18 -2.17 0.98
vertex -0.48 0.61 -1.09
vertex 1.22 -2.15 0.97
endloop
endfacet
facet normal -0.77 -0 0.64
outer loop
vertex 1.22 -2.15 0.97
vertex -0.48 0.61 -1.09
vertex -0.5 0.57 -1.12
endloop
endfacet
facet normal -0.77 -0 0.64
outer loop
vertex 1.22 -2.15 0.97
vertex -0.5 0.57 -1.12
vertex 1.2 -2.19 0.94
endloop
endfacet
facet normal 0.78 0.6 0.16
outer loop
vertex 1.2 -2.19 0.94
vertex -0.5 0.57 -1.12
vertex -0.53 0.59 -1.08
endloop
endfacet
facet normal 0.78 0.6 0.16
outer loop
vertex 1.2 -2.19 0.94
vertex -0.53 0.59 -1.08
vertex 1.18 -2.17 0.98
endloop
endfacet
facet normal -0.01 -0.83 0.56
outer loop
vertex -6.99 -0 0.52
vertex -3.06 1.3 2.51
vertex -3.09 1.32 2.55
endloop
endfacet
facet normal -0.01 -0.83 0.56
outer loop
vertex -6.99 -0 0.52
vertex -3.09 1.32 2.55
vertex -7.01 0.02 0.56
endloop
endfacet
facet normal 0.45 -0 -0.89
outer loop
vertex -7.01 0.02 0.56
vertex -3.09 1.32 2.55
vertex -3.07 1.27 2.56
endloop
endfacet
facet normal 0.45 -0 -0.89
outer loop
vertex -7.01 0.02 0.56
vertex -3.07 1.27 2.56
vertex -7 -0.02 0.57
endloop
endfacet
facet normal -0.44 0.83 0.34
outer loop
vertex -7 -0.02 0.57
vertex -3.07 1.27 2.56
vertex -3.06 1.3 2.51
endloop
endfacet
facet normal -0.44 0.83 0.34
outer loop
vertex -7 -0.02 0.57
vertex -3.06 1.3 2.51
vertex -6.99 -0 0.52
endloop
endfacet
facet normal -0.52 -0.37 -0.77
outer loop
vertex -0.48 -0.76 0.76
vertex -2.14 3.06 0.04
vertex -2.1 3.07 0.01
endloop
endfacet
facet normal -0.52 -0.37 -0.77
outer loop
vertex -0.48 -0.76 0.76
vertex -2.1 3.07 0.01
vertex -0.44 -0.75 0.73
endloop
endfacet
facet normal -0.4 -0 0.92
outer loop
vertex -0.44 -0.75 0.73
vertex -2.1 3.07 0.01
vertex -2.14 3.05 -0.01
endloop
endfacet
facet normal -0.4 -0 0.92
outer loop
vertex -0.44 -0.75 0.73
vertex -2.14 3.05 -0.01
vertex -0.48 -0.77 0.71
endloop
endfacet
facet normal 0.92 0.37 -0.15
outer loop
vertex -0.48 -0.77 0.71
vertex -2.14 3.05 -0.01
vertex -2.14 3.06 0.04
endloop
endfacet
facet normal 0.92 0.37 -0.15
outer loop
vertex -0.48 -0.77 0.71
vertex -2.14 3.06 0.04
vertex -0.48 -0.76 0.76
endloop
endfacet
facet normal -0.28 -0.85 0.46
outer loop
vertex 1.57 1.43 -1.25
vertex 7.91 -0 -0.03
vertex 7.9 0.03 0.02
endloop
endfacet
facet normal -0.28 -0.85 0.46
outer loop
vertex 1.57 1.43 -1.25
vertex 7.9 0.03 0.02
vertex 1.57 1.46 -1.21
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex 1.57 1.46 -1.21
vertex 7.9 0.03 0.02
vertex 7.89 -0.03 0.01
endloop
endfacet
facet normal 0.19 -0 -0.98
outer loop
vertex 1.57 1.46 -1.21
vertex 7.89 -0.03 0.01
vertex 1.56 1.4 -1.21
endloop
endfacet
facet normal 0.09 0.85 0.53
outer loop
vertex 1.56 1.4 -1.21
vertex 7.89 -0.03 0.01
vertex 7.91 -0 -0.03
endloop
endfacet
facet normal 0.09 0.85 0.53
outer loop
vertex 1.56 1.4 -1.21
vertex 7.91 -0 -0.03
vertex 1.57 1.43 -1.25
endloop
endfacet
facet normal -0.29 -0.85 0.44
outer loop
vertex -6.98 -0 -2.62
vertex -3.05 1.3 2.52
vertex -3.09 1.33 2.55
endloop
endfacet
facet normal -0.29 -0.85 0.44
outer loop
vertex -6.98 -0 -2.62
vertex -3.09 1.33 2.55
vertex -7.02 0.03 -2.59
endloop
endfacet
facet normal 0.79 -0 -0.61
outer loop
vertex -7.02 0.03 -2.59
vertex -3.09 1.33 2.55
vertex -3.08 1.27 2.55
endloop
endfacet
facet normal 0.79 -0 -0.61
outer loop
vertex -7.02 0.03 -2.59
vertex -3.08 1.27 2.55
vertex -7.01 -0.03 -2.59
endloop
endfacet
facet normal -0.5 0.85 0.17
outer loop
vertex -7.01 -0.03 -2.59
vertex -3.08 1.27 2.55
vertex -3.05 1.3 2.52
endloop
endfacet
facet normal -0.5 0.85 0.17
outer loop
vertex -7.01 -0.03 -2.59
vertex -3.05 1.3 2.52
vertex -6.98 -0 -2.62
endloop
endfacet
facet normal 0.12 -0.87 0.49
outer loop
vertex -7.01 -0 2.57
vertex 6.99 -0 -0.93
vertex 7 0.03 -0.89
endloop
endfacet
facet normal 0.12 -0.87 0.49
outer loop
vertex -7.01 -0 2.57
vertex 7 0.03 -0.89
vertex -7 0.03 2.61
endloop
endfacet
facet normal -0.24 -0 -0.97
outer loop
vertex -7 0.03 2.61
vertex 7 0.03 -0.89
vertex 7 -0.03 -0.89
endloop
endfacet
facet normal -0.24 -0 -0.97
outer loop
vertex -7 0.03 2.61
vertex 7 -0.03 -0.89
vertex -7 -0.03 2.61
endloop
endfacet
facet normal 0.12 0.87 0.49
outer loop
vertex -7 -0.03 2.61
vertex 7 -0.03 -0.89
vertex 6.99 -0 -0.93
endloop
endfacet
facet normal 0.12 0.87 0.49
outer loop
vertex -7 -0.03 2.61
vertex 6.99 -0 -0.93
vertex -7.01 -0 2.57
endloop
endfacet
facet normal 0.86 -0.47 0.18
outer loop
vertex -0.01 2.33 0.35
vertex -2.26 -2.73 -2.01
vertex -2.25 -2.72 -2.06
endloop
endfacet
facet normal 0.86 -0.47 0.18
outer loop
vertex -0.01 2.33 0.35
vertex -2.25 -2.72 -2.06
vertex 0.01 2.34 0.31
endloop
endfacet
facet normal -0.72 -0 0.69
outer loop
vertex 0.01 2.34 0.31
vertex -2.25 -2.72 -2.06
vertex -2.22 -2.75 -2.02
endloop
endfacet
facet normal -0.72 -0 0.69
outer loop
vertex 0.01 2.34 0.31
vertex -2.22 -2.75 -2.02
vertex 0.04 2.31 0.34
endloop
endfacet
facet normal -0.14 0.47 -0.87
outer loop
vertex 0.04 2.31 0.34
vertex -2.22 -2.75 -2.02
vertex -2.26 -2.73 -2.01
endloop
endfacet
facet normal -0.14 0.47 -0.87
outer loop
vertex 0.04 2.31 0.34
vertex -2.26 -2.73 -2.01
vertex -0.01 2.33 0.35
endloop
endfacet
facet normal 0.21 -0.86 0.46
outer loop
vertex -6.46 -0 0.52
vertex -0.51 0.59 -1.13
vertex -0.5 0.62 -1.08
endloop
endfacet
facet normal 0.21 -0.86 0.46
outer loop
vertex -6.46 -0 0.52
vertex -0.5 0.62 -1.08
vertex -6.45 0.03 0.57
endloop
endfacet
facet normal -0.27 -0 -0.96
outer loop
vertex -6.45 0.03 0.57
vertex -0.5 0.62 -1.08
vertex -0.5 0.57 -1.08
endloop
endfacet
facet normal -0.27 -0 -0.96
outer loop
vertex -6.45 0.03 0.57
vertex -0.5 0.57 -1.08
vertex -6.44 -0.03 0.56
endloop
endfacet
facet normal 0.05 0.86 0.5
outer loop
vertex -6.44 -0.03 0.56
vertex -0.5 0.57 -1.08
vertex -0.51 0.59 -1.13
endloop
endfacet
facet normal 0.05 0.86 0.5
outer loop
vertex -6.44 -0.03 0.56
vertex -0.51 0.59 -1.13
vertex -6.46 -0 0.52
endloop
endfacet
facet normal -0.67 -0.73 -0.14
outer loop
vertex -0.45 -0.76 0.76
vertex -3.06 1.3 2.56
vertex -3.07 1.32 2.52
endloop
endfacet
facet normal -0.67 -0.73 -0.14
outer loop
vertex -0.45 -0.76 0.76
vertex -3.07 1.32 2.52
vertex -0.46 -0.74 0.72
endloop
endfacet
facet normal 0.57 -0 0.82
outer loop
vertex -0.46 -0.74 0.72
vertex -3.07 1.32 2.52
vertex -3.09 1.28 2.54
endloop
endfacet
facet normal 0.57 -0 0.82
outer loop
vertex -0.46 -0.74 0.72
vertex -3.09 1.28 2.54
vertex -0.49 -0.78 0.73
endloop
endfacet
facet normal 0.1 0.73 -0.68
outer loop
vertex -0.49 -0.78 0.73
vertex -3.09 1.28 2.54
vertex -3.06 1.3 2.56
endloop
endfacet
facet normal 0.1 0.73 -0.68
outer loop
vertex -0.49 -0.78 0.73
vertex -3.06 1.3 2.56
vertex -0.45 -0.76 0.76
endloop
endfacet
facet normal -0.17 -0.87 0.47
outer loop
vertex -6.69 -0 -4.43
vertex 7.91 -0 0.87
vertex 7.89 0.03 0.91
endloop
endfacet
facet normal -0.17 -0.87 0.47
outer loop
vertex -6.69 -0 -4.43
vertex 7.89 0.03 0.91
vertex -6.71 0.03 -4.39
endloop
endfacet
facet normal 0.34 -0 -0.94
outer loop
vertex -6.71 0.03 -4.39
vertex 7.89 0.03 0.91
vertex 7.89 -0.03 0.91
endloop
endfacet
facet normal 0.34 -0 -0.94
outer loop
vertex -6.71 0.03 -4.39
vertex 7.89 -0.03 0.91
vertex -6.71 -0.03 -4.39
endloop
endfacet
facet normal -0.17 0.87 0.47
outer loop
vertex -6.71 -0.03 -4.39
vertex 7.89 -0.03 0.91
vertex 7.91 -0 0.87
endloop
endfacet
facet normal -0.17 0.87 0.47
outer loop
vertex -6.71 -0.03 -4.39
vertex 7.91 -0 0.87
vertex -6.69 -0 -4.43
endloop
endfacet
facet normal -0.4 -0.74 -0.54
outer loop
vertex -0.43 -2.55 -0.61
vertex -4.24 -0.21 -0.98
vertex -4.23 -0.19 -1.03
endloop
endfacet
facet normal -0.4 -0.74 -0.54
outer loop
vertex -0.43 -2.55 -0.61
vertex -4.23 -0.19 -1.03
vertex -0.41 -2.53 -0.65
endloop
endfacet
facet normal -0.1 -0 1
outer loop
vertex -0.41 -2.53 -0.65
vertex -4.23 -0.19 -1.03
vertex -4.25 -0.23 -1.03
endloop
endfacet
facet normal -0.1 -0 1
outer loop
vertex -0.41 -2.53 -0.65
vertex -4.25 -0.23 -1.03
vertex -0.44 -2.57 -0.66
endloop
endfacet
facet normal 0.5 0.74 -0.45
outer loop
vertex -0.44 -2.57 -0.66
vertex -4.25 -0.23 -1.03
vertex -4.24 -0.21 -0.98
endloop
endfacet
facet normal 0.5 0.74 -0.45
outer loop
vertex -0.44 -2.57 -0.66
vertex -4.24 -0.21 -0.98
vertex -0.43 -2.55 -0.61
endloop
endfacet
facet normal -0.16 -0.86 -0.49
outer loop
vertex 6.1 -0 0.03
vertex -3.91 1.61 0.41
vertex -3.91 1.63 0.37
endloop
endfacet
facet normal -0.16 -0.86 -0.49
outer loop
vertex 6.1 -0 0.03
vertex -3.91 1.63 0.37
vertex 6.1 0.03 -0.02
endloop
endfacet
facet normal 0.04 -0 1
outer loop
vertex 6.1 0.03 -0.02
vertex -3.91 1.63 0.37
vertex -3.92 1.58 0.37
endloop
endfacet
facet normal 0.04 -0 1
outer loop
vertex 6.1 0.03 -0.02
vertex -3.92 1.58 0.37
vertex 6.1 -0.03 -0.01
endloop
endfacet
facet normal 0.12 0.86 -0.5
outer loop
vertex 6.1 -0.03 -0.01
vertex -3.92 1.58 0.37
vertex -3.91 1.61 0.41
endloop
endfacet
facet normal 0.12 0.86 -0.5
outer loop
vertex 6.1 -0.03 -0.01
vertex -3.91 1.61 0.41
vertex 6.1 -0 0.03
endloop
endfacet
facet normal -0.46 -0.86 -0.22
outer loop
vertex -0.59 -1.19 -2.51
vertex -3.47 -0.58 1.18
vertex -3.5 -0.56 1.15
endloop
endfacet
facet normal -0.46 -0.86 -0.22
outer loop
vertex -0.59 -1.19 -2.51
vertex -3.5 -0.56 1.15
vertex -0.62 -1.16 -2.54
endloop
endfacet
facet normal 0.79 -0 0.62
outer loop
vertex -0.62 -1.16 -2.54
vertex -3.5 -0.56 1.15
vertex -3.51 -0.61 1.16
endloop
endfacet
facet normal 0.79 -0 0.62
outer loop
vertex -0.62 -1.16 -2.54
vertex -3.51 -0.61 1.16
vertex -0.63 -1.21 -2.53
endloop
endfacet
facet normal -0.33 0.86 -0.4
outer loop
vertex -0.63 -1.21 -2.53
vertex -3.51 -0.61 1.16
vertex -3.47 -0.58 1.18
endloop
endfacet
facet normal -0.33 0.86 -0.4
outer loop
vertex -0.63 -1.21 -2.53
vertex -3.47 -0.58 1.18
vertex -0.59 -1.19 -2.51
endloop
endfacet
facet normal 0.02 -0.86 0.5
outer loop
vertex -3.5 -0.58 1.13
vertex -0.47 -0.76 0.71
vertex -0.46 -0.73 0.75
endloop
endfacet
facet normal 0.02 -0.86 0.5
outer loop
vertex -3.5 -0.58 1.13
vertex -0.46 -0.73 0.75
vertex -3.49 -0.56 1.18
endloop
endfacet
facet normal -0.14 -0 -0.99
outer loop
vertex -3.49 -0.56 1.18
vertex -0.46 -0.73 0.75
vertex -0.47 -0.79 0.75
endloop
endfacet
facet normal -0.14 -0 -0.99
outer loop
vertex -3.49 -0.56 1.18
vertex -0.47 -0.79 0.75
vertex -3.49 -0.61 1.18
endloop
endfacet
facet normal 0.12 0.86 0.49
outer loop
vertex -3.49 -0.61 1.18
vertex -0.47 -0.79 0.75
vertex -0.47 -0.76 0.71
endloop
endfacet
facet normal 0.12 0.86 0.49
outer loop
vertex -3.49 -0.61 1.18
vertex -0.47 -0.76 0.71
vertex -3.5 -0.58 1.13
endloop
endfacet
facet normal -0.13 -0.86 0.49
outer loop
vertex -7.55 -0 2.02
vertex -1.72 -0.58 2.58
vertex -1.72 -0.56 2.62
endloop
endfacet
facet normal -0.13 -0.86 0.49
outer loop
vertex -7.55 -0 2.02
vertex -1.72 -0.56 2.62
vertex -7.55 0.03 2.07
endloop
endfacet
facet normal 0.1 -0 -1
outer loop
vertex -7.55 0.03 2.07
vertex -1.72 -0.56 2.62
vertex -1.72 -0.61 2.62
endloop
endfacet
facet normal 0.1 -0 -1
outer loop
vertex -7.55 0.03 2.07
vertex -1.72 -0.61 2.62
vertex -7.55 -0.03 2.06
endloop
endfacet
facet normal 0.04 0.86 0.51
outer loop
vertex -7.55 -0.03 2.06
vertex -1.72 -0.61 2.62
vertex -1.72 -0.58 2.58
endloop
endfacet
facet normal 0.04 0.86 0.51
outer loop
vertex -7.55 -0.03 2.06
vertex -1.72 -0.58 2.58
vertex -7.55 -0 2.02
endloop
endfacet
facet normal 0.35 -0.87 0.35
outer loop
vertex 6.98 -0 -0.02
vertex 7.88 -0 -0.92
vertex 7.91 0.03 -0.89
endloop
endfacet
facet normal 0.35 -0.87 0.35
outer loop
vertex 6.98 -0 -0.02
vertex 7.91 0.03 -0.89
vertex 7.01 0.03 0.01
endloop
endfacet
facet normal -0.71 -0 -0.71
outer loop
vertex 7.01 0.03 0.01
vertex 7.91 0.03 -0.89
vertex 7.91 -0.03 -0.89
endloop
endfacet
facet normal -0.71 -0 -0.71
outer loop
vertex 7.01 0.03 0.01
vertex 7.91 -0.03 -0.89
vertex 7.01 -0.03 0.01
endloop
endfacet
facet normal 0.35 0.87 0.35
outer loop
vertex 7.01 -0.03 0.01
vertex 7.91 -0.03 -0.89
vertex 7.88 -0 -0.92
endloop
endfacet
facet normal 0.35 0.87 0.35
outer loop
vertex 7.01 -0.03 0.01
vertex 7.88 -0 -0.92
vertex 6.98 -0 -0.02
endloop
endfacet
facet normal -0.1 -0.86 -0.49
outer loop
vertex 6.11 -0 0.03
vertex -1.71 -0.58 2.64
vertex -1.73 -0.56 2.59
endloop
endfacet
facet normal -0.1 -0.86 -0.49
outer loop
vertex 6.11 -0 0.03
vertex -1.73 -0.56 2.59
vertex 6.09 0.03 -0.01
endloop
endfacet
facet normal 0.32 -0 0.95
outer loop
vertex 6.09 0.03 -0.01
vertex -1.73 -0.56 2.59
vertex -1.72 -0.61 2.59
endloop
endfacet
facet normal 0.32 -0 0.95
outer loop
vertex 6.09 0.03 -0.01
vertex -1.72 -0.61 2.59
vertex 6.1 -0.03 -0.01
endloop
endfacet
facet normal -0.22 0.86 -0.46
outer loop
vertex 6.1 -0.03 -0.01
vertex -1.72 -0.61 2.59
vertex -1.71 -0.58 2.64
endloop
endfacet
facet normal -0.22 0.86 -0.46
outer loop
vertex 6.1 -0.03 -0.01
vertex -1.71 -0.58 2.64
vertex 6.11 -0 0.03
endloop
endfacet
facet normal 0.04 -0.86 -0.51
outer loop
vertex 7.89 -0 -0.87
vertex -1.19 0.93 -3.22
vertex -1.18 0.96 -3.27
endloop
endfacet
facet normal 0.04 -0.86 -0.51
outer loop
vertex 7.89 -0 -0.87
vertex -1.18 0.96 -3.27
vertex 7.91 0.03 -0.91
endloop
endfacet
facet normal -0.25 -0 0.97
outer loop
vertex 7.91 0.03 -0.91
vertex -1.18 0.96 -3.27
vertex -1.18 0.91 -3.27
endloop
endfacet
facet normal -0.25 -0 0.97
outer loop
vertex 7.91 0.03 -0.91
vertex -1.18 0.91 -3.27
vertex 7.9 -0.03 -0.92
endloop
endfacet
facet normal 0.21 0.86 -0.46
outer loop
vertex 7.9 -0.03 -0.92
vertex -1.18 0.91 -3.27
vertex -1.19 0.93 -3.22
endloop
endfacet
facet normal 0.21 0.86 -0.46
outer loop
vertex 7.9 -0.03 -0.92
vertex -1.19 0.93 -3.22
vertex 7.89 -0 -0.87
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.44 -0 -2.08
vertex 7.91 -0 0.87
vertex 7.9 0.03 0.91
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.44 -0 -2.08
vertex 7.9 0.03 0.91
vertex -6.45 0.03 -2.04
endloop
endfacet
facet normal 0.2 -0 -0.98
outer loop
vertex -6.45 0.03 -2.04
vertex 7.9 0.03 0.91
vertex 7.9 -0.03 0.91
endloop
endfacet
facet normal 0.2 -0 -0.98
outer loop
vertex -6.45 0.03 -2.04
vertex 7.9 -0.03 0.91
vertex -6.45 -0.03 -2.04
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -6.45 -0.03 -2.04
vertex 7.9 -0.03 0.91
vertex 7.91 -0 0.87
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -6.45 -0.03 -2.04
vertex 7.91 -0 0.87
vertex -6.44 -0 -2.08
endloop
endfacet
facet normal 0.12 -0.86 0.5
outer loop
vertex -7.56 -0 3.12
vertex -3.51 -0.58 1.13
vertex -3.48 -0.56 1.17
endloop
endfacet
facet normal 0.12 -0.86 0.5
outer loop
vertex -7.56 -0 3.12
vertex -3.48 -0.56 1.17
vertex -7.54 0.03 3.16
endloop
endfacet
facet normal -0.44 -0 -0.9
outer loop
vertex -7.54 0.03 3.16
vertex -3.48 -0.56 1.17
vertex -3.49 -0.61 1.18
endloop
endfacet
facet normal -0.44 -0 -0.9
outer loop
vertex -7.54 0.03 3.16
vertex -3.49 -0.61 1.18
vertex -7.55 -0.03 3.16
endloop
endfacet
facet normal 0.32 0.86 0.4
outer loop
vertex -7.55 -0.03 3.16
vertex -3.49 -0.61 1.18
vertex -3.51 -0.58 1.13
endloop
endfacet
facet normal 0.32 0.86 0.4
outer loop
vertex -7.55 -0.03 3.16
vertex -3.51 -0.58 1.13
vertex -7.56 -0 3.12
endloop
endfacet
facet normal 0.19 -0.84 -0.5
outer loop
vertex 7.9 -0 0.93
vertex -1.99 -2.39 1.17
vertex -2 -2.37 1.12
endloop
endfacet
facet normal 0.19 -0.84 -0.5
outer loop
vertex 7.9 -0 0.93
vertex -2 -2.37 1.12
vertex 7.89 0.03 0.89
endloop
endfacet
facet normal 0.02 -0 1
outer loop
vertex 7.89 0.03 0.89
vertex -2 -2.37 1.12
vertex -1.99 -2.42 1.12
endloop
endfacet
facet normal 0.02 -0 1
outer loop
vertex 7.89 0.03 0.89
vertex -1.99 -2.42 1.12
vertex 7.91 -0.03 0.88
endloop
endfacet
facet normal -0.22 0.84 -0.5
outer loop
vertex 7.91 -0.03 0.88
vertex -1.99 -2.42 1.12
vertex -1.99 -2.39 1.17
endloop
endfacet
facet normal -0.22 0.84 -0.5
outer loop
vertex 7.91 -0.03 0.88
vertex -1.99 -2.39 1.17
vertex 7.9 -0 0.93
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.44 -0 -2.63
vertex 6.11 -0 -0.03
vertex 6.1 0.03 0.01
endloop
endfacet
facet normal -0.1 -0.87 0.49
outer loop
vertex -6.44 -0 -2.63
vertex 6.1 0.03 0.01
vertex -6.45 0.03 -2.59
endloop
endfacet
facet normal 0.2 -0 -0.98
outer loop
vertex -6.45 0.03 -2.59
vertex 6.1 0.03 0.01
vertex 6.1 -0.03 0.01
endloop
endfacet
facet normal 0.2 -0 -0.98
outer loop
vertex -6.45 0.03 -2.59
vertex 6.1 -0.03 0.01
vertex -6.45 -0.03 -2.59
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -6.45 -0.03 -2.59
vertex 6.1 -0.03 0.01
vertex 6.11 -0 -0.03
endloop
endfacet
facet normal -0.1 0.87 0.49
outer loop
vertex -6.45 -0.03 -2.59
vertex 6.11 -0 -0.03
vertex -6.44 -0 -2.63
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -7.55 -0 -0.03
vertex 7.9 -0 -0.03
vertex 7.9 0.03 0.01
endloop
endfacet
facet normal 0 -0.87 0.5
outer loop
vertex -7.55 -0 -0.03
vertex 7.9 0.03 0.01
vertex -7.55 0.03 0.01
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -7.55 0.03 0.01
vertex 7.9 0.03 0.01
vertex 7.9 -0.03 0.01
endloop
endfacet
facet normal 0 -0 -1
outer loop
vertex -7.55 0.03 0.01
vertex 7.9 -0.03 0.01
vertex -7.55 -0.03 0.01
endloop
endfacet
facet normal -0 0.87 0.5
outer loop
vertex -7.55 -0.03 0.01
vertex 7.9 -0.03 0.01
vertex 7.9 -0 -0.03
endloop
endfacet
facet normal 0 0.87 0.5
outer loop
vertex -7.55 -0.03 0.01
vertex 7.9 -0 -0.03
vertex -7.55 -0 -0.03
endloop
endfacet
facet normal -0.19 -0.86 -0.48
outer loop
vertex 7 -0 0.93
vertex -3.07 1.3 2.57
vertex -3.07 1.33 2.52
endloop
endfacet
facet normal -0.19 -0.86 -0.48
outer loop
vertex 7 -0 0.93
vertex -3.07 1.33 2.52
vertex 7 0.03 0.88
endloop
endfacet
facet normal 0.16 -0 0.99
outer loop
vertex 7 0.03 0.88
vertex -3.07 1.33 2.52
vertex -3.08 1.27 2.53
endloop
endfacet
facet normal 0.16 -0 0.99
outer loop
vertex 7 0.03 0.88
vertex -3.08 1.27 2.53
vertex 6.99 -0.03 0.89
endloop
endfacet
facet normal 0.03 0.86 -0.51
outer loop
vertex 6.99 -0.03 0.89
vertex -3.08 1.27 2.53
vertex -3.07 1.3 2.57
endloop
endfacet
facet normal 0.03 0.86 -0.51
outer loop
vertex 6.99 -0.03 0.89
vertex -3.07 1.3 2.57
vertex 7 -0 0.93
endloop
endfacet
facet normal 0.26 -0.82 -0.51
outer loop
vertex 7 -0 -0.87
vertex -0.43 -2.55 -0.61
vertex -0.44 -2.53 -0.65
endloop
endfacet
facet normal 0.26 -0.82 -0.51
outer loop
vertex 7 -0 -0.87
vertex -0.44 -2.53 -0.65
vertex 6.99 0.02 -0.91
endloop
endfacet
facet normal 0.03 -0 1
outer loop
vertex 6.99 0.02 -0.91
vertex -0.44 -2.53 -0.65
vertex -0.42 -2.58 -0.66
endloop
endfacet
facet normal 0.03 -0 1
outer loop
vertex 6.99 0.02 -0.91
vertex -0.42 -2.58 -0.66
vertex 7.01 -0.02 -0.92
endloop
endfacet
facet normal -0.3 0.82 -0.49
outer loop
vertex 7.01 -0.02 -0.92
vertex -0.42 -2.58 -0.66
vertex -0.43 -2.55 -0.61
endloop
endfacet
facet normal -0.3 0.82 -0.49
outer loop
vertex 7.01 -0.02 -0.92
vertex -0.43 -2.55 -0.61
vertex 7 -0 -0.87
endloop
endfacet
facet normal -0.51 -0.77 0.39
outer loop
vertex -6.44 -0 -0.03
vertex -1.99 -2.39 1.11
vertex -1.99 -2.37 1.16
endloop
endfacet
facet normal -0.51 -0.77 0.39
outer loop
vertex -6.44 -0 -0.03
vertex -1.99 -2.37 1.16
vertex -6.44 0.02 0.02
endloop
endfacet
facet normal 0.25 -0 -0.97
outer loop
vertex -6.44 0.02 0.02
vertex -1.99 -2.37 1.16
vertex -2.01 -2.41 1.15
endloop
endfacet
facet normal 0.25 -0 -0.97
outer loop
vertex -6.44 0.02 0.02
vertex -2.01 -2.41 1.15
vertex -6.47 -0.02 0.01
endloop
endfacet
facet normal 0.26 0.77 0.58
outer loop
vertex -6.47 -0.02 0.01
vertex -2.01 -2.41 1.15
vertex -1.99 -2.39 1.11
endloop
endfacet
facet normal 0.26 0.77 0.58
outer loop
vertex -6.47 -0.02 0.01
vertex -1.99 -2.39 1.11
vertex -6.44 -0 -0.03
endloop
endfacet
facet normal -0.86 -0.31 -0.41
outer loop
vertex -4.22 -0.21 -1.03
vertex -3.77 -2.39 -0.33
vertex -3.79 -2.39 -0.28
endloop
endfacet
facet normal -0.86 -0.31 -0.41
outer loop
vertex -4.22 -0.21 -1.03
vertex -3.79 -2.39 -0.28
vertex -4.24 -0.2 -0.98
endloop
endfacet
facet normal 0.84 -0 -0.54
outer loop
vertex -4.24 -0.2 -0.98
vertex -3.79 -2.39 -0.28
vertex -3.82 -2.4 -0.32
endloop
endfacet
facet normal 0.84 -0 -0.54
outer loop
vertex -4.24 -0.2 -0.98
vertex -3.82 -2.4 -0.32
vertex -4.27 -0.22 -1.02
endloop
endfacet
facet normal 0.01 0.31 0.95
outer loop
vertex -4.27 -0.22 -1.02
vertex -3.82 -2.4 -0.32
vertex -3.77 -2.39 -0.33
endloop
endfacet
facet normal 0.01 0.31 0.95
outer loop
vertex -4.27 -0.22 -1.02
vertex -3.77 -2.39 -0.33
vertex -4.22 -0.21 -1.03
endloop
endfacet
facet normal 0.13 -0.86 0.48
outer loop
vertex -3.5 -0.58 1.13
vertex 7.89 -0 -0.93
vertex 7.9 0.03 -0.89
endloop
endfacet
facet normal 0.13 -0.86 0.48
outer loop
vertex -3.5 -0.58 1.13
vertex 7.9 0.03 -0.89
vertex -3.49 -0.56 1.18
endloop
endfacet
facet normal -0.18 -0 -0.98
outer loop
vertex -3.49 -0.56 1.18
vertex 7.9 0.03 -0.89
vertex 7.9 -0.03 -0.89
endloop
endfacet
facet normal -0.18 -0 -0.98
outer loop
vertex -3.49 -0.56 1.18
vertex 7.9 -0.03 -0.89
vertex -3.49 -0.61 1.18
endloop
endfacet
facet normal 0.05 0.86 0.5
outer loop
vertex -3.49 -0.61 1.18
vertex 7.9 -0.03 -0.89
vertex 7.89 -0 -0.93
endloop
endfacet
facet normal 0.05 0.86 0.5
outer loop
vertex -3.49 -0.61 1.18
vertex 7.89 -0 -0.93
vertex -3.5 -0.58 1.13
endloop
endfacet
facet normal -0.7 -0.71 0.02
outer loop
vertex -4.22 -0.21 -1.03
vertex -1.97 -2.39 1.12
vertex -1.99 -2.37 1.16
endloop
endfacet
facet normal -0.7 -0.71 0.02
outer loop
vertex -4.22 -0.21 -1.03
vertex -1.99 -2.37 1.16
vertex -4.24 -0.19 -0.99
endloop
endfacet
facet normal 0.69 -0 -0.72
outer loop
vertex -4.24 -0.19 -0.99
vertex -1.99 -2.37 1.16
vertex -2.01 -2.41 1.14
endloop
endfacet
facet normal 0.69 -0 -0.72
outer loop
vertex -4.24 -0.19 -0.99
vertex -2.01 -2.41 1.14
vertex -4.26 -0.23 -1.01
endloop
endfacet
facet normal 0.01 0.71 0.7
outer loop
vertex -4.26 -0.23 -1.01
vertex -2.01 -2.41 1.14
vertex -1.97 -2.39 1.12
endloop
endfacet
facet normal 0.01 0.71 0.7
outer loop
vertex -4.26 -0.23 -1.01
vertex -1.97 -2.39 1.12
vertex -4.22 -0.21 -1.03
endloop
endfacet
facet normal 0.3 -0.83 0.46
outer loop
vertex -7.55 -0 3.12
vertex -3.08 1.3 2.51
vertex -3.08 1.32 2.56
endloop
endfacet
facet normal 0.3 -0.83 0.46
outer loop
vertex -7.55 -0 3.12
vertex -3.08 1.32 2.56
vertex -7.56 0.02 3.17
endloop
endfacet
facet normal -0.13 -0 -0.99
outer loop
vertex -7.56 0.02 3.17
vertex -3.08 1.32 2.56
vertex -3.06 1.28 2.55
endloop
endfacet
facet normal -0.13 -0 -0.99
outer loop
vertex -7.56 0.02 3.17
vertex -3.06 1.28 2.55
vertex -7.54 -0.02 3.16
endloop
endfacet
facet normal -0.17 0.83 0.53
outer loop
vertex -7.54 -0.02 3.16
vertex -3.06 1.28 2.55
vertex -3.08 1.3 2.51
endloop
endfacet
facet normal -0.17 0.83 0.53
outer loop
vertex -7.54 -0.02 3.16
vertex -3.08 1.3 2.51
vertex -7.55 -0 3.12
endloop
endfacet
facet normal 0.07 -0.74 0.67
outer loop
vertex -6.47 -0 2.58
vertex -3.81 -2.39 -0.33
vertex -3.77 -2.37 -0.31
endloop
endfacet
facet normal 0.07 -0.74 0.67
outer loop
vertex -6.47 -0 2.58
vertex -3.77 -2.37 -0.31
vertex -6.43 0.02 2.6
endloop
endfacet
facet normal -0.74 -0 -0.67
outer loop
vertex -6.43 0.02 2.6
vertex -3.77 -2.37 -0.31
vertex -3.79 -2.42 -0.29
endloop
endfacet
facet normal -0.74 -0 -0.67
outer loop
vertex -6.43 0.02 2.6
vertex -3.79 -2.42 -0.29
vertex -6.45 -0.02 2.62
endloop
endfacet
facet normal 0.67 0.74 0.01
outer loop
vertex -6.45 -0.02 2.62
vertex -3.79 -2.42 -0.29
vertex -3.81 -2.39 -0.33
endloop
endfacet
facet normal 0.67 0.74 0.01
outer loop
vertex -6.45 -0.02 2.62
vertex -3.81 -2.39 -0.33
vertex -6.47 -0 2.58
endloop
endfacet
facet normal -0.36 -0.71 -0.61
outer loop
vertex 1.3 0.54 0.95
vertex -2.13 3.06 0.04
vertex -2.11 3.08 0
endloop
endfacet
facet normal -0.36 -0.71 -0.61
outer loop
vertex 1.3 0.54 0.95
vertex -2.11 3.08 0
vertex 1.33 0.56 0.91
endloop
endfacet
facet normal -0.26 -0 0.97
outer loop
vertex 1.33 0.56 0.91
vertex -2.11 3.08 0
vertex -2.14 3.04 -0
endloop
endfacet
facet normal -0.26 -0 0.97
outer loop
vertex 1.33 0.56 0.91
vertex -2.14 3.04 -0
vertex 1.3 0.52 0.91
endloop
endfacet
facet normal 0.61 0.71 -0.35
outer loop
vertex 1.3 0.52 0.91
vertex -2.14 3.04 -0
vertex -2.13 3.06 0.04
endloop
endfacet
facet normal 0.61 0.71 -0.35
outer loop
vertex 1.3 0.52 0.91
vertex -2.13 3.06 0.04
vertex 1.3 0.54 0.95
endloop
endfacet
endsolid brain
```

<!-- brain-stl:end -->

### How it is wired

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
- To put a newer champion in the 3D viewer: `python tools/export_brain3d.py best_evo_XXXXX.npz --readme`
  (rewrites `brain3d/brain.json`, `brain3d/champion.stl` and the model in this README). To view the page locally,
  run `python -m http.server` in the repo folder and open `http://localhost:8000/brain3d/`.

---

## Files

| File | Purpose |
|---|---|
| [`evo_rule.py`](evo_rule.py) | **The experiment**: genome, continuous game, pair selection, mutation, crossover, exams |
| [`brain_ttt_gpu.py`](brain_ttt_gpu.py), [`brain_ttt.py`](brain_ttt.py) | Shared game logic and the earlier learning-rule experiments |
| [`perfect.py`](perfect.py) | Perfect minimax player as a lookup table (exam only) |
| [`battle.py`](battle.py) | Replays a saved brain vs itself or vs the perfect player, recording every neuron every tick |
| [`dashboard.py`](dashboard.py) + [`dashboard.html`](dashboard.html) | Live training dashboard and battle viewer |
| [`brain3d/`](brain3d) | The interactive 3D brain: a static page (three.js) that draws the champion and runs it live in the browser |
| [`tools/export_brain3d.py`](tools/export_brain3d.py) | Exports a saved brain to `brain3d/brain.json`, lays it out with the rubber-band rule, and builds the STL model |
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
