# AI Brain: design notes

A fully connected recurrent "brain" that plays tic-tac-toe, lives continuously,
learns from outcomes during its life, and is replaced by evolution when it dies.

Guiding principle: **we provide primitives, never the plan.** Anything that can be
a gene is a gene. Anything the brain could infer, it must infer.

## The brain

- Nodes: 18 board sensors (9 "mine" + 9 "theirs", relative to the brain's own side),
  82 hidden nodes, 9 output nodes (one per cell). 109 total, every node connected
  to every node (109 x 109 weights).
- Ticks continuously: `x = tanh(W x + b + r * m)`. Board sensors are overwritten
  each tick with the real board. Every game starts from a blank brain: all
  activations and stored numbers are zeroed at game start (2026-11-09; before
  this only the stored numbers were zeroed and activations leaked from one game
  into the next, which added noise to the opening move).
- A move happens when an output crosses the firing threshold; the strongest
  firing output is the cell. Firing on an occupied cell is an illegal move.
  Not firing within the tick limit is a freeze.
- Stored number (superseded 2026-11-09, see "The merged node" below): every node
  had a second, slow value `s` surviving tick to tick, entering the node as
  `r * s`, updated as `s = tanh((d0 + G x) * s + M x)`. In the evolution-only
  brain this is folded into the node itself: `x = tanh(d0 x + W x + b + noise)`.
- No special "pain" or "pleasure" nodes. The outcome reaches the brain only
  through the learning rule (below). The brain must infer "new game" and
  "I move first/second" from the board alone: an empty board with zero "theirs"
  cells means first; exactly one "theirs" cell means second.

## Learning: backpropagation through time from the fired output

The key idea (2026-11-09). The brain is fully connected, so there are no layers to
backpropagate through. But there is time. The output that fired at tick T was
computed from every node at tick T-1, which were computed from every node at
T-2, and so on. Unrolled over the last 5 ticks, the recurrent brain is a 5-layer
feedforward network sharing one weight matrix.

When a decision is judged (illegal move, win, loss), an error signal starts at
the one output that fired: negative for illegal or loss, positive for a win. It
flows backward one tick per step, split among nodes by weight and activity,
for 5 ticks. Each connection changes by (sender activity at tick t-1) x (error
arriving at the receiver at tick t), summed over the 5 ticks, times that
connection's own learning-rate gene. Board sensors are clamped, so the wave stops
there.

**Credit by synaptic trace (2026-11-09, later).** The wave from the fired output
only ever reached the last decision, so a lost game could not teach the move that
actually lost it. Now every thinking tick leaves a trace: the output the brain is
currently leaning toward sends its wave back through the last 5 ticks, and that
wave is added to a per-synapse trace that fades by 0.9 each tick. Nothing is
applied yet. When the game is judged (win +1, loss -0.5, illegal -1, freeze +0.5
on the nearest output) every synapse moves by outcome x its trace, averaged over
the judged games. Traces are zeroed at the start of each game. No move is stored
and no one names the mistake: recency and strength set the weight, statistics do
the blaming. First effect: wins 50% -> 62%, losses 45% -> 31% within 12 games.
Legal moves themselves earn nothing (that reward turned out to teach "repeat
yourself" once legality was learned).

**Readout and seed (2026-11-09, evening).** Outputs saturate at exactly +-1, so
picking the cell by the largest activation was a coin toss among ties. The cell
is now the largest *pre-tanh* output, and the error seed is placed at that level:
+r on the chosen output, -r/8 on the other eight, so a judgement moves preference
rather than pushing every output up or down together. Peak 64% wins at step ~700.

**No credit decay (2026-11-09, night).** With traces fading 0.9 per tick and
about 3 ticks per move, the opening move received a quarter of the last move's
credit, and the opening never settled. The decay is a human prior about where
blame lives and it is wrong for tic-tac-toe (the first move decides most games)
and unknowable for harder tasks. Traces now do not fade within a game: every move
of a game receives the full outcome, and the averaging over 65k games does the
blaming.

**Reward-modulated Hebbian rule (2026-11-09, late; `--rule hebb`).** The user's
objection to the backward wave: in a brain, a synapse is rewarded for having
been active, not for a computed share of blame. So the alternative rule drops
the wave entirely. Every tick each connection adds (sender activity a tick
earlier) x (receiver activity now) to its trace; nothing fades within the game.
At game end every connection moves by (outcome - the brain's running average
outcome) x its trace, averaged over the 65k games. Subtracting the average is
the reward-prediction-error idea from dopamine neuroscience: a win is only
rewarding to the extent it was not already expected. Learns W and the memory
write matrix M; the other memory genes are frozen. A freeze counts as a loss.
Started from the backprop brain at step 1357 (kept as
`bp_checkpoint_0001357.npz`) at learning rate 0.001, since Hebbian traces are
several times larger than wave traces. The backprop run it replaced had just
started climbing on the second seat (57 -> 63% over 40 games).

First attempt used one running-average baseline over all games. That erased the
second seat within 40 games (53% -> 42%, mean weight 0.385 -> 0.293): first-seat
games win 80% so nearly all of them land above the average and push every
co-active connection up; second-seat games win 45% so nearly all land below and
pull theirs down, whoever actually won. Now each life's outcome is compared only
with the other 65k lives of the same game (same seat, same brain), and the whole
game is judged once at its end. The update is then purely the difference between
what winning and losing brains did.

Result (66 games from the step-1357 brain): first seat held at 91%, second seat
went 64% -> 44% wins with illegal moves 2% -> 18%, accelerating. Stopped; the
degraded saves were deleted and the step-1357 backprop brain restored as the
latest. Reading: co-activation says which connections were busy, not which ones
pushed toward the move, so the signal that separates winning from losing games
is buried under the shared activity of both. 65k games of averaging was not
enough to dig it out, and the second seat (the harder, noisier seat) drifted.

Third variant, the literal form: no baseline at all, win +1 / loss -0.5 /
illegal -1 times each connection's co-activation. Fastest collapse of the three:
in 12 games the first seat went 92% -> 83% and the second 64% -> 47%, with
illegal moves at 10-15% on both seats. Re-run for 250 games at the user's request: wins
fell to 20% on the second seat and ~55% on the first, with illegal moves at
55-65% and 35-40%. A plateau from game 80 to 160, then a second slide; it never
turned around. All three Hebbian variants degrade a brain that the backward wave
had built; none improved it.

**Neural noise (2026-11-09, night; `--noise`).** Why every activity-based rule
failed: the brain was deterministic. Same board, same move, every game. So the
difference between a won and a lost game came from the random opponent, not
from anything the brain did differently, and a rule that compares activity in
won vs lost games was learning about the opponent's luck. The backward wave
escaped this with a counterfactual ("the move you made in the lost game: less
of it"); an activity rule cannot. Real brains vary their own behaviour: every
neuron fires a little differently each time, and reward lands on the variations
that worked (birdsong learning is the textbook case). So now every hidden and
output node gets a fresh Gaussian kick (std 0.5) on its input sum every tick,
in training and in the dashboard battle alike; the width is saved with the
brain. Board sensors stay exact. Effect before learning: the step-1357 brain
opens on 7 different cells instead of 1, first seat 92% -> 78%, second
64% -> 54%. Run: Hebbian rule with the same-round baseline, lr 0.001, from
step 1357.

**Neural noise did not rescue the activity rule.** With noise 0.5 and the
same-round baseline: flat at lr 0.001, sliding into illegal moves at 0.005 and
0.002 (second seat 54% -> 32% wins, illegal 3% -> 42% over 246 games). Likely
cause: a game's trace grows with its length, so long (won) games dominate the
centred sum and what they share is "fire harder", not "the right move".

## Evolution only (2026-11-09, afternoon; `evo_rule.py`)

The user's conclusion after the Hebbian runs: every learning rule we wrote was
a human theory of credit, and the rule is the most designed part left. So the
rule itself becomes genes, along with everything else about the brain. Nothing
inside a life is written by hand; the only optimiser is selection.

One genome = one lineage. Fixed by tensor shapes: 18 sensors, 9 outputs, at most
82 hidden nodes. Genes:
- structure: live hidden nodes (<= 82), firing threshold, refractory ticks, tick
  limit before a freeze;
- birth: the initial weight matrix W0 and biases (a "genius" can be born),
  memory read gains, base multipliers, write and multiply matrices;
- noise: per-node std of the random kick on the input sum each tick;
- plasticity: trace *= lam each tick; trace += A pre post + B pre + C post + D
  + E pre xi (xi = the noise kick, the node-perturbation term); at game end
  dW = eta (mod (Rb - bl meanRb) + (1 - mod)) trace / ticks; W *= 1 - wd. Rb is
  the brain's OWN valuation of the outcome: genes r_win r_loss r_draw r_ill r_frz.
- mutation step for the matrices is itself a gene.

Selection uses the fixed external score (win +1, loss -0.5, illegal -1, draw 0,
freeze -0.5): that is the world, not the brain. 256 genomes x 256 brains =
65,536 brains; each brain starts life from its genome's W0 plus a tiny jitter,
plays 60 games (alternating seats) against the random mover, and changes only
through its plasticity genes. Fitness = mean external score over the last 20
games. Best quarter survives; the rest are mutated copies of survivors.
Generation 0 (5.5 min): the best lineage scored -0.53 by freezing in 94% of
games, since a freeze (-0.5) beats an illegal move (-1). The user had the world
pay for legal moves again (+0.3 per legal move, `--reward-legal`), so a lineage
that fires legally outscores one that never moves; the brain's own value of a
legal move is a gene too (r_legal).
Speed: 65k brains x 60 games ran at ~6 s/game (6 min per generation; each tick
touches every brain's 109 x 109 trace). Now 256 lineages x 128 brains, 40-game
lives (~1.5-2 min per generation). Breeding (user's request): the top 10% of
lineages survive unchanged and form the breeding pool; every other slot is a
child of two pool members, then mutated. Crossover is neuron-wise: each node
is inherited whole (bias, noise, memory genes and all incoming weights from the
same parent), because uniform per-synapse crossover scrambled unrelated nets
in the first evolution runs.
Generations 1-17 (~1.7 min each): best lineage from -0.20 to +0.28 (wins 6% -> 21%,
illegal 84% -> 54%), population mean from -0.39 to +0.12. The best rule genes
converged on full outcome modulation with a strong same-round baseline.

**The merged node (2026-11-09, 14:30).** The user's observation: the stored
number (persistent, decaying, written to) is what a node should be; the
recomputed-every-tick activation is the special case where nothing is kept.
And a node must not amplify itself (normal networks have no such thing): it
may keep, fade or forget what it holds, never grow it on its own. So the two
systems are one now. Each node has a single persistent value:

    x = tanh( d0 * x + W x + b + noise ),   d0 in [0, 1] per node (a gene)

d0 = 0 is a plain recomputed node, d0 near 1 holds its value across ticks.
The controlled gate G (an LSTM forget gate), the separate write matrix M and
the read gain r are gone. Genes now: W0, b, d0, per-node noise, structure
(live hidden nodes, threshold, refractory, tick limit), plasticity (A B C D E,
eta, mod, bl, lam, wd, the five outcome values and the legal-move value), and
the mutation step. The population was carried over (W0, b, d0, noise, rule
genes), so the 17 generations of selection were kept.

Generations 19-133 with the merged node (~2 min each): best lineage +0.26 ->
+1.25 (wins 19% -> 65%, illegal 62% -> 20%), population median +0.08 -> +1.08.
The leaders flattened around +1.22 from generation 105 on while the median kept
rising. Genes the top lineages agree on: refractory 8-9 ticks (from 3), threshold
~0.92, 67 of 82 hidden nodes alive, trace fading to about half each tick,
outcome-modulated plasticity with a strong same-round baseline, leak 0.5.

**Co-evolution (2026-11-09, evening; `--opponent peer`).** Brains now play each
other instead of the random mover. Every game the whole population is paired at
random (seats random), so no lineage farms its relatives. No free wins: if the
opponent makes an illegal move or freezes, the opponent is punished and the
survivor gets a draw plus its legal-move pay; a win must be earned on the board.
Fitness is the external score against peers. Because that number is not
comparable to the earlier generations, each generation also plays 6 exam games
against the random mover, logged but not scored. Resumed from the generation-133
population.

Co-evolution, generations 134-161 (~2.5 min each): peer fitness +0.64 -> +0.82,
median +0.52 -> +0.69, illegal vs peers 13% -> 8%; but the exam vs the random
mover stayed flat (~60% wins, 20-25% illegal). The population had converged:
every top lineage shared the same structure genes. User's fix: 30% of children
get a 5x stronger mutation on every gene (matrix, per-node and scalar), the rest
the normal step, so each generation keeps exploring outside the parents' cluster.

Strong mutation, generations 162-187: best lineage band lifted from +0.74..0.80
to +0.78..0.86, peer illegal 13% -> 6-8%, exam vs random still ~63% wins /
20-24% illegal. Draws vs peers 23-30%, but that figure mixed real full-board
draws with the survivor's draw after an opponent's blunder; from generation 188
the log separates them (`draw` = board full, `oppdied` = opponent made an illegal
move or froze).

**Perfect opponent (`perfect.py`, `--opponent perfect|mixed`).** A lookup table
(board code -> best cell) from depth-aware minimax: quickest win, slowest loss,
ties broken centre > corners > edges, so it opens in the centre, always blocks,
always finishes a line, never loses (self-test: perfect vs perfect draws; vs
random 2000 games per seat, 0 losses). Against it a brain can at best draw, so
fitness there is draws + legal-move pay. `mixed` gives each brain a perfect or a
random opponent, 50/50 per game. Not yet used for a run.

Why this is not the evolution that failed before: that evolved weights alone
(3.9 million births to 45% illegal) or gave evolution a learning-rate gene with
pretrained weights (it switched learning off). Here a lineage can win either by
being born good or by learning fast, and the search space for a rule is tiny.

**Weight decay (same).** Without it the largest weights pinned at the clamp (+-5)
and the brain oscillated, losing the 64% peak. Each game every weight shrinks by
0.1%; only connections that keep earning their weight stay large.

Details that mattered:
- Most nodes sit at exactly +-1 (memory drives them there) where tanh' is 0 and
  the wave would die instantly. The derivative is floored at 0.1.
- Every one of the 109 x 109 connections has its own learning-rate gene. Genes
  start positive (gradient direction) and may evolve negative or to zero.
  Random-sign genes made learning pure noise.
- Legal moves currently earn no reward; only outcomes do.

### Vision sensors and a continuous game (2026-11-09, evening)

Up to generation 203 the game opened a "move window" for the brain: on its turn
the brain ticked up to 30 times, could not fire during the first `refr` ticks,
and the window closed on its first fire. The board reached it as 18 sensors,
"mine" and "theirs" per cell. That window is a piece of human plan: it tells the
brain when to think and when a decision counts.

Now the board is treated like vision and the brain lives through the whole game:

- **Sensors: 29.** For each cell three nodes, empty / X / O (absolute, not
  relative to the player), plus two nodes saying which seat I am. 120 nodes in
  all: 29 sensors, up to 82 hidden, 9 outputs.
- **No move window.** The brain ticks every tick of the game, both on its own
  turn and while waiting. Any output above `thr` (outside a refractory period of
  `refr` ticks after each fire; `maxt` is gone) is a move: the cell with the
  strongest drive.
- **The game answers like a board.** A fire on a taken cell *or when it is not
  the brain's turn* is an illegal move and ends the game (the opponent gets a
  draw's score, OPPDIED, not a win). Sitting on one's own turn for 30 ticks is a
  freeze. The scripted opponent (random / perfect / mixed) answers after a random
  1..8 ticks, so the brain has to notice the board changing instead of counting
  ticks; peers simply play each other in the same stream, both ticking.
- **Fitness is unchanged:** win +1, loss -0.5, illegal -1, freeze -0.5, draw 0,
  +0.3 per legal move; mean over the last third of a 40-game life.

The population of generation 203 (vs the perfect player, 0% draws, 3% illegal)
carried over: hidden and output nodes kept their genes and the weights among
them; the sensor nodes and the weights out of them are fresh (`migrate_nodes`).
The first games of generation 204 were 88% illegal: almost every brain fires
again before the opponent has answered. Waiting is the first thing it has to
learn.

The live battle (`battle.py`) plays continuous brains the same way and names the
reason for an illegal move ("fired out of turn" / "played an occupied cell"). It
also records every node's activation after every tick for both brains, and the
dashboard replays the game tick by tick with a panel per brain: what it sees
(three mini boards: empty / X / O, plus the two seat nodes), the hidden nodes,
and what it wants (the nine outputs as a board, framed red above the firing
threshold). One colour rule for both brains: blue = positive, red = negative,
white = quiet, for nodes and for the lines (signal = source activation x weight).
Below it a memory panel in the same layout: a glow per node for what it carries
into the next tick (leak gene x value; blue/red by sign, larger the more it
holds) and a ring for how much its value changed this tick (green rising, orange
falling). Under each avatar a stack of its decisions: the output grid frozen at
the tick it fired, newest on top, older ones sliding down and fading.
A self-inflicted end (illegal move or freeze) shows "no winner"; nobody is
celebrated for the other side's mistake.

**Speed (same evening).** The GPU (RTX A6000) sat at 95% utilisation with 19 of
49 GB in use, so spare VRAM could not buy speed; the cost is memory traffic on
the 32768 x 120 x 120 trace and weight tensors (1.9 GB each). Two changes, with
bit-identical results on the same seed: the three trace terms are one rank-3
product `[u v e] . [new 1 xi]`, so the update is one multiply and one batched
matmul instead of eight passes; and once fewer than 45% of brains are still
playing, only their rows of W and T are touched (the loop runs until the last
brain's game ends, up to 400 ticks, while the average game is 20-30 ticks).
4 games at full scale: 14 s -> 10 s -> 8 s. The game log now prints ticks/game.

**Both worlds (later that evening, from generation 235).** Trained on the perfect
player alone (gens 189-234: 0% -> 18% draws, illegal 3% -> 15-28%), the brains
looked lost in the live self-play battle: they had only ever seen one opponent
and one line of play. `--opponent both` mixes the two worlds inside every game:
half of the brains, drawn at random each game, are paired with a peer, the other
half each play the perfect player (seat alternating with the game number).
Fitness is the mean over whatever a brain got. The GEN line reports the combined
outcome plus a `vs perfect:` split, and the dashboard shows both.

### Taking the game rules out of the brain (2026-11-10, from generation 253)

Audit of what was still hand-written *inside* the brain, as opposed to the world
it lives in. The world (opponent delay 1..8 ticks, 30-tick freeze, the scores,
no free win, the breeding program) may be arbitrary; the organism may not.
Three things were game rules in disguise and are gone:

- **The refractory timer** (`refr` gene, 0-10 ticks after a fire). It had
  evolved to the cap: brains waited by clock instead of by watching the board,
  because the perfect player always answered within 8 ticks. Now an output
  **spikes the tick it crosses its threshold from below** and cannot spike
  again until it has fallen back under it. Nothing else gates firing.
- **The sibling baseline** in the learning rule (`Rb - bl * mean over the
  lineage's 128 brains`). No brain knows how its siblings did. The learning
  signal is now the brain's own valuation of its own outcome, full stop.
- **The strongest-drive cell pick.** The output that crossed is the move; if
  several cross on the same tick, the highest.

Kept as physics: weight clamp +-5, hidden-node cap 82, tanh nodes with per-node
leak in [0,1] and Gaussian noise, the second-order Hebbian form of the rule with
every coefficient a gene, and the per-tick normalisation of the trace.

The goal, stated by the user: not a tic-tac-toe player, but a small brain that
understands the game. Resumed from the generation 252 population with the two
genes dropped; brains that relied on the timer double-fire and die out first.

**Peers only (from generation 255).** The perfect player is out of training: a
fixed, deterministic opponent lets brains fit its one line of play instead of
the game. Brains now only ever play each other (`--opponent peer`). The perfect
player and the random mover remain as exams after each life, six games each,
for the log and the dashboard only; they never select.

**Noise is never repeatable (generation 263 on).** The brains' noise stream is
seeded from OS entropy at every start; `--seed` now only drives breeding and
mutation (the world), and `--noise-seed` exists solely for a deliberate replay.
Within a run every node gets a fresh Gaussian kick every tick; this closes the
one gap where each resume used to reopen the stream at the same place.

**New world scores (generation 266 on, set by the user).** Ten generations of
peers-only play (255-265) sat in a do-nothing equilibrium: no board results, the
best genome freezing in ~95% of exam games, the threshold gene at 0.99. The
world's prices changed to make that untenable: win on the board +100000, legal
move +10, draw / opponent died 0, illegal move -1, loss on the board -10, freeze
-100000. A freeze now costs exactly what a win earns, a hundred thousand times a
wrong move, so sitting still is the worst thing a brain can do, and a single
board win outweighs everything else a lineage could earn. The brain itself is
unchanged.

**Reward only, never punish (generation 269 on, set by the user).** Two
generations under the punitive prices flipped the population from "never try" to
"always try, badly": freezes vanished, illegal moves went 94% -> 61%, and the
first board wins appeared. The user's conclusion: if punishment teaches
avoidance, use none. New prices: a legal move +1, a win on the board +100, and
every other event (loss, draw, illegal move, freeze, opponent died) exactly 0.
A mistake costs only the future it forecloses: the legal moves and the win the
brain could still have earned had the game continued. Selection is by rank, so
this ranks lineages purely by how much they accomplish, never by what they
avoid.

**Illegal move back to -1 (generation 286 on).** Fifteen generations of reward-only
scoring stayed in two-tick games: every brain fires at the first tick, makes one
legal move (+1) and then spikes the same cell again, which under reward-only
scoring cost nothing. The illegal move now costs -1 again, so marking the board
twice nets zero and a brain has to make its second move somewhere else.

### Run 2: from generation 0 (2026-11-10)

Run 1 (generations 0-291, archived in `runs/v1_gens0-291/` with its log) carried
every design change of the day on one population: random opponent, perfect
player, co-evolution, the vision sensors and continuous game, spiking outputs,
no sibling baseline, four price lists. The user called the concept settled and
asked for a clean start. Run 2 begins at generation 0 with the design as it
stands:

- 29 sensors (empty / X / O per cell + I am X / I am O), <= 82 hidden, 9 outputs;
  one value per node, `x = tanh(d0 x + Wx + b + noise)`; an output spikes the
  tick it crosses its threshold from below; nothing else gates it.
- The brain ticks through the whole game; the world: 30 ticks on your turn or
  you freeze, a fire out of turn or on a taken cell is illegal, 400-tick cap.
- Peers only. Exams vs the random mover and the perfect player for the log.
- Prices: legal move +1, win on the board +100, illegal move -1, everything else
  (loss, draw, freeze, opponent died) 0.
- Every gene a gene; nothing learned by hand. Noise seeded from entropy.

The starting population is broadly randomized, per the user ("randomized node
values for each"): every node's leak uniform in [0,1], noise std log-normal
around 0.3 with spread 0.6, bias std 0.15; trace decay lam 0.5 +- 0.3, threshold
0.6 +- 0.2, live hidden nodes 60 +- 15; weights N(0, 0.1). Previously lam and
n_hid started identical across the whole population.

**Survival selection (run 2, generation 8 on, from the user).** Seven generations
in, run 2 had converged on the waiting strategy exactly as run 1 had: the free
freeze made stillness and the double tap score the same. Rather than another
price list, selection itself changed to a tournament: a brain's fitness is the
share of its games it *survived*, where a win, a draw, or the opponent dying
first count as surviving and a loss, an illegal move or a freeze do not. The
top half of the lineages goes to the next generation unchanged (`--elite 0.5`)
and their children fill the other half. No prices at all (`--select survival`).

Why this is different: waiting is honoured as the smart move it is, as long as
the other side dies first. When two waiters meet, X's clock runs out first, so
the O waiter survives; an X that makes one move then forces O's clock to run,
so the X that moves survives against the O that waits; and so on up the
ladder. Each step up is forced by the step below, with nothing hand-priced.
Ties, which are common, are broken at random.

**Mutation x2.5 (run 2, generation 21 on).** With survival selection the top
half always goes through, so the population drifts slowly; the user asked for
stronger mutation overall. `--mutation` multiplies every mutation step (scalar
genes, per-node vectors, weight matrices). Set to 2.5. And instead of 30% of
children mutating 5x harder, 10% are "freaks": every gene mutated 20x harder
(times the 2.5), landing well outside the parents' neighbourhood.

**Random mover with a random delay (run 2, generation 25 on).** Seventeen
generations of the tournament among peers kept the same waiter-then-mover
lineage on top at 68-75% survival; peers die too easily to push the ladder.
The opponent is now the random legal mover, answering after a random 1..25
ticks (`--opponent random --opp-delay 25`) so there is no rhythm to learn.
Under survival selection a random mover never blunders, so surviving a game
means winning or drawing it: the first opponent that cannot be beaten by
waiting. The perfect player remains an exam.

**An attempt beats no attempt (run 2, generation 31 on, from the user).** Six
generations against the random mover under pure survival scoring: best 0.7%,
population mean 0.0%. Almost every brain scored exactly nothing every game, so
the top half was being chosen by the random tie-break; selection had gone
blind. Survival scoring now gives an illegal move 0.1 (`--reward-attempt`),
while a freeze and a loss stay at 0: surviving (1) > trying and failing (0.1) >
not trying (0). We are rewarding an illegal move, which is new; the point is
that a brain that acts is closer to a brain that acts correctly than one that
sits still.

**Patience pays (run 2, generation 33 on, from the user).** Waiting has to be
learned too, so survival scoring also credits 0.001 for every tick a brain is
alive while it is the opponent's turn (`--reward-wait`). Against the random
mover's 1..30 tick delay that is roughly 0.015 per opponent move: small next
to the 0.1 for an attempt and the 1 for surviving, enough to rank a brain that
holds still through the opponent's turn above one that fires into it.

**A legal move earns (run 2, generation 45 on).** Ten flat generations at the
attempt credit showed the gap: under survival scoring a legal move earned
nothing on its own, so "one legal move then an illegal one" ranked level with
"illegal at once". Each legal move now earns 1 (`--reward-legal 1`), the same
as surviving the game, so the ladder reads: freeze 0 < illegal 0.1 < one legal
move 1 < two 2 ... and a survived game adds 1 on top, with 0.001 per tick of
waiting throughout.

### Run 3: from generation 0 (2026-11-10, evening)

Run 2 (generations 0-45, archived in `runs/v2_gens0-45/`) carried the switch
from prices to the tournament, the move from peers to the random mover, and
the attempt / waiting / legal-move credits. The user asked for a clean start
with the rules as they now stand:

- Opponent: the random legal mover, answering after a random 1..30 ticks.
- Selection: tournament, reward only. Score per game = 100 for a win on the
  board + 1 per legal move + 0.1 for an illegal move + 0.001 per tick alive on
  the opponent's turn. A draw, a loss, a freeze or the opponent dying first
  earn nothing by themselves (the user: none of those should be rewarded).
  Fitness = mean over the last third of a 40-game life. Top half of the 256
  lineages carries over unchanged; children of the top half fill the rest,
  mutation x2.5, one child in ten a freak at x20 on top.
- Brain unchanged: 29 vision sensors, <= 82 hidden, 9 spiking outputs, one
  value per node with leak, noise and plasticity all genes; nothing timed or
  priced inside the brain. Starting population broadly randomized.
- Exams vs the random mover and the perfect player for the log; live battle
  alternates self-play and the perfect player.

**Top 10% (run 3, generation 32 on).** The surviving pool drops from the top
half to the top tenth of the lineages (`--elite 0.10`): 25 genomes carry over
unchanged and 231 slots are their children each generation. More of each
generation is new, and the pool that breeds is the one that actually scored.

**Bigger sample (run 3, generation 42 on).** Brains per genome raised from 128
to 512 (`--brains 512`): 131,072 brains per generation, so each lineage's
fitness averages over four times as many lives. At 65,536 brains the card used
13.6 of 49 GB; the fused trace update left far more headroom than the old code.
Generations take proportionally longer.

**No plasticity (run 3, generation 55 on, from the user).** The last hand-shaped
thing inside the brain was the within-life plasticity: every coefficient was a
gene, but the *form* of the rule (a second-order Hebbian trace, applied at game
end, scaled by the brain's own valuation genes) was mine. The user had assumed
it was already gone and wants it gone. Removed: the trace tensor, the weight
update, and the genes A B C D E, log_eta, mod, lam, wd and the six valuations.
A brain is now exactly its genome from its first tick to its last; the only
things in the genome are the wiring W0, the per-node bias, leak and noise, the
threshold, the live-node count and the weight-mutation step.

Memory stays: node values persist across ticks within a game through the leak
gene. Learning, in the sense of a brain changing during its life, is gone; all
adaptation is evolution.

Consequences: a life is one game (`--life 1`), since nothing a brain does in
game one changes game two; fitness is that game's score averaged over the
genome's 512 brains. Exams run every fifth generation (`--exam-every 5`) so
they do not dominate the run. Without the trace the tick costs a fraction of
what it did. Resumed from the generation 54 population with the removed genes
dropped; the evolved W0, biases, leaks, noise and thresholds carry over.

**Peers return, mixed with the random mover (run 3, generation 480 on).** Against
the random mover alone the best genome climbed from 8% to 49% wins in 300
generations (board losses 0%: it never loses, it kills itself), then flattened
around 48-51% for fifty generations. Peer play, which collapsed into waiting
back when nobody could make a legal move, is now worth having: a peer plays
back and keeps changing. `--opponent both --both-kind random`: every game half
the brains play a peer, half the random mover, so the pressure to move legally
and win stays while tactics against a real opponent are selected for. The
perfect-player exam continues every fifth generation.

**Peers only (run 3, from the user, a few generations later).** The random
mover came out of the mix: with freezing neither rewarded nor punished and a
legal move worth a point, waiting out an opponent's blunder earns nothing, so
the old collapse cannot recur. `--opponent peer`; the random mover and the
perfect player remain as exams every fifth generation.

**The brains had only ever played X (found at generation 556).** Switching to
peers collapsed the population in one generation: best from 56 to under 1,
every game over in two ticks, half the brains dying by an illegal move on tick
one. Cause: in solo play the seat came from the game index (`me = 1 if gi % 2
== 0 else 2`), and with one-game lives `gi` is always 0, so for 500 generations
every brain was X against the random mover and never once sat as O. Seated as
O against a peer they opened on tick one like an X and died out of turn. Fixed:
solo brains now draw a random seat each game, so exams and any scripted play
cover both seats. The pre-collapse population (generation 555) was still on
disk and is the resume point; the collapsed saves are in
`runs/v3_collapse_gens556-584/`. Note that run 3's "49% wins vs random" was
X-only play; O has to be learned from here.

**O has to be learned before peers can teach (generation 556, second attempt).**
Resuming the 555 population into peers only collapsed again at once: as O every
brain fires on tick one and dies, so each game is decided by who drew X, the
ranking is seat luck, and the X skill drifts away within a dozen generations.
Interim: back to the random mover, now with half the brains seated as O every
game, until the exam shows the brains playing O; then peers. (Autonomous call,
to stop the erosion; the user had asked for peers only and can override.)

### Run 4: from generation 0, peers only (2026-11-10, late evening)

Run 3 (archived in `runs/v3_gens0-561/`) is discarded: its 500 generations of
"progress" were X-only play. Run 4 starts clean with everything as it stands,
and the brains only ever play each other.

- Brain: 29 vision sensors, <= 82 hidden, 9 spiking outputs; one value per node
  with leak, bias and noise genes; weights fixed for life (no plasticity).
- World: continuous game, 30 ticks on your turn or you freeze, a fire out of
  turn or on a taken cell is illegal. Opponent: a peer, paired at random every
  game, seats random. No scripted opponent in training.
- Scoring: win on the board 100, each legal move 1, illegal move 0.1, each tick
  alive on the opponent's turn 0.001; draw, loss, freeze, opponent died: 0.
- Lives of one game; fitness = that game's score averaged over the genome's 512
  brains; top 10% of 256 lineages carry over, mutation x2.5, 10% freaks at x20.
- Exams vs the random mover and the perfect player every fifth generation, both
  seats, for the log only.

**Same-tick fires (run 4, fixed before generation 1 of the real start).** In a
fresh peer population everyone fires on tick one: X's fire is a legal move, O's
is out of turn. The game loop used to end the game on O's illegal fire before
applying X's move, so X got nothing for a move it had made; the whole
population scored ~0.07 and there was no gradient. Legal moves now land first
and the illegal fire ends the game afterwards (a board result on the same tick
outranks it). X keeps its point; O still dies. Run 4 proper starts from here.

**Pair selection (run 4 proper, from the user).** Selection is now head-to-head
inside each game (`--select pair`). Both brains get the reward-only score (win
100, legal move 1, illegal 0.1, 0.001 per waiting tick); the higher score in
the pair survives (1), the other does not (0), a tie splits (0.5). If X never
opened, that is, froze without making a move, the O brain is re-seated as X on
an empty board with its memory cleared and gets 30 ticks to make any move: pass
and O is the winner, fail and both are thrown out. A lineage's fitness is the
share of pairs its 512 brains won. Two waiters meeting therefore both die,
which is the point: no brain can survive by out-waiting another.

**Two fixes after 460 flat generations of run 4.** (1) The open test now runs
both ways: when O kills itself on tick one before X has moved, X has not shown
it can play either, so X must open within one turn to take the pair, else both
go. Without this O's 0.1 attempt credit beat a patient X's 0, and impulsive
O's spread while selection was otherwise a coin flip on who drew X. (2) The
noise gene has a physical bound, std in [0.001, 3], and the bias in [-5, 5],
like the weight clamp; under near-random selection the 20x freaks had
random-walked noise into the tens of thousands. Run 4 restarted from scratch.

**Run 4, 800 flat generations: weights unbounded, one coin flip per brain.** The
mean |W0| had grown from 0.1 to 58: the +-5 weight clamp had lived in the
plasticity update and left with it, so the weight genes had no bound and the
20x freaks random-walked them until every node saturated on tick one. Every
game then ended on tick one and every pair was decided by the seat draw, so
the "best" genome was a different random one each generation with 55-59% pair
wins, the sampling-noise band around 50%. Fixes: `WCLIP = 5` applied to the
weight genes in mutation. Lives stay at one game, per the user: 512 brains per
genome is already 512 pairs per lineage, enough to rank them; the plateau was
the saturated weights, not the sample size. Restarted from scratch.

**Illegal fires are ignored (run 4, from the user, after the loophole).** The
pair rule let a brain bank its lead by ending the game with an illegal move:
open, then fire on the same cell, game over, X wins every pair from the X seat
and the population converged on exactly that. New world rule: an illegal fire
(out of turn, or on a taken cell) does nothing and earns nothing; the board
ignores it and the game keeps ticking while the brain's own clock runs. A turn
ends only with a legal move. If the 30-tick clock runs out the brain times out
and the opponent takes the pair. Board results decide otherwise; a draw falls
back to the score. The first-move rule stays, per the user: if X times out
without ever opening, O is re-seated as X with a cleared memory and must move
within one turn (a move wins the pair, none and both are thrown out); any later
timeout is simply a loss for the brain whose clock ran out. The attempt credit
is 0. The game log now prints illegal fires
per brain per game. The live battle shows illegal fires briefly and plays on;
a timeout is a win for the other side.

**A cost per illegal fire (run 4, from the user).** With illegal fires ignored,
brains were making about 14 of them per game. Each one costs 0.001 in the
score (`--reward-attempt -0.001`, applied per fire). It was raised to 1 for
generations 65-110; over those generations timeouts rose from 10% to 14% and
board results fell while fires barely moved, which read as brains hesitating
rather than aiming, so the user put it back. Under pair
selection the board and the clock still decide a pair outright, so this cost
only separates brains whose game ends in a draw or otherwise on equal terms;
it is a nudge toward quieter outputs, not a punishment that ends anything.

**Dashboard for the brain-vs-brain world.** In a population that only plays
itself the pair-win share is relative (the mean is 50% by construction), so
the dashboard now leads with absolute measures. Tiles: Population; Games
reaching the board (all brains, with draws, timeouts, illegal fires per brain
and ticks per game, parsed from the per-game `gen` lines); Best lineage pair
wins, labelled relative, with the best genome's outcomes; Exam, draws vs the
perfect player with losses, timeouts and the random-mover result, from the
latest exam generation. Charts: Population play (reached the board / draw /
timeout over generations), Pair wins (relative), Best brain outcomes, Exams
(draw vs perfect, win vs random on exam generations). The dead lifespan,
survivors and evaluation cards are repurposed or hidden. The live battle's
move log is a fixed scrollable box that folds repeated ignored fires into one
counted chip, and the battle card reserves its height so nothing reflows.

**Seat fairness (run 4, generation 140 on, from the user).** Two things gave the
X seat an edge in a level game. (1) In a drawn game X fills the ninth cell and
collects a fifth legal move for a cell that changes nothing. Now, when one cell
is left and filling it cannot win for the player to move, the game ends as a
draw at once, in training and in the live battle: four legal moves each. (2)
The waiting credit: O always waits through X's opening, so O always earned
more of it. Removed (`--reward-wait 0`). A level game is now decided by illegal
fires alone (0.001 each), or is a true tie.

### Run 5: from generation 0 (2026-11-10, night)

Run 4 (archived in `runs/v4_gens0-179/`) carried the pair rule from its first
draft to its settled form. The user asked for a clean start under the settled
rules, with a randomized population.

Who survives a game (two brains, X and O, random seats, one survives):
1. a win on the board: the winner;
2. a timeout (30 ticks on your own turn without a legal move): the other one;
   if X timed out without ever opening, O is re-seated as X with a cleared
   memory for one turn: a move and it survives, none and both are out;
3. otherwise (a draw): the fewer illegal fires, at 0.001 each; equal is a tie.
Nothing else scores: no credit for a win, a legal move or waiting. The game
decides; the score only breaks a draw.

World: an illegal fire does nothing and the clock keeps running; a draw is
called as soon as the last cell cannot win for the player to move (four legal
moves each). Lineage fitness = share of games its 512 brains survived; top 10%
of 256 lineages carry over, children fill the rest, mutation x2.5, one in ten a
freak at x20; weights clamped +-5, noise std <= 3, leak in [0, 1]. Brain
unchanged: 29 vision sensors, <= 82 hidden, 9 spiking outputs, memory by leak,
no plasticity. Exams vs the random mover and the perfect player every fifth
generation, both seats, for the log only.

**A population five times larger, in GPU passes (run 5 restart, from the user).**
A generation no longer has to fit the card in one go: the lineages are dealt
into chunks (`--chunk 320` genomes x 512 brains, about 30% more than the card
held before), each chunk is born, plays its games and is scored in turn, and
the whole population is ranked at the end. Lineages are reshuffled into new
chunks every generation, so over time a brain meets peers from the whole
population, not a fixed neighbourhood. Run 5 restarted with 1280 lineages x 512
brains = 655,360 brains per generation in four passes; top 10% = 128 lineages
carry over. The exams and the best-brain save run on the chunk that holds the
best genome. Raised again at the user's request to 3000 lineages x 512 brains
= 1,536,000 brains per generation in ten passes of 300; top 10% = 300 lineages.

**Neurons and connections switch on and off individually (run 5 restart, from
the user).** The hidden-node count gene (`n_hid`, which switched on the first n
hidden neurons) is gone. Instead every hidden neuron carries its own on/off
gene (`alive`), inherited with the neuron in crossover and flipped by mutation
with a 1% chance per neuron (times the mutation scale, freaks included, capped
at 50%); a lineage starts with about 70% of hidden neurons on; sensors and
outputs are always on. Every connection likewise has its own on/off (`conn`),
inherited with the receiving neuron, flipped with a 0.5% chance, starting at
50% density; the weight gene `W0` is the strength a connection has when it
exists, and a brain's wiring is `W0 * conn`. A synapse can now be pruned or
grown outright instead of a weight having to drift to zero, and a sparse brain
is possible. Run 5 restarted from scratch with 3000 lineages.

## Current training: one brain, many games, averaged backprop (`single.py`)

Evolution turned out to be the crutch for a learning rule that did not work. The
run that finally learned is a single brain whose weights are shared by 262,144
games played in parallel (each with its own board, activations and stored
numbers). When games are judged in a tick, every judged game computes its own
backward wave; the waves are averaged over the judged games and applied once.
Rewards: illegal -1, win +1, loss -0.5, freeze +0.5 on the output nearest to
firing; legal moves earn nothing. No deaths, no births. Batch is now 65,536 games.

- Learning rate 0.3 reached 20% illegal in 100 games but pinned weights at the
  clamp and oscillated; later, with traces, anything above 0.01 destabilised, so
  0.01 is the working value. It can be changed live by writing a number into a
  file named `LR`.
- Controls while running: `SAVE` file (checkpoint), `STOP` file, Ctrl+C or
  `--minutes N` (save and exit), `--init-from brain.npz` to continue.
- Memory-path gradients are clipped per element (+-1) and pass through the tanh
  derivative of the stored number; without this they went to NaN in 10 games.

## Life, death and evolution (continuous mode, `continuous.py`)

- No generations. All brains live at once and play game after game forever.
- A brain has 30 lives. An illegal move or a freeze costs one. At zero it is
  terminated.
- A dead slot is refilled immediately: 8 living brains are drawn at random, the
  oldest (most games survived) is the parent. The child inherits the parent's
  *current learned* weights, memory genes and learning-rate genes, each with
  ~1% of entries mutated. Fresh memory, full lives, age 0.
- Opponent: a uniformly random legal mover. A good player should beat it ~95%.
- Checkpoints: the full population every 500 games, three rotating slots.

## What failed, and why (chronological)

1. **Generational evolution with uniform crossover and 20% elite.** Flat. Crossover
   between unrelated fully connected nets scrambles both; the top 20% of a large
   population is mostly luck. Fix: mutation only, 2% elite.
2. **Fitness = games survived.** Too coarse; most brains die in game 1 and all
   score zero. Fix: fitness = legal moves made before death.
3. **Kill on first mistake.** Nobody survives; selection has nothing to work with.
   Fix: lives.
4. **Hebbian trace rule with pleasure/pain inputs.** Evolution drove the learning
   rate to zero: learners died younger than non-learners. Pleasure after every
   legal move cemented the "fire instantly on my usual cell" habit; pain lingered
   for 5 ticks and also weakened the brain's own response to pain. A uniform
   learning rate pushed all 12,000 weights at once, drowning the useful correction.
5. **Seeding a population from one saved brain.** Lost all diversity; and the
   saved brain's behaviour depended on its saturated memory state, which clones
   did not have. Lesson: checkpoint the whole population.
6. **Backprop with random-sign learning-rate genes.** Half the connections learn
   against the gradient: noise. Fix: start genes positive.
7. **Backprop with tanh' = 0 at saturated nodes.** No update at all. Fix: floor.

## Results so far

- Single brain, averaged backprop, lr 0.3: 96% -> 20% illegal and 4% -> 45% wins
  in 100 games (about 12 minutes). Evolution needed 2000 games / 3.9 million
  births to reach 45% illegal.
- Earlier:

- Evolution alone (continuous, 3 lives, random opponent): 93% -> 45% illegal
  over 2000 games and 3.9 million births, then ~1 point per 100 games.
- First within-life learning signal (backprop, positive genes 0.1, fresh brains):
  illegal 87.8% -> 81.6% over a brain's first 30 games with zero deaths.

## Files

- `brain_ttt_gpu.py`: arena, brain, memory, learning rules, generational trainer.
- `single.py`: one shared brain, parallel games, averaged backprop or Hebbian (`--rule`), neural noise (`--noise`).
- `evo_rule.py`: evolution only; every aspect of the brain, including its learning rule, is a gene (the main experiment now).
- `perfect.py`: perfect tic-tac-toe player as a lookup table; the training dummy for `--opponent perfect`.
- `continuous.py`: continuous life/death/replacement loop (evolution; superseded).
- `long_life.py`: let one evolved brain live and learn for thousands of games.
- `dashboard.py` / `dashboard.html`: live charts from the training logs.
- `brain_ttt.py`: original CPU/numpy version.
