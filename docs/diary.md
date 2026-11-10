# Diary

Informal, first person, one entry per day of work. The experiments log
([experiments.md](experiments.md)) records setups and numbers; this records what
I noticed and what I was thinking. Newest entry last.

## 2026-11-09 — the perfect player made the brain impatient

Today I realised the opponent I chose for training has been shaping the brain in
a way I never intended.

I trained against the perfect player (the minimax table) because it is the
cleanest yardstick: it never makes a mistake, so any loss is the brain's fault,
and it is fast, because a table lookup answers the instant the brain's move
lands. That second property is the problem. In the arena the opponent's reply is
settled in the same tick the brain fires. From the brain's point of view the
world always answers immediately, and always answers well.

So the brain learned two things it should never have been allowed to take for
granted:

1. **The opponent replies at once.** The brain never had to wait. It never
   experienced an opponent that is still thinking while the board is quiet.
2. **The opponent's reply is always the best one.** The brain only ever saw a
   narrow slice of board states, the ones a perfect player steers into. Its
   whole experience of tic-tac-toe is the perfect-play tree.

The result is a brain that is good at one thing: dealing with the perfect player.
Put it against anyone else and it falls apart.

- Against a **bad or random player** it sees boards that minimax would never
  produce, and it has no idea what to do with them. A free win is just as alien
  to it as a trap.
- Against a **copy of itself** it is worse. The copy is a continuously ticking
  brain that takes several ticks to decide. The trained brain is not used to
  waiting: it fires out of turn, or plays on a cell before the other has
  answered, or sits through the silence until it freezes. It is impatient in the
  literal sense: it has never had to be patient.

What this taught me is that the opponent is not a neutral measuring stick. It is
part of the environment, and the brain evolves to fit the environment it is
given, including its tempo. A perfect, instantaneous opponent is a very strange
environment. No real opponent is like that, and a brain that only ever lived
there has learned a lot about that opponent and very little about the game.

It is also a nice illustration of the project's own thesis turned on me. I have
been careful not to design the brain's structure, but I did design its world, and
the world I gave it was too regular. The brain found the regularity and leaned on
it, exactly as it should.

Things I want to try, roughly in order:

1. **Mix opponents during training.** Random, perfect, and self-play in the same
   lifespan, so no single tempo or style can be assumed. The `--opponent mixed`
   option exists; I should make it the default and widen what "mixed" means.
2. **Give the opponent a variable delay.** Even the table player should answer
   after a random number of ticks, so waiting is a normal part of life and
   firing during the opponent's turn is something the brain has to learn not to
   do.
3. **Self-play as a first-class opponent**, not just a demo. Two copies ticking
   against each other is the most honest test of whether the brain understands
   turns at all, since neither side gets an instant reply.
4. **Measure patience directly.** Log how often the brain fires out of turn and
   how many ticks it waits before doing so, by opponent type, so this failure is
   a number on the dashboard rather than something I only notice when I watch a
   game.

The brain is not stupid. It did exactly what I selected for. I just selected
for the wrong world.

## 2026-11-10 — trying was punished, so it decided not to try

After the perfect player came out of training the brains only played each
other. Within one generation the population settled into a strategy I had not
anticipated and then held it for ten generations without a single game reaching
a result on the board: **do nothing**. The best genome fired in under 5% of its
games, froze in about a quarter, and in the rest simply stood there until the
other brain killed itself first, which it did seven times out of ten. In the
exams it froze against the random mover 93% of the time and against the perfect
player 97%. Its firing threshold had climbed to 0.99, the top of its range: it
had evolved outputs that almost cannot fire.

The reason is not that the brains are stupid. It is the order the world ranked
their outcomes in. Selection here is by rank, so only the order matters, and the
order was:

    opponent dies while I wait (0) > I freeze (−0.5) > I try and get it wrong (−1)

Trying was the worst-ranked thing a brain could do. A legal move paid +0.3, but
one move leads to a freeze, because an output that has crossed its threshold
stays high and can never spike again in that game, and a move followed by a
freeze scores less than never moving at all. So the brains did the one thing
the ranking rewarded: nothing. Trying was punished, and they decided not to
try.

Two lessons.

- Punishment does not teach caution, it teaches avoidance. A world where every
  action can cost and inaction is free will be answered by an organism that
  does not act. Nature's version of this is the animal that freezes.
- Memory made it possible. The brain can hold its outputs below threshold
  indefinitely because every node keeps its value from tick to tick. Give a
  brain memory and a world that pays for stillness, and stillness is what you
  get. Figure 7 of the design diagrams records the moment.

What I changed is the world, not the brain. New prices: a win on the board
+100000, a legal move +10, a draw or the opponent dying 0, an illegal move −1,
a loss −10, a freeze −100000. Freezing is now the worst thing a brain can do,
a hundred thousand times worse than a wrong move, and one win outweighs
everything else a lineage could earn.

The first generation under the new prices (266) flipped the population
instantly. The best genome now fires in 94% of its games, illegally almost every
time, and never freezes. The population mean fell to −27,000 because the
freezers are being wiped out this generation. One brain won a game against the
random mover in the exam, the first board win in twelve generations. From
"never try" to "always try, badly" in one step; the ranking is doing exactly
what it says.

Still open: the physics that made a single move a one-way door. A spiked
output does not come back down on its own, so a brain cannot act twice in a
game unless its wiring pulls the output under the threshold again. A real
neuron resets when it spikes. Ours do not yet.

## 2026-11-10, later — with no cost for an illegal move, it marked the same cell twice

After the punitive prices had broken the do-nothing habit in two generations
(freezes gone, the first board wins), I took the next step in the same
direction: reward only, never punish. A legal move +1, a win on the board +100,
and everything else, loss, draw, freeze, illegal move, opponent dying, exactly
0. If punishment teaches avoidance, use none.

The population answered within a generation, and then sat there for fifteen.
Every game lasted two ticks. Every brain fired on the first tick, wherever it
was: the one playing X made a legal move (+1), the one playing O fired out of
turn (0, game over). And the brains playing X had found something better than
one move: fire, then fire again on the same cell the next tick. The second
spike is illegal and ends the game, but an illegal move now cost nothing, so
"mark it twice, fast" scored exactly the same +1 as a careful single move and
was far easier to build. It needs no sense of turn and no sense of the board,
only an output that stays above threshold for two ticks, which, thanks to
memory, every output does.

So the brain found the cheapest thing the world would pay for, again. Under the
old prices that was stillness; under these it was a rapid double tap. Both are
the same lesson from the other side: the organism does not learn the game, it
learns the price list. Anything the price list does not charge for, it will do
if that is the easiest route to what it does pay for.

The fix was one number: an illegal move costs -1 again (legal +1, win +100, all
else 0). A double tap now nets zero, so the only way to get ahead is a second
legal move, which means bringing the output back down and firing somewhere
else later. First generation after the change (288): the whole population went
negative as the double-tappers were priced out. Second (289): the best genome
started losing on the board against the perfect player 5% of the time, which
sounds bad and is the first good news in a while: a loss on the board means it
played past its first move.

## 2026-11-10, run 2 — the best brain is the smallest one

Run 2 started from generation 0 with a broadly randomized population and, from
generation 8, a tournament instead of a price list: a brain's fitness is the
share of its games it survived (win, draw, or the opponent died first), the top
half of the lineages goes through unchanged, and their children fill the rest.

Through generations 6 to 16 the lineage that keeps coming out on top has only
**37 of its 82 hidden nodes switched on**. Other winners along the way had 51
and 61; the population started spread between roughly 25 and 85. The smallest
brain in the running has been the most successful one, and it got there fast.

My reading: it simplified itself because the winning strategy is simple. The
top strategy in this phase is still waiting, letting the other side die first,
and a waiter's only job is to keep its nine outputs under the threshold. Every
live hidden node is a noise source, kicked at random every tick, and the
outputs sum what the hidden nodes send them, so fewer nodes means a quieter
brain that holds still more reliably. Fewer connections also means less for
plasticity to disturb over a 40-game life. For a brain whose entire behaviour
is "do nothing", extra nodes are pure liability, and selection trimmed them.

This is worth keeping as an instrument. Brain size is a gene, so it tracks
the complexity of what the population is actually doing. If the hidden-node
count of the best genome climbs, the population is building machinery for
something: making a move and then stopping, reading whose turn it is, reading
the board. If it stays at 37, it is still just waiting, whatever the other
numbers say.

Also worth noting what did not happen: the brain did not shrink by losing
genes. Switched-off nodes keep their biases, leak, noise and incoming weights,
and are inherited whole in crossover. A lineage can grow back later and those
nodes wake up with whatever they were carrying.

## 2026-11-10, run 2, generation 17 — the first mover, with more brain

Nine generations into the tournament the top genome changed kind. The winner of
generation 17 has **58 live hidden nodes** against the waiter's 37, and in the
live battle it does something the waiters never did: as X it opens with a move,
always the same cell, bottom-right, and then holds still for the rest of the
game.

That is the next rung on the ladder the tournament was supposed to build. A
mover as X beats a waiter as O: X's move starts O's 30-tick clock, the waiting
O freezes, X survives. Against another mover it does no worse than a waiter.
In the log it still looks like a waiter, 72% of its games end with the opponent
dying first, because almost every opponent is still a waiter and dies on its
own clock; the difference shows in the live battle, not yet in the numbers.

What it had to build: a single timed pulse. One output has to rise over the
threshold once, at the start of the game, and then all nine outputs have to
stay under it for the rest of the game. A waiter needs none of that, and the
node count says so: the strategy got one step more complex and the brain that
carries it is half again as large. The cell it picks, bottom-right, is most
likely just the output this lineage found easiest to push over the line; the
board does not care where the first move lands.

So the size instrument from the previous entry gave its first reading, in the
expected direction. Next rungs, if the ladder holds: an O that answers that
one move, then an X that moves twice, with the node count climbing each time.

## 2026-11-10, run 2, generation 21 — stronger mutation, and freaks

Under the tournament the top half always goes through unchanged, which keeps
the population honest but also makes it drift slowly: thirteen generations of
survival selection and the best genome's survival rate has sat between 68% and
75% the whole time. The children were too much like their parents to find the
next rung quickly.

Two changes to the breeding program, both the author's.

- **Every mutation step is 2.5x larger** than before: the scalar genes, the
  per-node genes (bias, leak, noise), and the weights.
- **10% of the children are freaks.** Instead of 30% of children mutating 5x
  harder, one child in ten has every gene mutated 20x harder, on top of the
  2.5x. A freak lands well outside its parents' neighbourhood: a different
  threshold, a different brain size, different wiring. Most freaks will die in
  their first games. The point is the one that doesn't.

The thinking: the ladder from waiter to mover to responder needs qualitatively
new behaviour at each rung, and small steps around a local optimum are slow to
find that. Freaks are the cheap lottery ticket; the surviving half is the
insurance that the population never loses what it already has.

This is world-side (the breeding program), not brain-side, so it is allowed to
be arbitrary. Resumed from generation 20.

## 2026-11-10, run 2, generation 25 — pitting them against each other did not work

Seventeen generations of the tournament among peers (8 to 24), the last four
with 2.5x mutation and 10% freaks, and the same lineage sat on top the whole
time at 68% to 75% survival. The median crept from 54% to 60% as the population
became more uniform, but nothing displaced the waiter-then-mover, and no game
ever reached a result on the board.

Why peers are a poor teacher here: they die too easily. Seven in ten games
ended with the opponent killing itself, so the winning move was always "do
less than the other guy", and the population converged on that. A ladder only
climbs when the rung below pushes; a peer that freezes on its own clock pushes
nobody. The perfect player had the opposite problem earlier in run 1, a single
fixed line that the brain fitted instead of the game. Both are the same lesson
from different ends: the opponent is part of the world, and the world has to
make the next behaviour necessary, not merely possible.

New opponent: the random legal mover, answering after a random 1 to 30 ticks,
the full width of the turn clock, so there is no rhythm to learn and the brain
has to watch the board to know when to act. Under survival selection the
random mover never blunders, so a brain survives a game only by winning or
drawing it. It is the first opponent that cannot be beaten by waiting, and it
cannot be fitted either, because it has no line. The perfect player stays as
an exam and as the opponent in the live battle, where it is the clearest
demonstration of what the brain can and cannot do.

Resumed from generation 24. First games against the random mover: 69% of all
brains froze and 31% fired illegally. The waiters are meeting an opponent that
does not die.

## 2026-11-10, run 2, generation 31 — I had to design a system that encourages the AI to keep trying

Against the random mover, pure survival scoring went blind. The random mover
never blunders, so a brain survives a game only by winning or drawing it, and
the population could do neither: best 0.7%, population mean 0.0%, six
generations in a row. Almost every brain scored exactly nothing in every game,
which meant the "top half" that breeds was being picked by the random
tie-break. The ranking had no information left in it.

What was missing was not a reward for the right answer but a reason to keep
attempting answers at all. So the survival score now credits an attempt: an
illegal move earns 0.1, a freeze earns 0. Surviving (1) beats trying and
failing (0.1), which beats not trying (0). We are, for the first time,
rewarding an illegal move. That felt wrong for a moment and then obviously
right: a brain that fires at the wrong time is one correction away from a brain
that fires at the right time, and a brain that never fires is nowhere.

A loss on the board stays at 0. I briefly gave it the same 0.1, worried that a
brain about to lose would fire anywhere to collect the attempt credit, and took
it out again: that is reasoning about tactics the brain does not have yet, and
every extra term in the score is one more thing it can learn instead of the
game.

The pattern of the whole day, in one line: every time the world made inaction
free, the brain stopped acting; every rule I added since has been about making
trying worth more than not trying, without telling the brain what to try.

## 2026-11-10, run 3, generation 55 — the last learning rule leaves the brain

I thought we had taken plasticity out when we took out the timer and the
sibling baseline. We had not. The within-life learning rule was still there,
every coefficient a gene but the shape of the rule mine: a Hebbian trace,
applied at the end of each game, scaled by the brain's own valuation of the
outcome. When I asked to shorten lives to one game, the answer was that it
would make the plasticity genes do nothing, and that was the first I realised
they were still doing something.

So: gone. The brain no longer changes during its life. Its weights are its
genome's weights from the first tick to the last. What remains inside the
organism is only what a neuroscientist would call the nervous system itself:
nodes with a bias, a leak (memory), a noise level, a firing threshold, and the
wiring between them. Everything that changes, changes by selection.

This also settles a question I had been carrying: is this brain "learning" or
"evolving"? Now it is only evolving, and a life can be one game. A generation
is 131,072 brains each playing once, judged, bred. That is a cleaner
experiment, and a faster one.

What I expect to lose: the possibility of a brain that adapts to an opponent
within a series of games. For tic-tac-toe against a random mover that was
never going to matter much. What I expect to gain: no more ambiguity about
where a behaviour came from. If a brain waits, it was born waiting.

## 2026-11-10, run 3, generation 556 — they had only ever played X

The population that won half its games against the random mover collapsed the
moment it met peers: best fitness from 56 to under 1 in a single generation,
every game over in two ticks, half the brains dying by an illegal move on
their first tick. My first thought was that peer play was poison again. It was
a seat bug. In solo games the brain's seat came from the game index, and once
lives became one game long that index was always zero, so for five hundred
generations every brain played X and not one of them ever played O. Seated as
O against a peer, they did the only thing they knew: fire on tick one.

Two things worth keeping from this. The result I had been happy about, "49%
wins against the random mover, never loses on the board", was a one-sided
skill. And the brain had an "I am O" sensor the whole time that had never once
been lit in training, so there was nothing wired behind it. Capacity that is
never exercised does not develop; the sensor was there, the behaviour was not.

Fixed by seating solo brains at random each game. The generation 555 population
was still on disk and is the resume point, peers only, where half the brains
are O in every game and have to learn it. The collapsed generations are
archived.

## 2026-11-10, run 4 — start over, peers only, winner of each pair survives

Run 3 is thrown away. Five hundred generations of what looked like progress
was one-sided play, and the author's verdict was the right one: start fresh, and
this time the brains only ever meet each other.

Selection is the author's rule, and it is the most tournament-like we have had.
Each game is a pair. Both brains get the reward-only score; whichever scored
higher survives and the other does not. One special case, also the author's: if
X never opened, the O brain is moved into the X seat with a cleared memory and
given one turn to make any move at all. If it can, it is the winner. If it
cannot, both are thrown out. So two waiters meeting each other both die, and
there is no corner of the world left where doing nothing pays.

One fix before the real start: when both brains fire on the same tick, X's
fire is a legal opening and O's is out of turn, and the game loop was ending
the game on O's fire before X's move landed, so X got nothing. Legal moves now
land first. Without that, a fresh population scored the same for every pair
and selection was blind on generation one.

## 2026-11-10, run 4, generation 410 — their goal is beating their cousin, and firing blind does that

Four hundred generations of the pair tournament and nothing has moved. Every
game ends on tick one. The best genome wins 55 to 58 percent of its pairs,
which is the noise band around 50, and it is a different genome every
generation.

The population found one behaviour and stayed there: fire on tick one, no
matter what the sensors say. As X that is a legal opening and wins the pair.
As O it is out of turn and loses the pair. It does not care whether it is X or
O, because it does not need to: every lineage does the same thing, so every
lineage wins its X games and loses its O games, fifty-fifty, and the pair rule
has nothing left to choose between. Their goal is not the game. Their goal is
beating the brain across the board, and since that brain is a cousin doing
exactly the same thing, the cheapest strategy that gets fifty percent is good
enough and nothing better is visible.

Why no lineage can climb out: two flaws in the rule, both mine.

- Ties go to the cheater. Equal legal moves, then one brain fires illegally:
  it collects the 0.1 attempt credit and wins the pair, because that is more
  than the fractions of a cent the other brain earned by waiting. Ending the
  game with an illegal move is the winning tie-break. The attempt credit was
  written for a world where the alternative was sitting still forever; inside
  a head-to-head it rewards the wrong brain.
- An O that waits gains almost nothing. Hold still one tick, play legally, and
  you have one point against X's one point; the pair is then decided by the
  tie-break above or by who can fire a second time. The one improvement that
  matters is invisible to the score.

So the mechanism is doing exactly what it was told. The higher score survives.
The scores just do not separate a brain that plays from a brain that fires
blind, and selection can only see what the scores separate. Fighting cousins
measures you against cousins; if every cousin fires blind, firing blind is the
measure of success.

Nothing changed in the code today after this was understood. Candidate fixes
for the rule, not yet taken: no attempt credit inside a pair, so an illegal
move never wins a tie; a tie goes to the brain that did not end the game.

## 2026-11-10, run 4, same evening — they found the loophole in my design

Looking harder at why every pair is a coin flip, I found the exploit, or rather
the brains found it and I finally read it off their behaviour.

From the X seat: open with a legal move, one point. Then fire again, on the
same cell. That is an illegal move, it ends the game on the spot, and X still
holds 1.1 against the fraction of a cent O earned by waiting. X wins the pair,
and O never gets a turn. "Move once, then break the game" is a guaranteed win
from X, and nothing O can do changes it. Fire on tick one and fire again is the
whole strategy, and it is exactly what the population converged on.

The flaw is that in my pair rule an illegal move works as an exit that banks
your score. You keep your lead by knocking the board over. In any real game the
one who breaks the rules forfeits; here the one who breaks the rules collects.
The attempt credit makes it slightly worse, but the loophole would exist
without it: 1 point against nothing is enough.

I did not design that strategy and would not have thought of it. The brains
did, by the only route evolution has: try everything, keep what works. It is
the third time today the population has found the cheapest thing the world
pays for, after the waiters and the double-tappers, and the lesson is the same
each time. The organism optimises the rule as written, not the rule as meant.
If there is a gap between the two, selection will find it before it finds the
game.

Candidate fix, not yet applied: in a pair, the brain that ends the game by an
illegal move or a freeze loses the pair outright, whatever the scores. Board
results decide when the game reaches one; a draw is a tie. Then the only way to
beat an opponent is to outplay it.

## 2026-11-10, run 4 — an illegal move is no longer an exit

The author's fix for the loophole, and it is the right shape: take the exit away.
An illegal fire now does nothing. The board ignores it, nobody is paid for it,
and the game keeps ticking while the brain's own 30-tick clock runs down. A
turn ends only when a legal move lands. If the clock runs out, the brain times
out and the opponent takes the pair.

So "open, then break the game" no longer exists as a move. A brain that fires
blind after its opening just stands there spending its clock, and the only
things that still decide a pair are the board and the clock: win, lose, draw,
or fail to move in time. That is the game as meant, finally with no gap
between it and the game as written. Fresh start, same brain.

## 2026-11-10, run 4, generation 110 — a harsh price for a wrong move makes the brain scared

At generation 65 I raised the cost of an illegal fire from a thousandth of a
point to a full point, the same as a legal move earns. Forty generations later
the picture was: games won on the board 34% down to 31%, timeouts 10% up to
14%, games four ticks longer, and illegal fires per brain only from 18.7 down
to 16.2. Draws stayed where they were.

That is a brain being made scared, not a brain being made careful. A crude
brain has no way to fire wrongly less often except to fire less, full stop, so
the price lands on the legal moves too. It hesitates, the clock runs, and it
times out in games it might have played. The behaviour the price was meant to
remove barely changed; the behaviour we want was damaged alongside it.

Under the rule as it stands the price only decides games that end level, so
it cannot have caused all of the drift by itself, and the timeout climb had
started before the change. But it did not help, and the direction was wrong.
The illegal-fire cost is back at a thousandth of a point. Illegal fires are
already ignored by the world; that was the real fix, the price was only ever
a nudge, and a nudge that frightens is worse than none.

Same lesson as the morning, from the other side. Punishment taught the first
population to do nothing. A harsh price taught this one to hesitate. What has
moved these brains forward every time is a world where acting is possible and
acting well is what survives.
