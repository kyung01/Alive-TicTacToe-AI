# Narration script

Twelve slides, about 6 to 7 minutes at a relaxed pace. Each slide lists the
on-screen beats (press → to reveal each one) and the voiceover. Press **A** to
reveal all beats on a slide at once if you prefer to talk over a finished slide.

---

## 1 · Title (0:00)

> What if we stopped designing AI?
>
> Not "designed it better." Stopped. No layers, no training algorithm, no rules
> written down anywhere. Just a blob of connected nodes, and one rule: if you make
> a mistake, you die.
>
> I've been testing that idea on the simplest game I could find. Here's what
> happened.

## 2 · Every modern AI is a drawing someone made (0:30)

Beats: tokens in → attention → feed-forward → stack it → trained by nudging.

> Every large AI model today, including the ones you talk to, is a transformer.
> And a transformer is a diagram. Someone sat down and decided: the text comes in
> here, gets turned into numbers. Then a block called attention lets every word
> look at every other word. Then a small network processes each word on its own.
> Stack that a hundred times, and the answer comes out the end.
>
> Then you train it by showing it trillions of words and nudging every weight a
> tiny bit toward a goal. A goal that, again, a person picked.

## 3 · The theory (1:15)

Beats: the data → the goal → the shape → the big zero.

> Here's the theory. A system shaped by a human carries the limits of that human.
>
> The data is our writing, so it inherits our biases and tops out at what we know.
> The goal is a stand-in we chose, like "predict the next word" or "make the rater
> happy." Push on a stand-in hard enough and you get a model that flatters you and
> is confidently wrong. And the shape is frozen. Nothing in it changes after
> training day.
>
> Count the connections in a transformer that were not decided by a person.
> Zero.

## 4 · A brain was never designed (2:00)

Beats: three cards, then the closing line.

> Compare that to the only general intelligence we actually know about.
>
> A brain has no prescribed wiring. Nobody chose which neuron talks to which.
> Connections that helped an animal survive got kept.
>
> A brain never stops. There is no forward pass that starts with an input and ends
> with an output. It's running before the input arrives, and it acts when its
> internal activity crosses a threshold.
>
> And in evolution, mistakes are fatal. Nobody nudged a weight. The animal that got
> it wrong didn't have kids.
>
> Modern AI violates all three.

## 5 · 109 nodes, every node connected to every node (2:50)

> So I built the opposite of a transformer.
>
> A hundred and nine nodes. Eighteen are sensors, they just see the tic-tac-toe
> board, nine cells, mine or theirs. Nine are outputs, one per cell. The other
> eighty-two are free. No layers. No roles. Every single node is wired to every
> other node, including itself.
>
> Each tick, every node sums up what everyone else is saying, squashes it, and
> that's its new value. Then it ticks again. Forever. Eleven thousand eight hundred
> and eighty-one weights, and I didn't place one of them.

## 6 · It fires when it is ready (3:35)

Beats: five lines, animation runs on its own.

> Here's how it makes a move. The opponent plays. The brain doesn't get asked
> anything. It just keeps ticking, and the nine output nodes rise and fall.
>
> The first one to cross the threshold, that's the move. If it picks a cell that's
> already taken, that's an illegal move, and the brain dies. If nothing fires in a
> hundred ticks, it froze. It dies.
>
> I'm not telling it the rules. It has to notice them, or not exist.

## 7 · No teacher, no gradient (4:15)

Beats: birth → life → death → breed.

> There is no training in the normal sense. It's selection.
>
> Fifty thousand random brains are born. Each one plays a hundred and fifty games
> against a random opponent. Each one dies at its first mistake. Its score is how
> many legal moves it managed before it died, plus the games it won.
>
> The top twenty percent get to reproduce. A child takes each weight from one of
> two parents and gets a little random mutation. Then we do it again.

## 8 · Every brain, every game, every tick at once (4:50)

> This only works because of scale. Fifty thousand brains times a hundred and
> fifty games is seven and a half million games per generation. All of them tick
> in lockstep as one giant matrix multiply on the GPU. Thirty-five seconds a
> generation, about thirty gigabytes of memory.
>
> My first version ran two hundred brains on the CPU in the same time. That's
> one twentieth of one percent of the search.

## 9 · The brain can play. It cannot yet obey. (5:20)

Beats: diagnostic result → faithful result.

> So, results. Two of them, and they pull in opposite directions.
>
> First, a diagnostic. If I handle legality for the brain, just block the occupied
> cells, the exact same architecture learns to beat a random player seventy-eight
> percent of the time. And as it gets better, it takes longer to decide. It's
> actually using those recurrent loops. The architecture works.
>
> Second, the real test. Left to learn legality from survival alone, every run is
> flat. The best brain lives about six games. Most die on their first move.

## 10 · Eighteen weights against a wall of noise (6:00)

> Why is legality so hard? Look at one output node. It's listening to eighty-two
> hidden nodes with random weights, and together they roar.
>
> To stay quiet on an occupied cell, the two sensor wires for that cell have to
> shout "no" louder than all of that. Nine cells, eighteen wires, out of nearly
> twelve thousand. And every time two unrelated parents mix, a half-finished
> solution gets torn apart.
>
> Nature had exactly this problem. Nature also had a billion years.

## 11 · Change the starting conditions, never the rules (6:35)

Beats: A → B → C → D → closing line.

> So what's next. Four things, and notice none of them tell the brain what a legal
> move is.
>
> Let the big run finish, four hundred generations. Play games in sequence so a
> brain carries memory from one game into the next, a real lifespan. Breed more
> gently, so a brain that learned one cell doesn't get scrambled. And start
> quieter, with smaller random weights, so eighteen wires have a chance of being
> heard.

## 12 · Outro (7:00)

> That's the question I'm chasing. Can structure emerge instead of being drawn?
> Tic-tac-toe is the first test. If it works, the next game is bigger. If it
> doesn't, we'll know exactly which part of the theory was carrying less weight
> than I thought.
>
> Either way, I'll show you the numbers.
