# The theory: current AI is flawed because a human designed it

## The claim

Every modern AI model, including the transformer behind today's language models,
is a structure a human drew on a whiteboard. Someone decided: tokens go in here,
attention happens there, a feed-forward block follows, stack it N times, read the
answer off the end. The learning algorithm (gradient descent on a loss we chose)
is also a human decision.

The theory is that this is the root limitation. A system whose shape was chosen
by a human carries the limits of that human's imagination. It can only become
what the designer made room for.

## What a brain does differently

A brain was not designed. It was selected. Three properties follow from that:

1. **No prescribed wiring.** Neurons are massively interconnected. Nobody
   decided which neuron talks to which. Connections that help survival were kept,
   the rest were not.
2. **It never stops.** A brain runs continuously from birth to death. There is no
   "forward pass" that starts with an input and ends with an output. Inputs arrive
   while it is already running, and actions happen when internal activity crosses
   a threshold.
3. **Mistakes are fatal.** Evolution does not nudge a weight by 0.001 when an
   animal makes an error. The animal dies and does not reproduce. The survivors
   define the next generation.

Modern AI violates all three. Its wiring is a fixed layered design, it runs as a
one-shot function from input to output, and it learns by small corrections to a
human-chosen objective.

## Where the theory is strongest

- **The data is human output.** Models trained on human text inherit human
  biases and a ceiling at human knowledge.
- **The objective is human-chosen.** "Predict the next token" and "please the
  human rater" are proxies. Optimizing proxies hard produces sycophancy and
  confident errors.
- **The architecture is static.** Nothing in the model changes after training.
  A brain is never finished.

## Where the theory is contested

These counterpoints came up in discussion and are recorded so the experiment can
address them honestly rather than ignore them:

- Human-designed systems have already exceeded humans in narrow domains (chess
  engines, AlphaGo). Design sets the search space; it does not cap what the search
  finds inside it.
- The transformer is not a copy of human cognition. It is alien, built around
  what GPUs do well. So "human-shaped" may describe the data and the objective
  more than the architecture.
- Evolution is slow and expensive. Gradient descent exists because it is millions
  of times more sample-efficient. The question is whether that efficiency costs
  something we cannot see.

## What would count as evidence

The theory makes a testable prediction: a system with **no designed wiring**, that
**runs continuously**, and is shaped only by **survival**, should learn a task
without anyone specifying how. Tic-tac-toe is the first task. If a fully
connected, continuously ticking brain can learn legal, competent play through
selection alone, the theory survives its first test. If it cannot, that tells us
which of the three properties is doing less work than the theory assumes.
