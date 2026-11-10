# Problem 1: the structure was never built to be a mind

Every major lab says it is building AGI. But the thing being built has a shape
that was never designed to be an intelligence. It was designed to be a function:
input on one side, output on the other, and a stack of layers in between that the
signal passes through exactly once.

## What a human looks like from the outside

A human has inputs: eyes, ears, skin. These are sensors. They are how we detect
the world. A human has outputs: a mouth, hands. These are how we act on the
world. Between the two sits the part nobody can see from outside, the part where
seeing turns into deciding. We call it the brain.

Modern AI copies this outside picture. Tokens in, tokens out, a hidden part in the
middle. The outside is right. The middle is wrong.

## How the middle is actually wired

The brain is a mass of neurons, and each one connects to many others. Nobody
assigned the connections a direction. A signal can travel from any neuron to any
other neuron: forward, backward, sideways, and around in loops. Activity
circulates. A thought can feed back into itself and keep going.

A neural network is a pipeline. The signal enters at layer one, moves to layer
two, then three, and leaves at the last layer. Each layer talks only to the layer
after it. Nothing flows back. Nothing circulates. A transformer lets tokens inside
one layer look at each other, but the stack is still one-directional and
computation ends at the top. That is not how a brain is structured. It is how a
calculator is structured.

## The brain never stops

The second difference follows from the first. Because a brain can loop, it can
keep running without new input. And it does. From birth to death it is active.
There is no moment where it finishes a thought and goes dark until the next
prompt arrives. Inputs land in a brain that is already busy. The only time a human
stops thinking is death, or brain death.

A neural network is idle until it is called. It computes once, front to back, and
halts. Between calls it does not exist. Its entire "thinking" is a single pass.
The people building these systems bolt on loops from the outside, calling the
model again and again, but the loop is in the harness, not in the mind. The mind
itself still cannot think a second thought about its first one.

## Why this blocks AGI

These two differences, one-directional flow and start-stop execution, are treated
as implementation details. They are not. They are the structure. A system that
cannot send a signal backward cannot reconsider. A system that halts between
inputs cannot have a train of thought. You can scale that structure to any size
and it stays a very large function. A general intelligence has to be something
that runs, and the current architecture cannot run; it can only be run.

That is the first problem, and it is structural. No amount of data or training
fixes the shape of the thing being trained.

## What this project does instead

- **Every node connects to every node**, including itself. There are no layers
  and no direction. Signals go wherever the weights send them, including back.
- **The brain ticks continuously.** It is not called with an input. The board is
  written into its sensor nodes while it is already running, and it keeps
  running after it moves.
- **A move is a threshold event, not the end of a pass.** When an output node
  crosses the firing level, that is the action. The brain decides *when* it has
  decided.

See [problem-thought.md](problem-thought.md) for the second problem: the model
holds thought only as text. See [theory.md](theory.md) for the third: the wiring is designed by a
human instead of selected. See [approach.md](approach.md) for how the brain is
built.
