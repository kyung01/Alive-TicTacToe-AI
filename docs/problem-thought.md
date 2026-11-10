# Problem 2: the AI cannot contain a thought

The second thing blocking AGI is that modern AI has nowhere to hold a thought.
It has text going in and text coming out, and the only thing it can hold in
between is more text.

## Thought is not text

A thought is not a sentence. It is not a picture either. It is something we
cannot measure or write down in full. It is an abstraction of an idea, held in
many formats at once. Right now you can feel several things, picture several
things, and reach back to several moments in the past, all at the same time, and
none of that is a string of words. When you try to describe a thought you lose
most of it. The change from one thought to the next is not describable at all.
It just happens, and you notice afterward.

Thought is also most of what we are. We spend our lives inside it. Any system that
claims to be a general intelligence has to have something like it.

## What the AI has instead

In a modern model, thought *is* text. When it "reasons", it writes words to
itself and then reads them back. Its inner monologue is a transcript. People see
that transcript and believe the model is thinking, because it looks like what
thinking would look like if you wrote it down.

But look at what is actually happening between the input and the output. The
middle of the model is a filter. Text enters, is transformed through a fixed
sequence of layers, and leaves as text. Nothing is retained. Nothing is held.
The only state that survives from one step to the next is the words it already
produced. If a thought cannot be spelled, the model cannot have it. If a thought
is three feelings and two half-formed images at once, the model has to flatten it
into a sentence or lose it.

So the model does not think. It processes. It is very good at producing output
that reads like the output of thought, and that is exactly why the confusion is
so easy to fall into.

## Why this blocks AGI

A mind needs a place for things that are not yet words: a hunch, a mood, two
options being weighed without either being named, a memory that colours a
decision without being recalled. That place is internal state that is rich,
continuous, and not forced through language. The current architecture has no
such place. Its internal state between calls is zero, and its internal state
within a call is a single pass that is thrown away. Everything it keeps, it
keeps as text.

A system that can only hold what it can say cannot form a complex thought. It can
only form a complex sentence.

## What this project does instead

- **The thought is the activity.** The brain has 82 hidden nodes that are not
  sensors and not outputs. Their values are the brain's internal state. They are
  never read out as text, never decoded, never named. They are simply what the
  brain is doing right now.
- **It is held, not passed.** Because every node feeds every node, and the brain
  ticks continuously, this state persists and changes from tick to tick. It is
  not thrown away after a pass. It is the brain's ongoing condition.
- **It is many things at once.** Eighty-two continuous values changing together
  are not a sentence. They can represent several leanings at the same time, and
  the brain does not have to collapse them into one until an output fires.
- **Deciding takes as long as it takes.** Untrained brains fire on the first
  allowed tick. Trained ones have been seen taking 4 to 12 ticks on the same
  board. Something is happening in those ticks that is not an input and not an
  output. That is the closest thing in this project to a thought.

See [problem-structure.md](problem-structure.md) for the first problem (one-way,
start-stop wiring). See [theory.md](theory.md) for the third: the wiring is
designed by a human rather than selected.
