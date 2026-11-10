# Voiceover script: "AI can't think"

Twelve slides, about 6 minutes at a relaxed pace. Press → to reveal each beat,
or A to reveal a whole slide at once. Timestamps are rough.

---

## 1 · Title (0:00)

> Every big lab says it's building AGI. A machine that thinks the way we do.
>
> I think they can't get there. Not because the models aren't big enough, or
> the data isn't good enough. Because the thing they're building is the wrong
> shape. There are two structural problems, and I want to walk you through
> both. Then I'll show you what we're trying instead.

## 2 · Inputs, outputs, hidden middle (0:30)

Beats: senses → arrows → brain → arrows → actions.

> Start with you. You have inputs: eyes, ears, skin. That's how you detect the
> world. You have outputs: a mouth, hands. That's how you act on it. And in
> between there's a part nobody can see from outside, where seeing turns into
> deciding. We call it the brain.

## 3 · AI copies the outside (0:55)

Beats: prompt → box → answer.

> Modern AI copies this picture. Text in, text out, something in the middle.
> And the outside is right. It's the middle they got wrong.

## 4 · In a brain, signals go everywhere (1:15)

> Here's what the middle actually looks like in you. A brain is a mass of
> neurons, each connected to many others, and nobody assigned the connections
> a direction. A signal can go from any neuron to any other. Forward, backward,
> sideways, and around in loops. Activity circulates. A thought can feed back
> into itself and keep going.

## 5 · In an AI, signals go one way (1:50)

> Now the model. A neural network is a pipeline. The signal enters at layer one,
> moves to layer two, then three, and leaves at the end. Each layer talks only
> to the one after it. Nothing flows back. Nothing circulates. Transformers let
> words inside one layer look at each other, but the stack is still one
> direction, and it ends at the top.
>
> That's not how a brain is structured. That's how a calculator is structured.

## 6 · A brain never stops (2:30)

Beats: brain timeline, then AI timeline.

> And the second difference follows from the first. Because a brain can loop,
> it can keep running without new input. And it does. From birth to death you
> are thinking. Inputs land in a brain that is already busy. The only time you
> stop is when you die.
>
> A model is idle until it's called. It computes once, front to back, and
> halts. Between calls it doesn't exist.

## 7 · You can't get a mind out of a pipe (3:05)

Beats: three red pills, then the closing line.

> People treat these as implementation details. They're not. A system that
> can't send a signal backward can't reconsider. A system that halts between
> inputs can't have a train of thought. Scale it to any size and it's still a
> very large function.
>
> Yes, engineers bolt loops on from the outside, calling the model again and
> again. But that loop lives in the harness, not in the mind. The model still
> can't think a second thought about its first one.

## 8 · A thought is not text (3:40)

Beats: cloud → feeling, image, memory → things without names → right-side lines.

> That's problem one. Problem two is about what a thought even is.
>
> A thought is not a sentence. It's not a picture either. Right now you can
> feel several things, picture several things, and reach back to several
> moments in the past, all at the same time. None of that is words. When you
> try to describe a thought you lose most of it, and the way one thought
> changes into the next can't be described at all.

## 9 · For an AI, thought is text (4:20)

Beats: bubble → loop → "reads it back".

> In a modern model, thought is text. When it "reasons", it writes words to
> itself and reads them back. Its inner monologue is a transcript. People see
> that transcript and think the model is thinking, because that's what thinking
> would look like if you wrote it down.
>
> But if a thought can't be spelled, the model can't have it.

## 10 · The middle is a filter (4:55)

Beats: text in → text out → three pills → two lines.

> Look at what actually happens between the input and the output. Text enters,
> gets transformed through fixed layers, leaves as text. Nothing is held. The
> only state that survives is the words already produced.
>
> There's no place for a hunch. No place for a mood. No place for two options
> being weighed before either has a name. It doesn't think. It processes. It's
> just very good at producing what thought looks like written down.
>
> A system that can only hold what it can say can form a complex sentence. Not
> a complex thought.

## 11 · What we're building (5:30)

Beats: sensors → hidden → outputs → two lines.

> So here's what we're trying. A brain where every node connects to every
> other node, including itself. No layers, no direction. It ticks continuously.
> The board is written into its sensor nodes while it's already running, and
> it keeps running after it moves. A move happens when an output crosses a
> threshold. The brain decides when it has decided.
>
> And the hidden nodes in the middle are never read as text. They're just what
> the brain is doing right now. That's the closest thing in this project to a
> thought.

## 12 · Outro (6:05)

> The first test is tic-tac-toe. Nobody tells it the rules. It sees the board,
> it fires, and it either survives or it doesn't.
>
> Next video: how a brain like this is grown instead of trained.
