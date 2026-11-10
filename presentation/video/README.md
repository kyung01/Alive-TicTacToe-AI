# "AI can't think" — self-playing video

An animated explainer that plays itself, presented by two ladybug characters in
an evening forest: the blue one with glasses is the presenter, the red one is the
sidekick who reacts. 34 slides, each with its own picture and animation and at
most three sentences of narration; about 6.7 minutes with the recorded narration.
Several slides are parables (the factory line, the strawberry, and the blind men
and the elephant told over five slides) that restate an idea as an everyday
picture. The forest moves through four locations (clearing, brook,
mushroom grove, dawn meadow) that crossfade and drift slowly, with swaying
leaves at the corners and fireflies, so no frame is ever still.
Every slide has its own audio file and marker on the bar. Diagrams are
inline SVG; real objects, the characters and the backdrop are generated images.

| File | Purpose |
|---|---|
| `index.html` | The video player and all the scenes |
| `script.md` | Transcript, one block per slide (34) |
| `audio/NN.mp3` | Narration, one file per slide (01.mp3 … 34.mp3); picked up automatically |
| `audio-old/` | Takes that were replaced, and the narration of two removed slides; not used by the player |
| `sfx/*.wav` | Sound effects (see below); made by `tools/build_sfx.py` |
| `img/bg-forest*.png`, `img/fg-leaves-*.png`, `img/parable-*.png` | The four forest locations, the two swaying leaf overlays and the seven parable pictures; prompts beside the copies in `ChatGPT/` |
| `img/char-*.png` | Character poses (blue: idle, point-right, explain, think, no, cheer; red: idle, surprised, sleep, confused, love, dizzy, peek) and the forest backdrop; prompts beside the copies in `ChatGPT/` |
| `img/icon-*.png` | Generated flat icons (brain, eye, ear, hand, mouth, calculator, funnel, scale, bulb, heart-cloud, ruler, baby, tombstone); copies of `ChatGPT/icon-*.png`, prompts beside them |

## Play it

```bash
python -m http.server 8777 --directory presentation
```

Open http://localhost:8777/video/ and press Space. Double-clicking
`index.html` also works (only the Google font needs the network).

| Key | Action |
|---|---|
| Space | play / pause |
| ← → | previous / next slide |
| , . | step one second back / forward inside the slide |
| M | sound effects on / off |
| K | clean mode: hide the controls and hint (for recording) |
| F | fullscreen |
| click on the bar | seek anywhere |

URL options: `#5` starts on slide 5; `?t=3` seeks 3 s into it;
`?clean` hides the UI; `?sfx=0.5` sets the level of the sound effects
(1 is the default, 0 is off).

## Adding the voice

1. Record each block of `script.md` as its own file.
2. Save them as `audio/01.mp3`, `audio/02.mp3`, … (mp3, m4a also works if you
   change `audioFor` in `index.html`).
3. Reload. A slide with an audio file plays it and waits for the recording to
   finish, then holds for 0.5 s (`PAUSE` in `index.html`) before the next slide.
   All narration plays through one audio player (`voice`), because a phone only lets a page start
   sound on a player the viewer's tap has already started. Over http(s) the sound effects use Web Audio
   for the same reason; from `file://` they fall back to `<audio>` elements.
   The pause is counted from the moment the audio really ends, so a recording
   that starts late is not cut. Slides without audio keep their placeholder length.

Each slide in `index.html` is one `S(...)` call (or `scenes.push`): name,
placeholder seconds, narration, kicker, headline, the SVG art, and a `run`
function whose `at` values are the animation beats in seconds from the slide
start. After the voice is in, nudge a beat that lands too early or late.

A scene marked `carry` (set after the scene list: slides 4, 16, 18, 21 and 26 to 29) keeps a
drawing from the previous slide in place, so it skips the entrance pop that
every other slide gets, and its `run` must not fade or pop that drawing in
again. Use the same viewBox on both slides, or the drawing jumps.

## Backgrounds and parables

The forest is a four-depth parallax set. Each place has a far painting
(`img/bg-<place>-far.png`: sky, light and the distant tree line, barely moving)
and a transparent trees layer (`img/bg-<place>-trees.png`: that place's trunks,
bushes, roots and ground, drifting more and leaning slightly). In front of
those, a mid layer of generic foliage pieces (`img/mid-bushes.png`,
`mid-vines.png`, `mid-trunks.png`; each place shows its own mix, some mirrored)
drifts further and sways, and the near leaf overlays (`img/fg-leaves-*.png`)
drift most and sway hardest. The single full paintings (`img/bg-<place>.png`)
remain as a fallback when a far layer is missing. A gust (`--g`) multiplies all
sway when the place changes and calms over a few seconds; `BG_LIGHTS` adds
breathing glows (the hollow, fungi, the moon) per place. `BG_PLAN` in index.html names the slide that starts each new place
(twelve places over the slides); every slide after it keeps that place
until the next cut.

Each slide carries a `bg` name (set through the kicker constants `K1`, `K2`, `K3`,
`KP1`, `KP2` or directly on the title and outro); `setBg` crossfades between two
drifting layers. Parable slides use `P1(...)` (one picture) or `P2(...)` (two
pictures with a "vs" badge); their pictures are `img/parable-*.png`. The
elephant parable is one shared scene (`ELEPH(...)`) framed five ways: the dark
elephant is the thought nobody can see, the three men (`img/blind-man-*.png`)
step up one per slide and get a snake, tree or rope icon in their bubble, and
the last slide dims it all under a question mark.

## Sound effects

The animation helpers note a cue whenever something appears, moves, is drawn or
changes; `score()` in `index.html` thins the cues out and the player plays them.
Things that disappear make no sound, and neither does the change of slide.

There are four sets of the same five kinds of sound: `pop`, `tick`, `swish`,
`sparkle` and `morph`, as `sfx/pop-a.wav` … `sfx/morph-d.wav`. Each slide uses
the next set (`setOf`; slide 7 has set a), so two slides in a row never share a
sound. A row of pops or ticks climbs a small scale.

Slide 18 is a game: its balls play `sfx/game-bounce.wav`, `game-merge.wav`,
`game-split.wav` and `game-burst.wav`, lower for bigger balls, about four sounds
a second. The waits in `gameSfx` set how many are heard.

### Sounds that carry meaning

On top of the automatic sounds, a slide can place a sound by hand with
`snd(name, at, rate = 1, vol = 1)` (`at` is seconds into the slide, `st(n)`
gives the rate for `n` semitones up). Each one always means the same thing, so
the ear learns it:

| Sound | Means |
|---|---|
| `note` | one step in a sequence (a higher note for each step) |
| `ring` | something that is kept or remembered |
| `blip` | a signal arrives |
| `ding` | a result comes out |
| `chord` | many things at once, or one whole |
| `nope` | not this way |
| `down` | something stops or dies |
| `up` | something starts or wakes |
| `ask` | a question |
| `type` | text is written |
| `key` | a button is pressed |
| `whirl` | going round in a loop |

An automatic sound that falls on a placed one is dropped. Two slides in a row
never use the same placed sound: run `soundCheck()` in the console, and
`shared` must come back empty. The beats of each slide sit on the narrator's
words (times estimated from the recordings), so a sound lands when the idea is
said. Slide 7 has only automatic sounds; slide 18 only its game sounds.

The `SFX` table holds the level of each sound (the narration is 1).
`tools/build_sfx.py` cuts, softens and levels the files from the sound library
in `D:\Projects\Audio File Storage`; change a source file there and run it again.

## Characters

`CAST` near the end of `index.html` places the characters per slide:
`[pose, x, y, width, startSeconds, {flip, until, enter}]` in 1920x1080 stage
coordinates. `enter` is `rise` (default), `pop` or `slide`; `until` fades a pose
out so another can replace it. Every character bobs and sways on its own.

## Regenerating an icon or a character

Each icon has a prompt at `ChatGPT/icon-<name>.prompt.txt`. Edit it, then from the
project root run one at a time:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/codex_image.ps1 -Prompts ChatGPT/icon-<name>.prompt.txt -Force
```

and copy the new PNG into `presentation/video/img/`. Real objects are icons;
abstract shapes (pipes, nodes, bubbles, lines, the tangle) stay as inline SVG.

## Recording the final video

Press K (clean) and F (fullscreen) on a 1920x1080 display, press Space, and
screen-record at 1080p60 with the browser's audio. Alternatively render frames
headlessly with Chrome (`?t=` plus `#scene`) and assemble with ffmpeg.

## Narration with ElevenLabs

The current recordings were made on elevenlabs.io with the voice "Adam - American,
Dark and Tough" on Eleven v4 (Stability 0.5, Similarity 0.1, Clone strength 1.0).
The exact lines sent, with their [audio tags], are in `narration-tagged.md`; "AGI"
is written "A G I" there so it is spelled out. To redo one slide, paste its line
into the website, generate, and replace `audio/NN.mp3`.

If the last word of a take sounds clipped, add a throwaway sentence after it
(slide 1 was sent with ` [long pause] Here is why.` at the end) and cut the file in
the silent gap before that sentence, for example
`ffmpeg -i take.mp3 -t 6.90 -af "afade=t=out:st=6.80:d=0.10" -b:a 192k 01.mp3`.


`tools/elevenlabs_tts.py` reads `script.md` and writes `audio/NN.mp3` for every
slide in your own ElevenLabs voice. The key is read from the environment and is
never stored in the project:

    $env:ELEVENLABS_API_KEY = "<your key>"            # PowerShell
    python tools/elevenlabs_tts.py --list-voices       # find your voice id
    python tools/elevenlabs_tts.py --voice <id> --only 1
    python tools/elevenlabs_tts.py --voice <id>        # all slides, skips existing files

`--model eleven_v4` is more expressive; the default `eleven_multilingual_v2` is the
most stable for long narration. Each request sends the previous and next slide's
text as context so the delivery flows across slides. After editing a slide's
narration, regenerate `script.md` and rerun with `--only <n> --force`.
