# Presentation

A 12-slide deck for a YouTube-style video, plus the narration script.

| File | Purpose |
|---|---|
| `index.html` | The deck. Self-contained, 1920×1080, dark theme, animated diagrams |
| `script.md` | Voiceover per slide with rough timestamps (about 7 minutes) |

## Run it

```bash
python -m http.server 8777 --directory presentation
```

Then open http://localhost:8777. (Double-clicking `index.html` also works; only
the Google font needs the network, and it falls back to the system font.)

## Controls

| Key | Action |
|---|---|
| → · Space · click right side | next beat, then next slide |
| ← · click left side | previous beat / slide |
| A | reveal every beat on the current slide |
| F | fullscreen |
| H | hide the help hint in the corner |
| Home / End | first / last slide |

The URL hash tracks the slide (`#5`), so you can jump straight to one.

## Recording

1. Set the display or browser window to 1920×1080 and press **F**.
2. Press **H** to hide the corner hint.
3. Record with OBS or any screen recorder at 1080p, 60 fps (slides 5, 6 and 10
   animate continuously).
4. Read `script.md` as voiceover, pressing → on each beat.

## Editing

Slides are `<section class="slide">` blocks in `index.html`. Elements with the
class `frag` appear one at a time. The animated diagrams are small inline scripts
at the bottom of the file, one per slide, registered in `hooks[slideIndex]` so
they only run while their slide is visible.

Numbers in the results slide are from `docs/experiments.md`. Update both when a
run finishes.
