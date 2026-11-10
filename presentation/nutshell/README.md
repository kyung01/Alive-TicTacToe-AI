# "AI can't think" deck (flat-vector explainer style)

A 12-slide YouTube deck for the two problems in `docs/problem-structure.md` and
`docs/problem-thought.md`. Deep-navy flat-vector look with generated hero
illustrations and animated SVG fallbacks.

| File | Purpose |
|---|---|
| `index.html` | The deck. Self-contained, 1920x1080, keyboard driven |
| `script.md` | Voiceover per slide, about 6 minutes |
| `img/nutshell-*.png` | Generated illustrations (copies of `ChatGPT/nutshell-*.png`) |
| `robot-blue.png`, `robot-red.png` | The project's robot avatars, used on a few slides |

## Run it

```bash
python -m http.server 8777 --directory presentation
```

Then open http://localhost:8777/nutshell/. Double-clicking `index.html` also
works (only the Google font needs the network).

## Controls

| Key | Action |
|---|---|
| → · Space · click right half | next beat, then next slide |
| ← · click left half | previous beat / slide |
| A | reveal every beat on the slide |
| V | toggle painted (generated PNG) vs vector (animated SVG) art |
| F | fullscreen · H hides the corner hint |
| Home / End | first / last slide |

The URL hash tracks the slide (`#5`).

## Regenerating an illustration

Each slide's picture has a prompt at `ChatGPT/nutshell-<name>.prompt.txt`.
Edit the prompt, then from the project root:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools/codex_image.ps1 -Prompts ChatGPT/nutshell-<name>.prompt.txt -Force
```

and copy the new `ChatGPT/nutshell-<name>.png` into `presentation/nutshell/img/`.
Run one prompt at a time; the script picks the newest image Codex wrote, so
parallel runs would copy the same picture.

## Recording

Set the window to 1920x1080, press F, press H, record at 1080p60, and read
`script.md` while pressing → on each beat.
