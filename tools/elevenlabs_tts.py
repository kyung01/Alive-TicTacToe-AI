"""Narrate the video with your own ElevenLabs voice.

Reads presentation/video/script.md (one block per slide) and writes
presentation/video/audio/NN.mp3 for every slide. The player picks those files up
and stretches each slide to its recording's length.

Your API key is never written into this project. Put it in an environment
variable before running:

    PowerShell:  $env:ELEVENLABS_API_KEY = "<your key>"
    Git Bash:    export ELEVENLABS_API_KEY="<your key>"

(Find the key on elevenlabs.io under your profile, "API Keys".)

Usage:
    python tools/elevenlabs_tts.py --list-voices          # find your voice's id
    python tools/elevenlabs_tts.py --voice <id> --only 1  # try one slide first
    python tools/elevenlabs_tts.py --voice <id>           # all slides (skips ones already made)
    python tools/elevenlabs_tts.py --voice <id> --force   # redo everything
    python tools/elevenlabs_tts.py --dry-run              # show what would be sent, no API calls

Options:
    --model     eleven_multilingual_v2 (default, most stable for long narration)
                or eleven_v4 / eleven_v3 (more expressive)
    --only      comma list of slide numbers, e.g. 3,7,12
    --stability, --similarity, --style, --speed   voice settings (see ElevenLabs docs)
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "presentation", "video", "script.md")
OUT = os.path.join(ROOT, "presentation", "video", "audio")
API = "https://api.elevenlabs.io/v1"


def slides():
    text = open(SCRIPT, encoding="utf-8").read()
    blocks = re.findall(r"^## (\d+) · (.+?)\n\n(.+?)(?=\n\n## |\Z)", text, re.S | re.M)
    return [(int(n), name.strip(), body.strip()) for n, name, body in blocks]


def call(path, key, body=None, query=""):
    req = urllib.request.Request(API + path + query, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "*/*"},
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:400]
        if e.code == 401: sys.exit("401: the API key was rejected. Check ELEVENLABS_API_KEY.")
        if e.code == 403: sys.exit(f"403: your plan does not allow this model or feature.\n{msg}")
        if e.code == 429: return None  # rate limited, caller retries
        sys.exit(f"HTTP {e.code}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice"); ap.add_argument("--model", default="eleven_multilingual_v2")
    ap.add_argument("--only"); ap.add_argument("--force", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--stability", type=float, default=0.5); ap.add_argument("--similarity", type=float, default=0.8)
    ap.add_argument("--style", type=float, default=0.0); ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    items = slides()
    if a.only: keep = {int(x) for x in a.only.split(",")}; items = [s for s in items if s[0] in keep]
    if a.dry_run:
        for n, name, body in items: print(f"{n:02d} {name} ({len(body)} chars): {body[:90]}...")
        print(f"\n{len(items)} slides, {sum(len(b) for _, _, b in items)} characters in total (that is what ElevenLabs bills)."); return

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key: sys.exit("Set ELEVENLABS_API_KEY first (see the top of this file).")

    if a.list_voices:
        data = json.loads(call("/voices", key))
        for v in data.get("voices", []):
            print(f"{v['voice_id']}  {v.get('category', ''):12}  {v['name']}")
        print("\nYour own voices are usually category 'cloned', 'generated' or 'professional'."); return

    if not a.voice: sys.exit("Pass --voice <id> (run --list-voices to find it).")
    os.makedirs(OUT, exist_ok=True)
    allslides = slides(); text_of = {n: b for n, _, b in allslides}
    for n, name, body in items:
        dst = os.path.join(OUT, f"{n:02d}.mp3")
        if os.path.exists(dst) and not a.force: print(f"{n:02d} exists, skipped"); continue
        payload = {"text": body, "model_id": a.model,
                   "voice_settings": {"stability": a.stability, "similarity_boost": a.similarity, "style": a.style, "use_speaker_boost": True, "speed": a.speed},
                   # neighbouring slides as context, so the delivery flows from one slide into the next
                   "previous_text": text_of.get(n - 1, ""), "next_text": text_of.get(n + 1, "")}
        for attempt in range(5):
            audio = call(f"/text-to-speech/{a.voice}", key, payload, "?output_format=mp3_44100_128")
            if audio: break
            time.sleep(5 * (attempt + 1))
        else: sys.exit("Rate limited too many times; try again later.")
        open(dst, "wb").write(audio); print(f"{n:02d} {name}: {len(audio) // 1024} KB")
    print("done")


if __name__ == "__main__":
    main()
