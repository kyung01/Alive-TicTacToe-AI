"""Builds the soundtrack of the exported video, exactly as the player would have played it.

    python tools/mix_audio.py            -> presentation/video/export/soundtrack.wav (48 kHz stereo)

Reads presentation/video/export/render_events.json, written by tools/render_video.mjs:
  - narration: each slide's recording starts when its slide starts (level 1, as in the player);
  - sound effects: every sound the page asked for, at its time, playback rate and level, with the same
    "at most N of one sound at once" rule the player uses (the oldest one is cut);
  - music: forest loop 1 from the start (1.2 s fade-in), a 3 s cross-fade to loop 2 at "Problem two",
    a 3 s cross-fade to the credits piece when the end screen starts (started late enough that its own
    ending lands on the last frame), all at the player's music level.
"""
import json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "presentation", "video")
EXP = os.path.join(V, "export")
FF = r"C:\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
SR = 48000


def decode(path, ch):
    raw = subprocess.run([FF, "-v", "quiet", "-i", path, "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    return a.reshape(-1, ch).copy() if ch > 1 else a.copy()


def ramp(env, t0, dur, a, b):
    """linear gain ramp from a to b over [t0, t0+dur], and b after it"""
    i0, i1 = int(t0 * SR), int((t0 + dur) * SR)
    n = len(env)
    if i0 < n:
        seg = np.linspace(a, b, max(1, i1 - i0), endpoint=False)
        env[i0:min(i1, n)] = seg[:max(0, min(i1, n) - i0)]
    env[min(i1, n):] = b


def main():
    info = json.load(open(os.path.join(EXP, "render_events.json"), encoding="utf-8"))
    total = info["to"]
    N = int(round(total * SR))
    mix = np.zeros((N, 2), np.float32)

    # narration
    for s in info["scenes"]:
        if not s["file"]:
            continue
        a = decode(os.path.join(V, s["file"]), 1)
        i = int(round(s["start"] * SR)); j = min(N, i + len(a))
        if i < N:
            mix[i:j] += a[:j - i, None]
    print("narration placed:", sum(1 for s in info["scenes"] if s["file"]))

    # sound effects, with the player's per-sound voice limit
    cache, voices = {}, {}
    for tm, key, rate, vol, limit in sorted(info["events"], key=lambda e: e[0]):
        if key not in cache:
            cache[key] = decode(os.path.join(V, "sfx", key + ".wav"), 1)
        src = cache[key]
        n = int(len(src) / rate)
        y = np.interp(np.arange(n) * rate, np.arange(len(src)), src).astype(np.float32) * vol
        i = int(round(tm * SR))
        live = voices.setdefault(key, [])
        live[:] = [v for v in live if v["end"] > i]
        if len(live) >= limit:  # as many at once as the pool allows: the oldest stops
            old = live.pop(0); cut = i - old["start"]
            if 0 < cut < len(old["y"]):
                f = min(480, cut); old["y"] = old["y"][:cut].copy(); old["y"][-f:] *= np.linspace(1, 0, f); old["end"] = i
        v = {"start": i, "end": i + n, "y": y}; live.append(v); voices.setdefault("_all", []).append(v)
    for v in voices.get("_all", []):
        i, y = v["start"], v["y"]; j = min(N, i + len(y))
        if i < N:
            mix[i:j] += y[:j - i, None]
    print("sound effects placed:", len(info["events"]))

    # music
    m = info["music"]; vol = m["vol"]; P, E = m["problemTwo"], m["endStart"]
    tracks = [decode(os.path.join(V, f), 2) for f in m["files"]]
    t = np.arange(N) / SR
    env0, env1, env2 = np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
    ramp(env0, 0, 1.2, 0, vol); ramp(env0, P, 3, vol, 0)
    ramp(env1, P, 3, 0, vol); ramp(env1, E, 3, vol, 0)
    ramp(env2, E, 3, 0, vol)
    for k, env in ((0, env0), (1, env1)):
        tr = tracks[k]; idx = np.arange(N) % len(tr)   # loops that run from the first frame
        mix += tr[idx] * env[:, None]
    cr = tracks[2]; endDur = total - E; off = max(0.0, len(cr) / SR - endDur)
    i0 = int(round(E * SR)); k0 = int(round(off * SR)); n = min(N - i0, len(cr) - k0)
    mix[i0:i0 + n] += cr[k0:k0 + n] * env2[i0:i0 + n, None]
    print("music: loop 1 -> loop 2 at %.2f s -> credits at %.2f s (credits start %.2f s into the piece)" % (P, E, off))

    # a gentle fade on the very last half second, and a safety ceiling
    f = int(.5 * SR); mix[-f:] *= np.linspace(1, 0, f)[:, None]
    peak = float(np.abs(mix).max()); print("peak before ceiling: %.2f" % peak)
    if peak > .98:
        mix *= .98 / peak
    out = os.path.join(EXP, "soundtrack.wav")
    subprocess.run([FF, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_s24le", out], input=mix.astype(np.float32).tobytes(), check=True)
    print("wrote", out, "%.2f s" % (N / SR))


if __name__ == "__main__":
    main()
