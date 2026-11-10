"""Build the sound effects for the video from the user's sound library: four sets (a-d) of the same five
   kinds of sound, so that two slides in a row never use the same files, plus the game sounds of slide 18."""
import subprocess, wave, os, numpy as np
FF = r"C:\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
LIB = r"D:\Projects\Audio File Storage\Ultimate SFX Bundle - HD Remaster"
OUT = r"C:\Projects\AI Brain\presentation\video\sfx"; os.makedirs(OUT, exist_ok=True)
SR = 44100

def find(name):
    for d, _, fs in os.walk(LIB):
        if name in fs: return os.path.join(d, name)
    raise FileNotFoundError(name)
def read(name):
    raw = subprocess.run([FF, "-v", "error", "-i", find(name), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)
def trim(x, db=-45, pad=.004):
    thr = np.abs(x).max() * 10 ** (db / 20); idx = np.nonzero(np.abs(x) > thr)[0]
    a = max(0, idx[0] - int(pad * SR)); b = min(len(x), idx[-1] + int(.03 * SR)); return x[a:b]
def filt(x, lo=None, hi=None, order=4):
    n = len(x); X = np.fft.rfft(x, n * 2); f = np.fft.rfftfreq(n * 2, 1 / SR); H = np.ones_like(f)
    if hi: H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo: H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-9)) ** (2 * order))
    return np.fft.irfft(X * H)[:n]
def fades(x, fin=.004, fout=.03):
    a, b = int(fin * SR), int(fout * SR); x = x.copy()
    if a: x[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if b: x[-b:] *= np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x
def glide(x, r0, r1):
    """resample with a playback rate that slides from r0 to r1 (pitch glide)"""
    pos, out = 0.0, []; n = len(x)
    while pos < n - 1:
        i = int(pos); fr = pos - i; out.append(x[i] * (1 - fr) + x[i + 1] * fr); pos += r0 + (r1 - r0) * (pos / n)
    return np.array(out)
def level(x, rms_db=-20, peak_db=-3):
    """same perceived level for every file: RMS of the loud part, with a peak ceiling"""
    e = np.abs(x); act = x[e > e.max() * .1]; g = 10 ** (rms_db / 20) / np.sqrt(np.mean(act ** 2))
    g = min(g, 10 ** (peak_db / 20) / e.max()); return x * g
def save(name, x):
    x = level(x); pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(os.path.join(OUT, name + ".wav"), "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    e = np.abs(x); act = x[e > e.max() * .1]; X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1 / SR)
    print(f"{name:8} {len(x) / SR:.2f}s  peak {20 * np.log10(e.max()):6.1f} dB  rms {20 * np.log10(np.sqrt(np.mean(act ** 2))):6.1f} dB  centroid {np.sum(f * X) / np.sum(X):5.0f} Hz  {len(pcm) * 2 // 1024} KB")


def lead(x, ms=40):
    """cut a slow lead-in so the loudest moment comes within `ms` of the start (sounds must land on the beat)"""
    p = int(np.abs(x).argmax()); a = max(0, p - int(ms / 1000 * SR)); return x[a:]
def pitch(x, r): return glide(x, r, r)
def swish(name): return fades(filt(trim(read(name), -40), lo=150, hi=1800, order=2), .03, .09)

# ---- set a (slide 7 and every fourth slide): round bloop, tiny tick, airy swish, soft chime, rising arpeggio
save("pop-a", fades(filt(trim(read("Pop sound 1.wav")), hi=3500), .003, .03))
save("tick-a", fades(filt(trim(read("Pop sound 5.wav")), hi=4500), .002, .02))
save("swish-a", swish("Arm Swing, Whoosh 3.wav"))
save("sparkle-a", fades(filt(trim(read("Collect star 2.wav"), -40)[: int(.6 * SR)], hi=5000, order=2), .004, .3))
save("morph-a", fades(filt(trim(read("Powerup upgrade 11.wav"), -40), hi=4000, order=2), .01, .2))

# ---- set b: a mallet "tok", a tiny pure blip, a short bright chime
save("pop-b", fades(filt(trim(read("Pop sound 16.wav")), hi=3500), .003, .04))
save("tick-b", fades(filt(pitch(trim(read("Ui Hover over button 17.wav")), .75), hi=3000), .004, .02))
save("swish-b", swish("Arm Swing, Whoosh 13.wav"))
save("sparkle-b", fades(filt(trim(read("Collect star 1.wav"), -40)[: int(.5 * SR)], hi=5000, order=2), .004, .25))
save("morph-b", fades(filt(trim(read("Powerup upgrade 16.wav"), -40), hi=4000, order=2), .01, .15))

# ---- set c: a soft rising "ding", a little rising blip, a three-note chime
save("pop-c", fades(filt(trim(read("Notification sound 3.wav")), hi=3500), .004, .04))
save("tick-c", fades(filt(trim(read("Notification sound 14.wav")), hi=4000)[: int(.14 * SR)], .003, .05))
save("swish-c", swish("Arm Swing, Whoosh 19.wav"))
save("sparkle-c", fades(filt(trim(read("Notification sound 16.wav"), -40), hi=5000, order=2), .004, .2))
save("morph-c", fades(filt(trim(read("Powerup upgrade 13.wav"), -40), hi=4000, order=2), .01, .15))

# ---- set d: a low gentle "bwoop", a falling two-tone blip, a bell
save("pop-d", fades(filt(pitch(lead(trim(read("Pop sound 3.wav"), -30)), 1.3), hi=3500), .008, .04))
save("tick-d", fades(filt(pitch(trim(read("Go back sound 4.wav")), .7), hi=4000), .004, .03))
save("swish-d", swish("Arm Swing, Whoosh 6.wav"))
save("sparkle-d", fades(filt(trim(read("Powerup upgrade 18.wav"), -40)[: int(.45 * SR)], hi=5000, order=2), .004, .2))
save("morph-d", fades(filt(trim(read("Powerup upgrade 15.wav"), -40), hi=4000, order=2), .01, .2))

# ---- slide 18, the bouncing balls: a tiny boing for a bounce, a rising blip for a merge, a falling boing for a split, a pop with a chime for a burst
save("game-bounce", fades(filt(trim(read("Boing sound 1.wav")), hi=4500), .002, .02))
save("game-merge", fades(filt(trim(read("Ui Popup 9.wav"), -40), hi=2200, order=2), .004, .05))
save("game-split", fades(filt(trim(read("Boing sound 2.wav"), -40), hi=4000), .003, .06))
bp = fades(filt(trim(read("Balloon sound (Balloon pop) 1.wav"), -40), hi=2500, order=2), .002, .05)
ch = fades(filt(pitch(trim(read("Collect star 4.wav"), -40), .5), hi=5000, order=2)[: int(.5 * SR)], .004, .3)
mix = np.zeros(max(len(bp), len(ch))); mix[: len(bp)] += bp / np.abs(bp).max(); mix[: len(ch)] += .5 * ch / np.abs(ch).max()
save("game-burst", mix)


# ---- sounds with a meaning, made here from sine waves instead of taken from the library, so that their pitch, length
#      and softness are exact. All are in C major pentatonic, the same scale the pops climb.
C4, D4, E4, G4, A4, C5, D5, E5, G5, C6 = 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99, 1046.5
def tone(f, dur, decay, parts=((1, 1, 1),), attack=.004, bend=None):
    """one struck note: parts are (multiple of f, level, how much faster it dies than the base tone)"""
    t = np.arange(int(dur * SR)) / SR; ph = 2 * np.pi * np.cumsum(f * (bend(t) if bend else np.ones_like(t))) / SR
    x = sum(a * np.sin(k * ph) * np.exp(-t * q / decay) for k, a, q in parts)
    return x * np.minimum(1, t / attack)
def place(n, *items):
    """mix (start second, sound, level) items into one n-second sound"""
    out = np.zeros(int(n * SR))
    for at, x, g in items: a = int(at * SR); x = x[: len(out) - a]; out[a: a + len(x)] += g * x
    return out
MALLET = ((1, 1, 1), (4, .16, 3), (10, .03, 8))        # a wooden bar: dull, short
BELL = ((1, 1, 1), (2.76, .18, 2), (5.4, .06, 4))      # a small metal bar: rings on
soft = lambda x, hi=5000, fout=.05: fades(filt(x, hi=hi, order=2), .002, fout)

# note: one step in a sequence (layers, counting). The page plays it higher for each next step.
save("note", soft(tone(C5, .6, .16, MALLET)))
# ring: something that is kept (a memory, a taste): the same note left ringing
save("ring", soft(tone(C5, 2.8, .75, BELL), fout=.4))
# blip: a signal arrives
save("blip", soft(tone(880, .13, .05, ((1, 1, 1),), .006, bend=lambda t: 1 + .12 * np.minimum(1, t / .06)), fout=.03))
# ding: a result comes out
save("ding", soft(tone(C6, 1.3, .32, ((1, 1, 1), (2, .3, 1.6), (3, .1, 2.5), (4.2, .05, 4))), hi=6000, fout=.2))
# chord: many things at once that are one whole (four notes, struck almost together)
warm = ((1, 1, 1), (2, .2, 2), (3, .05, 3))
save("chord", soft(place(2.6, (0, tone(C4, 2.6, .8, warm, .03), 1), (.025, tone(G4, 2.5, .75, warm, .03), .8),
                         (.05, tone(E5, 2.5, .65, warm, .03), .7), (.075, tone(D5 * 2, 2.4, .55, warm, .03), .45)), hi=4000, fout=.4))
# nope: not this way / wrong. Two low wooden notes going down, like "uh-uh"
save("nope", soft(place(.5, (0, tone(E4, .3, .07, MALLET), 1), (.13, tone(C4, .37, .1, MALLET), .9)), hi=2200))
# down: something stops or dies. A tone that sinks and fades
save("down", soft(tone(G4, .8, .3, ((1, 1, 1), (2, .15, 2)), .02, bend=lambda t: .36 ** np.minimum(1, t / .7)), hi=3000, fout=.15))
# up: something starts or wakes. A tone that rises, then a small note on top
save("up", soft(place(.85, (0, tone(G4 / 2, .4, .5, ((1, 1, 1), (2, .15, 2)), .05, bend=lambda t: 3.36 ** np.minimum(1, t / .36)) * np.minimum(1, (.4 - np.arange(int(.4 * SR)) / SR) / .06), .6),
                       (.34, tone(E5, .5, .14, MALLET), 1)), hi=4000))
# ask: a question. Two notes going up, the second one bending upward the way a voice does
save("ask", soft(place(.75, (0, tone(G4, .3, .09, MALLET), .9), (.15, tone(D5, .6, .2, ((1, 1, 1), (4, .1, 3)), bend=lambda t: 1 + .06 * np.clip((t - .05) / .2, 0, 1)), 1))))
# key: one soft press of a button;  type: a few of them, uneven like typing (text being written)
rng = np.random.default_rng(7)
def press(f=230, g=1): n = int(.09 * SR); t = np.arange(n) / SR; return g * (filt(rng.standard_normal(n) * np.exp(-t / .006), lo=900, hi=3500, order=2) * .5 + np.sin(2 * np.pi * f * t) * np.exp(-t / .022))
save("key", fades(press(), .001, .02))
save("type", fades(place(.46, (0, press(250), 1), (.075, press(300), .7), (.16, press(235), .9), (.225, press(280), .65), (.31, press(260), .85), (.37, press(310), .5)), .001, .03))
# whirl: going round in a loop. Up the scale and back to where it started
save("whirl", soft(place(.95, *[(.075 * i, tone(f, .45, .1, MALLET), 1 - .06 * i) for i, f in enumerate([C5, E5, G5, C6, G5, E5, C5])])))
