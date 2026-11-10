"""
Live training dashboard. Serves a page that re-reads a training log every few
seconds and plots lifespan, fitness and the best brain's outcome mix.

    python dashboard.py [--port 8765] [--log train_gpu_kill.log]
"""
import argparse
import glob
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

GEN_RE = re.compile(
    r"gen\s+(\d+)\s+best=([+-][\d.]+)\s+mean=([+-][\d.]+)"
    r"(?:\s+survivors=\s*(\d+)\s+life=\s*([\d.]+)/(\d+))?"
    r"(?:\s+alive=\s*(\d+))?"
    r"\s+ticks=\s*([\d.]+)\s+win=\s*([\d.]+)%\s+draw=\s*([\d.]+)%\s+loss=\s*([\d.]+)%"
    r"\s+illegal=\s*([\d.]+)%\s+timeout=\s*([\d.]+)%\s+\[\s*(\d+)s(?:\s+vram=([\d.]+)GB)?\]")
EVAL_RE = re.compile(
    r"eval vs (\w+)\s+\((\d+) games\): win=\s*([\d.]+)% draw=\s*([\d.]+)% loss=\s*([\d.]+)%"
    r" illegal=\s*([\d.]+)% timeout=\s*([\d.]+)%\s+ticks=([\d.]+)")


CONT_RE = re.compile(
    r"game\s+(\d+)\s+age mean=\s*([\d.]+) max=\s*(\d+) ever=\s*(\d+)\s+deaths=\s*(\d+)\s+births=\s*(\d+)"
    r"\s+win=\s*([\d.]+)% draw=\s*([\d.]+)% loss=\s*([\d.]+)% illegal=\s*([\d.]+)% timeout=\s*([\d.]+)%"
    r"\s+ticks=\s*([\d.]+).*?\[\s*(\d+)s\s+vram=([\d.]+)GB\]")


SINGLE_RE = re.compile(
    r"game\s+(\d+)\s+win=\s*([\d.]+)% draw=\s*([\d.]+)% loss=\s*([\d.]+)% illegal=\s*([\d.]+)% timeout=\s*([\d.]+)%"
    r"(?:\s+step=(\d+))?(?:\s+lr=([\d.eE+-]+))?(?:\s+\|M\|=([\d.]+) \|G\|=([\d.]+) \|r\|=([\d.]+))?"
    r"\s+ticks=\s*([\d.]+)\s+\|W\|=([\d.]+) Wmax=([\d.]+).*?\[\s*(\d+)s\]")


EVO_RE = re.compile(
    r"^GEN\s+(\d+)\s+best=([+-][\d.]+)\s+mean=([+-][\d.]+)\s+median=([+-][\d.]+).*?"
    r"win=\s*([\d.]+)% loss=\s*([\d.]+)% illegal=\s*([\d.]+)% draw=\s*([\d.]+)% freeze=\s*([\d.]+)%"
    r"(?:\s*\| exam vs random: win=\s*([\d.]+)% loss=\s*([\d.]+)% illegal=\s*([\d.]+)% freeze=\s*([\d.]+)%)?"
    r"(?:\s*\| oppdied=\s*([\d.]+)%)?"
    r"(?:\s*\| vs (?:random|peer): [^|]*)?"                 # both mode: the scripted-half split, not charted
    r"(?:\s*\| vs perfect: loss=\s*([\d.]+)% illegal=\s*([\d.]+)% draw=\s*([\d.]+)% freeze=\s*([\d.]+)%)?"
    r".*?\[\s*(\d+)s\]")
KIND_RE = re.compile(r"\(last \d+ (\w+) games\)")      # who the best genome played: random / perfect / mixed / peer
# the whole population's game, one line per game: how the world went for everyone, not just the best genome
POP_RE = re.compile(r"^gen\s+(\d+) game\s+\d+\s+all brains: win=\s*([\d.]+)% loss=\s*([\d.]+)% illegal=\s*([\d.]+)% freeze=\s*([\d.]+)%"
                    r"(?:\s+ticks/game=\s*([\d.]+))?(?:\s+illegal fires/brain=\s*([\d.]+))?")


def parse(path):
    rows, evals, header = [], [], ""
    last_gen = 0
    pop = {}                       # gen -> the population's last game line
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                pm = POP_RE.search(line)
                if pm:
                    g = pm.groups()
                    w, l, ill, fr = (float(g[1]), float(g[2]), float(g[3]), float(g[4]))
                    # per brain the outcomes sum to 100; in peer games a timeout on one side is an
                    # "opponent died" on the other, so draws are what is left after both are removed
                    draw = max(0.0, 100.0 - w - l - ill - 2 * fr)
                    pop[int(g[0])] = {"pop_win": w, "pop_loss": l, "pop_illegal": ill, "pop_freeze": fr,
                                      "pop_draw": draw, "pop_board": w + l + draw,
                                      "pop_ticks": float(g[5]) if g[5] else None, "pop_fires": float(g[6]) if g[6] else None}
                    continue
                m = EVO_RE.search(line)
                if m:
                    g = m.groups()
                    last_gen = int(g[0])
                    # evolution-only mode: one row per generation; win/loss are the best genome's
                    # last-third-of-life rates; "best"/"mean" are fitness (external score per game)
                    rows.append({
                        "gen": last_gen, "best": float(g[1]), "mean": float(g[2]),
                        "survivors": None, "life_mean": None, "life_max": None,
                        "ticks": 0.0, "win": float(g[4]), "draw": float(g[7]),
                        "loss": float(g[5]), "illegal": float(g[6]), "timeout": float(g[8]),
                        "exam_win": float(g[9]) if g[9] else None, "exam_loss": float(g[10]) if g[10] else None,
                        "exam_illegal": float(g[11]) if g[11] else None, "exam_freeze": float(g[12]) if g[12] else None,
                        "oppdied": float(g[13]) if g[13] else None,
                        "kind": (KIND_RE.search(line).group(1) if KIND_RE.search(line) else "random"),
                        "pf_loss": float(g[14]) if g[14] else None, "pf_illegal": float(g[15]) if g[15] else None,
                        "pf_draw": float(g[16]) if g[16] else None, "pf_freeze": float(g[17]) if g[17] else None,
                        "sec": int(g[18]), "vram": None})
                    continue
                m = SINGLE_RE.search(line)
                if m:
                    g = m.groups()
                    last_gen = int(g[0])
                    # single-brain mode: no ages or deaths; "best" = mean |W|, "mean" = max |W|
                    step = int(g[6]) if g[6] else None
                    if step is not None:
                        last_gen = step          # x axis = cumulative training step when the log has it
                    rows.append({
                        "gen": last_gen, "best": float(g[12]), "mean": float(g[13]),
                        "survivors": None, "life_mean": None, "life_max": None,
                        "ticks": float(g[11]), "win": float(g[1]), "draw": float(g[2]),
                        "loss": float(g[3]), "illegal": float(g[4]), "timeout": float(g[5]),
                        "lr": float(g[7]) if g[7] else None,
                        "memM": float(g[8]) if g[8] else None, "memG": float(g[9]) if g[9] else None,
                        "memr": float(g[10]) if g[10] else None,
                        "sec": int(g[14]), "vram": None})
                    continue
                m = CONT_RE.search(line)
                if m:
                    g = m.groups()
                    last_gen = int(g[0])
                    # continuous mode: x = games played, "life" = age (games survived),
                    # "survivors" = deaths in the window, best = oldest brain ever
                    rows.append({
                        "gen": last_gen, "best": float(g[3]), "mean": float(g[1]),
                        "survivors": int(g[4]), "life_mean": float(g[1]), "life_max": float(g[2]),
                        "ticks": float(g[11]), "win": float(g[6]), "draw": float(g[7]),
                        "loss": float(g[8]), "illegal": float(g[9]), "timeout": float(g[10]),
                        "sec": int(g[12]), "vram": float(g[13])})
                    continue
                m = GEN_RE.search(line)
                if m:
                    g = m.groups()
                    last_gen = int(g[0])
                    rows.append({
                        "gen": last_gen, "best": float(g[1]), "mean": float(g[2]),
                        "survivors": int(g[3]) if g[3] else None,
                        "life_mean": float(g[4]) if g[4] else None,
                        "life_max": float(g[5]) if g[5] else None,
                        "ticks": float(g[7]), "win": float(g[8]), "draw": float(g[9]),
                        "loss": float(g[10]), "illegal": float(g[11]),
                        "timeout": float(g[12]), "sec": int(g[13]),
                        "vram": float(g[14]) if g[14] else None})
                    continue
                m = EVAL_RE.search(line)
                if m:
                    evals.append({"gen": last_gen, "opp": m[1], "games": int(m[2]),
                                  "win": float(m[3]), "draw": float(m[4]),
                                  "loss": float(m[5]), "illegal": float(m[6]),
                                  "timeout": float(m[7]), "ticks": float(m[8])})
                    continue
                if line.startswith(("device=", "nodes:")):
                    header = line.strip()
    except FileNotFoundError:
        pass
    for r in rows:
        if "gen" in r and r["gen"] in pop:
            r.update(pop[r["gen"]])
    games = rows[-1]["life_max"] if rows and rows[-1]["life_max"] is not None else None
    m = re.search(r"games/brain/gen=(\d+)", header)
    if m:
        games = int(m[1])
    finished = any("saved best brain" in l or "saved oldest brain" in l or "saved brain" in l for l in tail(path, 3))
    single = header.startswith("device=") and "single brain" in header
    evo = header.startswith("device=") and "evolution-only" in header
    pair = evo and "opponent=peer" in header          # brains only play each other: relative fitness, exams are the yardstick
    # continuous.py's life/death loop; the evo trainer's "continuous game" header is not that mode
    continuous = header.startswith("device=") and "continuous" in header and not evo
    return {"log": os.path.basename(path), "header": header, "games": games,
            "rows": rows, "evals": evals, "finished": finished, "continuous": continuous,
            "single": single, "evo": evo, "pair": pair}


def tail(path, n):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dashboard.html"), encoding="utf-8").read()


class Handler(BaseHTTPRequestHandler):
    log_default = "train_single.log"

    def log_message(self, *a):
        pass

    def send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/data":
            q = parse_qs(u.query)
            log = os.path.basename(q.get("log", [self.log_default])[0])
            self.send(200, "application/json", json.dumps(parse(log)))
        elif u.path.startswith("/sounds/"):
            name = os.path.basename(u.path)
            path = os.path.join("sounds", name)
            if not os.path.isfile(path):
                self.send(404, "text/plain", "no such sound")
                return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/battle":
            try:
                import battle
                self.send(200, "application/json", json.dumps(battle.self_play("latest")))
            except Exception as e:  # keep the dashboard alive if the brain file is mid-write
                self.send(503, "application/json", json.dumps({"error": str(e)}))
        elif u.path.startswith("/icon/") and u.path.endswith(".png") and "/" not in u.path[6:]:
            # state icons generated in ChatGPT from ChatGPT/icon-<name>.prompt.txt; 404 until the PNG exists
            path = os.path.join("ChatGPT", "icon-" + u.path[6:])
            if not os.path.isfile(path):
                self.send(404, "text/plain", "no icon yet")
                return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/robot_perfect.png":
            # the perfect player's avatar (ChatGPT/robot-perfect.prompt.txt); the grey robot stands in until it exists
            path = next((os.path.join("ChatGPT", w) for w in ("robot-perfect.png", "robot-brain-icon.png")
                         if os.path.isfile(os.path.join("ChatGPT", w))), None)
            if path is None:
                self.send(404, "text/plain", "no icon yet")
                return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            self.wfile.write(data)
        elif u.path in ("/robot.png", "/robot_x.png", "/robot_o.png"):
            # X = blue robot, O = red robot; fall back to the earlier icons until they exist
            want = ["robot-blue.png", "robot-brain-icon.png"] if u.path != "/robot_o.png"                 else ["robot-red.png", "robot-brain-icon-o.png", "robot-brain-icon.png"]
            path = next((os.path.join("ChatGPT", w) for w in want if os.path.isfile(os.path.join("ChatGPT", w))),
                        os.path.join("ChatGPT", "robot-brain-icon.png"))
            if not os.path.isfile(path):
                self.send(404, "text/plain", "no icon yet")
                return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/logs":
            logs = sorted(glob.glob("*.log"), key=os.path.getmtime, reverse=True)
            self.send(200, "application/json", json.dumps(logs))
        else:
            self.send(200, "text/html; charset=utf-8", HTML)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default="train_single.log")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    Handler.log_default = a.log
    print(f"dashboard on http://localhost:{a.port}  (log: {a.log})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
