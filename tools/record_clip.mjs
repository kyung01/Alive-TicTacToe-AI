// Record the animated README clip of the 3D brain (assets/brain3d-clip.gif).
//
//   python -m http.server 8000          (in the repo folder, in another terminal)
//   node tools/record_clip.mjs [--url http://localhost:8000/brain3d/?clip] [--out assets/brain3d-clip.gif]
//                              [--seed N] [--chrome "path/to/chrome"]
//
// Needs Node 22+, Google Chrome and ffmpeg. The page runs in headless Chrome with ?clip: no panels, and every
// frame is advanced by this script (fixed time step), so the clip is smooth however slow the capture is. It
// first searches for a game the brain wins against the random player (each seed replays exactly), then
// records that game from the first tick to the end of the win banner.
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg('url', 'http://localhost:8000/brain3d/?clip');
const OUT = arg('out', 'assets/brain3d-clip.gif');
const CHROME = arg('chrome', process.env.CHROME || [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome'].find(existsSync));
const FPS = 12, TPS = 10, W = 1120, H = 630, PORT = 9555;

const profile = mkdtempSync(join(tmpdir(), 'brainclip-'));
const frames = mkdtempSync(join(tmpdir(), 'brainframes-'));
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${W},${H}`, '--hide-scrollbars', '--mute-audio', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', URL_],
  { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let target;
for (let i = 0; i < 100 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === 'page'); } catch {}
}
if (!target) throw new Error('Chrome did not start');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
let id = 0; const wait = new Map();
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result?.result?.value;
};

for (let i = 0; i < 150 && !(await evaluate('!!window.__ready')); i++) await sleep(200);
await evaluate('document.fonts.ready.then(() => true)');

// a game the brain wins, long enough to watch, short enough for a small GIF
let seed = arg('seed') ? +arg('seed') : null, game = null;
for (let s = 1; seed === null && s < 2000; s++) {
  const g = await evaluate(`__brain.trySeed(${s}, 'random')`);
  if (g.result.kind === 'win' && g.result.winner === g.brain && g.moves >= 5 && g.ticks >= 28 && g.ticks <= 60) { seed = s; game = g; }
}
game = game || await evaluate(`__brain.trySeed(${seed}, 'random')`);
console.log(`seed ${seed}: ${JSON.stringify(game)}`);
await evaluate(`__brain.setSpeed(${TPS}); __brain.start(${seed}, 'random'); true`);

const total = Math.ceil((game.ticks / TPS + 2.5) * FPS) + 4;
for (let f = 0; f < total; f++) {
  await evaluate(`__brain.frame(${1000 / FPS}); true`);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(frames, `f${String(f).padStart(4, '0')}.png`), Buffer.from(shot.result.data, 'base64'));
  if (f % 20 === 0) process.stdout.write(`frame ${f}/${total}\r`);
}
ws.close(); chrome.kill();

const ff = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', join(frames, 'f%04d.png'),
  '-vf', 'scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
  '-loop', '0', OUT], { stdio: 'inherit' });
if (ff.status !== 0) throw new Error('ffmpeg failed');
console.log(`\nwrote ${OUT} (${total} frames)`);
rmSync(frames, { recursive: true, force: true });
try { rmSync(profile, { recursive: true, force: true }); } catch {}
