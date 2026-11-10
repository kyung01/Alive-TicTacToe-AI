// Renders presentation/video/index.html (or one part of it) into a 1920x1080 video, frame by frame, through headless Chrome.
//
//   node tools/render_video.mjs [--fps 30] [--from 0] [--to 600] [--name video] [--port 9333] [--cpu] [--info]
//   node tools/render_parallel.mjs      <- the usual way: six parts at once, then the soundtrack and the final MP4
//
// The page runs in ?render mode: a virtual clock drives every animation, so each frame is exact however slow the
// capture is. Chrome draws on the GPU (NVIDIA, via ANGLE/D3D11; --cpu falls back to software) and the frames go
// straight into ffmpeg's NVENC H.264 encoder. The page logs every sound effect it would have played; they are
// written to export/render_events[_name].json for tools/mix_audio.py. --info prints the slide list and exits.
// Needs: Chrome, ffmpeg (C:\ffmpeg-7.1.1-full_build\bin), an NVIDIA GPU and the local server on :8777. Nothing to install.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'presentation', 'video', 'export');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, flag = k => process.argv.includes('--' + k);
const FPS = +arg('fps', 30), FROM = +arg('from', 0), TO = arg('to', null), NAME = arg('name', 'video'), PORT = +arg('port', 9333), GPU = !flag('cpu'), INFO = flag('info');
const URL_ = arg('url', 'http://localhost:8777/video/index.html'), TIMELINE = arg('timeline', null);   // --timeline export/render_info.json: lock the slide lengths
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe', FFMPEG = 'C:/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe';
mkdirSync(OUT, { recursive: true });
const log = (...a) => console.log(`[${NAME}]`, ...a);

const profile = join(tmpdir(), `ai-brain-render-profile-${PORT}`); rmSync(profile, { recursive: true, force: true });
const gpuFlags = GPU ? ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--force_high_performance_gpu'] : ['--disable-gpu'];
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, '--window-size=1920,1080', '--hide-scrollbars',
  '--mute-audio', '--force-device-scale-factor=1', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  ...gpuFlags, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target = null;
for (let k = 0; k < 75 && !target; k++) { await sleep(200); try { const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); target = list.find(t => t.type === 'page'); } catch (e) {} }
if (!target) throw new Error('Chrome did not start');

const ws = new WebSocket(target.webSocketDebuggerUrl); await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });
let nid = 0; const waiting = new Map(), listeners = [];
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && waiting.has(m.id)) { const { ok, no } = waiting.get(m.id); waiting.delete(m.id); m.error ? no(new Error(JSON.stringify(m.error))) : ok(m.result); } else if (m.method) listeners.forEach(f => f(m)); };
const cdp = (method, params = {}) => new Promise((ok, no) => { const id = ++nid; waiting.set(id, { ok, no }); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async expr => { const r = await cdp('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails)); return r.result.value; };
const quit = () => { try { ws.close(); } catch (e) {} chrome.kill(); };

await cdp('Page.enable'); await cdp('Runtime.enable');
await cdp('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
// the local server can drop a request when many renders start at once: reload until the page and its render code are really there
for (let attempt = 1; ; attempt++) {
  const loaded = new Promise(r => listeners.push(m => { if (m.method === 'Page.loadEventFired') r(); }));
  await cdp('Page.navigate', { url: `${URL_}?render&clean&nocap&v=${Date.now()}` }); await loaded;
  let ok = false; for (let k = 0; k < 40 && !ok; k++) { ok = await evaluate(`typeof __render === 'object'`); if (!ok) await sleep(250); }
  if (ok) break;
  if (attempt >= 5) throw new Error('the page never came up with its render code');
  log(`page incomplete, reloading (attempt ${attempt + 1})`); await sleep(1000 * attempt);
}
const gl = await evaluate(`(() => { const g = document.createElement('canvas').getContext('webgl'); const d = g && g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'no webgl'; })()`);
log('drawing on:', gl);
const lock = TIMELINE ? JSON.parse((await import('node:fs')).readFileSync(TIMELINE, 'utf8')).scenes.map(s => ({ dur: s.dur, adur: s.adur })) : null;
const info = await evaluate(`__render.ready(${JSON.stringify(lock)})`);
if (TIMELINE) { const ref = JSON.parse((await import('node:fs')).readFileSync(TIMELINE, 'utf8')); const off = info.scenes.find((s, i) => Math.abs(s.start - ref.scenes[i].start) > 1e-6); if (off || Math.abs(info.total - ref.total) > 1e-6) throw new Error('timeline lock failed at ' + (off && off.name)); log('timeline locked to', TIMELINE); }
if (INFO) { writeFileSync(join(OUT, 'render_info.json'), JSON.stringify(info, null, 1)); log('wrote export/render_info.json'); quit(); process.exit(0); }

const total = TO ? Math.min(+TO, info.total) : info.total;
const frameOf = s => Math.ceil(s * FPS - 1e-6);   // a part ends where the next one starts: no frame twice, none missing
const n0 = frameOf(FROM), n1 = frameOf(total);
if (FROM > 0) await evaluate(`__render.prime(${FROM})`);
log(`frames ${n0}..${n1 - 1} (${(n0 / FPS).toFixed(2)}-${(n1 / FPS).toFixed(2)} s) at ${FPS} fps`);

const enc = GPU ? ['-c:v', 'h264_nvenc', '-preset', 'p7', '-tune', 'hq', '-rc', 'vbr', '-cq', '17', '-b:v', '0', '-maxrate', '40M', '-bufsize', '80M', '-profile:v', 'high', '-bf', '3', '-g', String(FPS * 2)]
  : ['-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-threads', '4'];
const ff = spawn(FFMPEG, ['-v', 'error', '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-', ...enc,
  '-pix_fmt', 'yuv420p', '-r', String(FPS), '-movflags', '+faststart', join(OUT, `${NAME}_noaudio.mp4`)], { stdio: ['pipe', 'inherit', 'inherit'] });
const write = buf => new Promise(r => ff.stdin.write(buf) ? r() : ff.stdin.once('drain', r));

const t0 = Date.now();
for (let f = n0; f < n1; f++) {
  await evaluate(`__render.frame(${(f / FPS).toFixed(6)})`);
  const shot = await cdp('Page.captureScreenshot', { format: 'jpeg', quality: 94, captureBeyondViewport: false });
  await write(Buffer.from(shot.data, 'base64'));
  if ((f - n0) % 450 === 0 && f > n0) { const done = f - n0, el = (Date.now() - t0) / 1000; log(`frame ${f}/${n1}  ${(done / el).toFixed(1)} fps, eta ${((n1 - f) / (done / el) / 60).toFixed(1)} min`); }
}
const events = await evaluate('__render.events()');
writeFileSync(join(OUT, NAME === 'video' ? 'render_events.json' : `render_events_${NAME}.json`), JSON.stringify({ ...info, fps: FPS, from: n0 / FPS, to: n1 / FPS, events }, null, 1));
ff.stdin.end(); await new Promise(r => ff.on('close', r));
quit();
log(`done in ${((Date.now() - t0) / 60000).toFixed(1)} min -> export/${NAME}_noaudio.mp4 (${events.length} sounds)`);
process.exit(0);
