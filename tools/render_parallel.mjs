// Renders the whole video in parallel parts and builds the final YouTube file.
//
//   node tools/render_parallel.mjs [--parts 8] [--fps 60]
//
// 1. asks the page for its slide list (render_video.mjs --info)
// 2. cuts the video into parts at slide starts where nothing carries over from the slide before
// 3. renders the parts at the same time (each its own headless Chrome on the GPU + NVENC)
// 4. joins them without re-encoding, merges their sound logs, mixes the soundtrack (tools/mix_audio.py)
// 5. normalises the soundtrack to -14 LUFS (YouTube's level, two-pass) and muxes
//    -> presentation/video/export/AI_Brain_Part1_1080p60.mp4
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..'), OUT = join(ROOT, 'presentation', 'video', 'export');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const PARTS = +arg('parts', 8), FPS = +arg('fps', 60), FFMPEG = 'C:/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe';
const run = (cmd, args, opts = {}) => new Promise((ok, no) => { const p = spawn(cmd, args, { stdio: opts.capture ? ['ignore', 'pipe', 'pipe'] : 'inherit', cwd: ROOT });
  let err = ''; if (opts.capture) { p.stdout.on('data', d => { err += d; }); p.stderr.on('data', d => { err += d; }); }
  p.on('close', c => c === 0 ? ok(err) : no(new Error(`${cmd} ${args.join(' ')} exited ${c}\n${err.slice(-2000)}`))); });
const t0 = Date.now(), mins = () => ((Date.now() - t0) / 60000).toFixed(1);

await run('node', ['tools/render_video.mjs', '--info', '--port', '9350', '--name', 'info']);
const info = JSON.parse(readFileSync(join(OUT, 'render_info.json'), 'utf8'));
const cuts = [0];
for (let k = 1; k < PARTS; k++) { const want = info.total * k / PARTS;
  const c = info.scenes.filter(s => s.seamless && s.i > 0 && s.start > cuts.at(-1) + 20).sort((a, b) => Math.abs(a.start - want) - Math.abs(b.start - want))[0];
  if (c) cuts.push(c.start); }
cuts.push(info.total);
// parts only start at slide changes: a part that starts in the middle of a slide drew the swaying foreground slightly
// differently from the long part before it (a one-frame jump), so the long end screen stays one part and simply starts first
console.log('parts:', cuts.slice(0, -1).map((c, k) => `${k + 1}: ${c.toFixed(2)}-${cuts[k + 1].toFixed(2)} s`).join(' | '));

// parts start two seconds apart (easier on the local server); a part that fails is tried again, twice at most
const longest = cuts.slice(0, -1).map((c, k) => [cuts[k + 1] - c, k]).sort((a, b) => b[0] - a[0])[0][1];
const renderPart = async (c, k) => { await new Promise(r => setTimeout(r, (k === longest ? 0 : 1 + k) * 2000));
  for (let attempt = 1; ; attempt++) { try { return await run('node', ['tools/render_video.mjs', '--from', String(c), '--to', String(cuts[k + 1]), '--fps', String(FPS), '--name', `part${k + 1}`, '--port', String(9351 + k), '--timeline', join(OUT, 'render_info.json')]); }
    catch (e) { if (attempt >= 3) throw e; console.log(`part ${k + 1} failed, retrying (${e.message.split(String.fromCharCode(10))[0]})`); } } };
await Promise.all(cuts.slice(0, -1).map((c, k) => renderPart(c, k)));
console.log(`all parts rendered (${mins()} min); joining`);

const list = join(OUT, 'parts.txt');
writeFileSync(list, cuts.slice(0, -1).map((c, k) => `file 'part${k + 1}_noaudio.mp4'`).join('\n') + '\n');
await run(FFMPEG, ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', join(OUT, 'video_noaudio.mp4')]);
const evs = cuts.slice(0, -1).map((c, k) => JSON.parse(readFileSync(join(OUT, `render_events_part${k + 1}.json`), 'utf8')));
writeFileSync(join(OUT, 'render_events.json'), JSON.stringify({ ...evs[0], from: 0, to: evs.at(-1).to, events: evs.flatMap(e => e.events).sort((a, b) => a[0] - b[0]) }, null, 1));

console.log('mixing the soundtrack');
await run('python', ['tools/mix_audio.py']);
const wav = join(OUT, 'soundtrack.wav'), LN = 'loudnorm=I=-14:TP=-1.5:LRA=11';
const m = JSON.parse((await run(FFMPEG, ['-hide_banner', '-i', wav, '-af', LN + ':print_format=json', '-f', 'null', '-'], { capture: true })).match(/\{[\s\S]*?\}/)[0]);
const final = join(OUT, `AI_Brain_Part1_1080p${FPS}.mp4`);
await run(FFMPEG, ['-v', 'error', '-y', '-i', join(OUT, 'video_noaudio.mp4'), '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
  '-af', `${LN}:measured_I=${m.input_i}:measured_TP=${m.input_tp}:measured_LRA=${m.input_lra}:measured_thresh=${m.input_thresh}:offset=${m.target_offset}:linear=true`,
  '-ar', '48000', '-c:a', 'aac', '-b:a', '320k', '-shortest', '-movflags', '+faststart', final]);
cuts.slice(0, -1).forEach((c, k) => { rmSync(join(OUT, `part${k + 1}_noaudio.mp4`), { force: true }); rmSync(join(OUT, `render_events_part${k + 1}.json`), { force: true }); });
rmSync(list, { force: true });
console.log(`done in ${mins()} min -> ${final} (soundtrack was ${m.input_i} LUFS, now -14)`);
