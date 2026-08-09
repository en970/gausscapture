"""The offline page a 4D scene is delivered as.

The deliverable of the whole fixed-camera pipeline is one link: a page that
plays a few seconds of the subject as gaussians, on a camera rail spanning the
angles the capture actually observed. This module builds that page the same way
:mod:`gausscapture.report.splat_site` builds the static one -- a directory of
files with no CDN, no external font, no build step and no server requirement
beyond what the browser's own module rules impose.

Two things here are not decoration.

**The page states what it is showing.** The clip length, the gaussian count, the
measured cone, whether the fixed camera pose was verified or merely asserted,
and whether the export had to drop gaussians to fit a budget. Each of those is
a fact that changes how much the picture is worth, and each of them is
invisible in the picture itself.

**The cone is labelled, not fenced.** The camera is hard clamped, but there is
no warning, no red and no modal at the boundary. A user who reaches the edge
has not made a mistake -- they have reached the end of what the recording
contains -- so the interface says which angles were observed and lets the
control simply stop. The one moving part is a sub-line that cross-fades from
the cone's half-angles to "Edge of trained view" as the camera approaches it.

``--single-file`` inlines the viewer and base64-encodes the scene, because a
module page that fetches its own data next to it is blocked by ``file://``
origin rules in every browser. A directory is smaller and streams; a single
file opens by double-clicking. Both are built from the same template.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from typing import Any

from gausscapture.export.scene4d import Scene4D, read_scene4d_header, write_scene4d

_VIEWER_JS = Path(__file__).parent / "viewer_4d.js"

DEFAULT_CAVEAT = (
    "This was recorded from one fixed viewpoint. Only the angles inside the trained "
    "cone were ever observed; outside it there is no evidence, so the camera stops "
    "there rather than inventing a view."
)


def build_scene4d_site(
    scene: Scene4D | Path,
    out_dir: Path,
    title: str = "GaussCapture 4D",
    single_file: bool = False,
    max_gaussians: int | None = None,
    cone_source: str = "measured",
    fixed_pose: dict[str, Any] | None = None,
    caveat: str = DEFAULT_CAVEAT,
    rate: float = 0.5,
    cone_azimuth_deg: float | None = None,
    cone_elevation_deg: float | None = None,
) -> Path:
    """Write ``index.html`` and its data into ``out_dir``; return the page.

    ``scene`` is either a :class:`~gausscapture.export.scene4d.Scene4D` to
    encode now or an existing ``.g4d`` to wrap. ``cone_source`` and
    ``fixed_pose`` carry the provenance the page prints; their defaults claim
    the strongest case, so a caller with a weaker one has to say so explicitly
    rather than getting it by omission.

    ``cone_azimuth_deg`` and ``cone_elevation_deg`` may only ever *narrow* the
    cone stored in the file. Publishing a scene more cautiously than it was
    measured is a judgement its author is entitled to make; publishing it more
    widely is a claim about angles nobody recorded, so a request to widen is
    ignored and the fact that it was is recorded in the sidecar rather than
    raised -- the export is still valid, it just did not do what was asked.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "scene.g4d"

    if isinstance(scene, Scene4D):
        stats = write_scene4d(scene, target, max_gaussians=max_gaussians)
        meta = dict(scene.meta)
        loop = scene.loop
        verified = scene.fixed_pose_verified
        cone = (scene.cone_azimuth_deg, scene.cone_elevation_deg)
        created_target = True
    else:
        source = Path(scene)
        created_target = source.resolve() != target.resolve()
        if created_target:
            shutil.copyfile(source, target)
        header = read_scene4d_header(target)
        meta = {}
        loop = header.loop
        verified = header.fixed_pose_verified
        cone = (header.cone_azimuth_deg, header.cone_elevation_deg)
        stats = {
            "file": target.name,
            "bytes": target.stat().st_size,
            "gaussian_count": header.gaussian_count,
            "dynamic_count": header.dynamic_count,
            "node_count": header.node_count,
            "frame_count": header.frame_count,
            "skin_k": header.skin_k,
            "fps": float(header.fps),
            "clip_seconds": float(header.clip_seconds),
            "truncated": False,
            "trained_gaussian_count": header.gaussian_count,
            "chunk_bytes": {tag: length for tag, (_, length) in header.chunks.items()},
        }

    if fixed_pose is None:
        fixed_pose = {"source": "verified" if verified else "asserted"}

    published, cone_note = _narrow_cone(cone, cone_azimuth_deg, cone_elevation_deg)

    sidecar: dict[str, Any] = {
        "title": title,
        "file": None if single_file else target.name,
        "bytes": int(stats["bytes"]),
        "gaussian_count": int(stats["gaussian_count"]),
        "dynamic_count": int(stats["dynamic_count"]),
        "node_count": int(stats["node_count"]),
        "frame_count": int(stats["frame_count"]),
        "skin_k": int(stats["skin_k"]),
        "fps": float(stats["fps"]),
        "clip_seconds": float(stats["clip_seconds"]),
        "chunk_bytes": {str(k): int(v) for k, v in stats.get("chunk_bytes", {}).items()},
        "cone": {
            "azimuth_deg": float(published[0]),
            "elevation_deg": float(published[1]),
            # Kept alongside the published pair so a page that was narrowed can
            # still say what the capture actually reached.
            "measured_azimuth_deg": float(cone[0]),
            "measured_elevation_deg": float(cone[1]),
            "narrowed": bool(published != cone),
            "source": cone_source,
            "notes": cone_note,
        },
        "fixed_pose": fixed_pose,
        "loop": loop,
        "rate": float(rate),
        "truncated": bool(stats.get("truncated", False)),
        "trained_gaussian_count": int(stats.get("trained_gaussian_count",
                                                stats["gaussian_count"])),
        "caveat": caveat,
    }
    if meta.get("label"):
        sidecar["label"] = meta["label"]

    viewer_source = _VIEWER_JS.read_text(encoding="utf-8")

    if single_file:
        payload = base64.b64encode(target.read_bytes()).decode("ascii")
        # An inline module cannot re-export, so the viewer's `export` keywords
        # are dropped and everything simply shares the module's scope.
        prologue = (viewer_source
                    .replace("export class ", "class ")
                    .replace("export function ", "function "))
        loader = (
            "scene = " + json.dumps(sidecar) + ";\n"
            "const bytes = atob(SCENE_BASE64);\n"
            "const buffer = new Uint8Array(bytes.length);\n"
            "for (let i = 0; i < bytes.length; i++) buffer[i] = bytes.charCodeAt(i);\n"
            "onProgress(1, buffer.length);\n"
            "viewer.loadBuffer(buffer.buffer);\n"
        )
        inline_data = f'const SCENE_BASE64 = "{payload}";\n'
        if created_target:
            target.unlink()
    else:
        shutil.copyfile(_VIEWER_JS, out_dir / "viewer_4d.js")
        (out_dir / "scene.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        prologue = "import { Scene4DViewer } from './viewer_4d.js';"
        loader = (
            "scene = await (await fetch('./scene.json')).json();\n"
            "await viewer.load('./scene.g4d');\n"
        )
        inline_data = ""

    page = (_TEMPLATE
            .replace("__TITLE__", _escape(title))
            .replace("__PROLOGUE__", prologue)
            .replace("__INLINE_DATA__", inline_data)
            .replace("__LOADER__", loader)
            .replace("__CAVEAT__", _escape(caveat)))

    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def _narrow_cone(
    measured: tuple[float, float],
    azimuth: float | None,
    elevation: float | None,
) -> tuple[tuple[float, float], list[str]]:
    """Apply a publisher's narrower cone, refusing any request to widen."""
    notes: list[str] = []
    published = [float(measured[0]), float(measured[1])]
    for axis, (name, request) in enumerate(
        (("azimuth", azimuth), ("elevation", elevation))
    ):
        if request is None:
            continue
        value = float(request)
        if value < 0.0:
            raise ValueError(f"a {name} half-angle cannot be negative: {value}")
        if value > published[axis]:
            notes.append(
                f"Requested {name} half-angle ±{value:.1f}° is wider than the ±"
                f"{published[axis]:.1f}° the capture measured, so it was ignored. "
                "The cone can only ever be narrowed."
            )
            continue
        published[axis] = value
    return (published[0], published[1]), notes


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0b0d">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◍</text></svg>">
<style>
  :root {
    --glass: rgba(28, 28, 30, 0.66);
    --stroke: rgba(255, 255, 255, 0.12);
    --ink: #f5f5f7;
    --dim: rgba(235, 235, 245, 0.6);
    --faint: rgba(235, 235, 245, 0.38);
    --accent: #0a84ff;
    --warn: #ffd60a;
    --radius: 18px;
    --safe-top: env(safe-area-inset-top, 0px);
    --safe-bottom: env(safe-area-inset-bottom, 0px);
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { height: 100%; margin: 0; overflow: hidden; background: #0b0b0d; }
  body {
    font: 400 15px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text",
          "Segoe UI", Inter, system-ui, sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }
  canvas {
    position: fixed; inset: 0; width: 100%; height: 100%;
    display: block; touch-action: none; cursor: grab;
  }
  canvas:active { cursor: grabbing; }

  .panel {
    background: var(--glass);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    border: 0.5px solid var(--stroke);
    border-radius: var(--radius);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
  }

  header {
    position: fixed; z-index: 5;
    top: calc(var(--safe-top) + 14px); left: 14px; right: 14px;
    display: flex; align-items: flex-start; gap: 14px;
    padding: 11px 15px;
  }
  h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: -0.02em; }
  .sub { margin: 2px 0 0; font-size: 12.5px; color: var(--dim); font-variant-numeric: tabular-nums; }
  .spacer { flex: 1; }
  .badge {
    font-size: 11px; letter-spacing: 0.03em; text-transform: uppercase;
    color: var(--faint); border: 0.5px solid var(--stroke);
    border-radius: 999px; padding: 3px 9px; white-space: nowrap;
  }
  .badge.asserted { color: var(--warn); border-color: rgba(255, 214, 10, 0.35); }
  button.icon {
    appearance: none; border: 0; background: rgba(255,255,255,0.09); color: var(--ink);
    width: 30px; height: 30px; border-radius: 999px; cursor: pointer; font-size: 13px;
    display: grid; place-content: center; flex: none;
  }
  button.icon:hover { background: rgba(255,255,255,0.17); }

  /* The cone: a rail with the capture pose marked, and a label that is a
     statement about the recording rather than a warning about the control. */
  .cone {
    position: fixed; z-index: 5; left: 50%; transform: translateX(-50%);
    bottom: calc(var(--safe-bottom) + 96px);
    width: min(420px, calc(100vw - 28px));
    display: grid; gap: 7px; justify-items: center;
    pointer-events: none; text-align: center;
  }
  .rail {
    position: relative; width: 100%; height: 3px; border-radius: 2px;
    background: rgba(255, 255, 255, 0.14);
  }
  .rail .tick {
    position: absolute; left: 50%; top: -4px; width: 1px; height: 11px;
    background: rgba(255, 255, 255, 0.45); transform: translateX(-50%);
  }
  .rail .tick::after {
    content: "capture"; position: absolute; top: 13px; left: 50%;
    transform: translateX(-50%);
    font-size: 9.5px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--faint);
  }
  .rail .thumb {
    position: absolute; top: 50%; left: 50%; width: 9px; height: 9px;
    border-radius: 50%; background: var(--ink);
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.18);
  }
  .angles {
    font-size: 12px; color: var(--dim); font-variant-numeric: tabular-nums;
    letter-spacing: 0.01em;
  }
  .cone-label { position: relative; height: 15px; width: 100%; }
  .cone-label span {
    position: absolute; inset: 0; font-size: 11.5px; color: var(--faint);
    transition: opacity 260ms ease;
  }
  .cone-label .edge { color: var(--dim); opacity: 0; }

  /* Transport. */
  .transport {
    position: fixed; z-index: 6;
    bottom: calc(var(--safe-bottom) + 18px); left: 50%; transform: translateX(-50%);
    width: min(520px, calc(100vw - 28px));
    display: flex; align-items: center; gap: 12px; padding: 9px 13px;
  }
  .play {
    appearance: none; border: 0; cursor: pointer; flex: none;
    width: 38px; height: 38px; border-radius: 50%;
    background: rgba(255, 255, 255, 0.14); color: var(--ink);
    display: grid; place-content: center;
  }
  .play:hover { background: rgba(255, 255, 255, 0.22); }
  .play svg { width: 15px; height: 15px; fill: currentColor; }
  input[type=range] {
    appearance: none; -webkit-appearance: none; flex: 1; height: 22px;
    background: transparent; cursor: pointer; margin: 0;
  }
  input[type=range]::-webkit-slider-runnable-track {
    height: 3px; border-radius: 2px; background: rgba(255, 255, 255, 0.2);
  }
  input[type=range]::-moz-range-track {
    height: 3px; border-radius: 2px; background: rgba(255, 255, 255, 0.2);
  }
  input[type=range]::-webkit-slider-thumb {
    appearance: none; -webkit-appearance: none; margin-top: -5px;
    width: 13px; height: 13px; border-radius: 50%; background: var(--ink);
  }
  input[type=range]::-moz-range-thumb {
    width: 13px; height: 13px; border: 0; border-radius: 50%; background: var(--ink);
  }
  .clock {
    font-size: 12px; color: var(--dim); font-variant-numeric: tabular-nums;
    min-width: 76px; text-align: right; white-space: nowrap;
  }
  .chip {
    appearance: none; border: 0.5px solid var(--stroke); cursor: pointer;
    background: transparent; color: var(--dim);
    border-radius: 999px; padding: 5px 10px; font-size: 11.5px; white-space: nowrap;
  }
  .chip:hover { color: var(--ink); background: rgba(255,255,255,0.08); }

  /* The caveat sheet. */
  .sheet {
    position: fixed; z-index: 8; inset: auto 14px calc(var(--safe-bottom) + 18px) auto;
    right: 14px; width: min(360px, calc(100vw - 28px));
    padding: 16px 18px; display: none;
  }
  .sheet.on { display: block; }
  .sheet h2 { margin: 0 0 8px; font-size: 13px; font-weight: 600; }
  .sheet p { margin: 0 0 10px; font-size: 12.5px; color: var(--dim); }
  .sheet dl {
    margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 14px;
    font-size: 12px; font-variant-numeric: tabular-nums;
  }
  .sheet dt { color: var(--faint); }
  .sheet dd { margin: 0; color: var(--ink); text-align: right; }

  .loader {
    position: fixed; inset: 0; z-index: 10;
    display: grid; place-content: center; justify-items: center; gap: 16px;
    background: #0b0b0d; transition: opacity 450ms;
  }
  .loader.gone { opacity: 0; pointer-events: none; }
  .ring {
    width: 32px; height: 32px; border-radius: 50%;
    border: 2.5px solid rgba(255, 255, 255, 0.14); border-top-color: var(--ink);
    animation: spin 800ms linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loader p { margin: 0; font-size: 13px; color: var(--dim); font-variant-numeric: tabular-nums; }

  .error {
    position: fixed; inset: 0; z-index: 20; display: none;
    place-content: center; padding: 40px; text-align: center;
    background: #0b0b0d; color: var(--dim); font-size: 14px; line-height: 1.6;
  }

  @media (max-width: 560px) {
    header { padding: 10px 13px; }
    h1 { font-size: 15px; }
    .clock { min-width: 68px; }
    .sheet { left: 14px; right: 14px; width: auto; bottom: calc(var(--safe-bottom) + 76px); }
  }
  @media (prefers-reduced-motion: reduce) {
    .cone-label span { transition: none; }
  }
</style>
</head>
<body>

<canvas id="stage"></canvas>

<header class="panel">
  <div>
    <h1>__TITLE__</h1>
    <p class="sub" id="sub">—</p>
  </div>
  <div class="spacer"></div>
  <span class="badge" id="pose">—</span>
  <button class="icon" id="infoButton" aria-label="About this scene">i</button>
</header>

<div class="cone">
  <div class="angles" id="angles">—</div>
  <div class="rail"><div class="tick"></div><div class="thumb" id="thumb"></div></div>
  <div class="cone-label">
    <span class="trained" id="trained">—</span>
    <span class="edge" id="edge">Edge of trained view</span>
  </div>
</div>

<div class="transport panel">
  <button class="play" id="play" aria-label="Play or pause">
    <svg id="playIcon" viewBox="0 0 16 16"><path d="M3 1.5l11 6.5-11 6.5z"/></svg>
  </button>
  <input type="range" id="scrub" min="0" max="1000" value="0" step="1" aria-label="Scrub">
  <span class="clock" id="clock">0.00 / 0.00 s</span>
  <button class="chip" id="loop">loop</button>
  <button class="chip" id="rate">0.5×</button>
</div>

<div class="sheet panel" id="sheet">
  <h2>What you are looking at</h2>
  <p id="caveat">__CAVEAT__</p>
  <dl id="facts"></dl>
</div>

<div class="loader" id="loader">
  <div class="ring"></div>
  <p id="status">Loading…</p>
</div>

<div class="error" id="error"></div>

<script type="module">
__PROLOGUE__

__INLINE_DATA__

const loader = document.getElementById('loader');
const status = document.getElementById('status');
const errorBox = document.getElementById('error');

const el = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const deg = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}°`;

function onProgress(fraction, bytes) {
  status.textContent = fraction
    ? `${Math.round(fraction * 100)}% · ${(bytes / 1e6).toFixed(1)} MB`
    : `${(bytes / 1e6).toFixed(1)} MB`;
}

let viewer;
let drawing = 0;

try {
  viewer = new Scene4DViewer(el('stage'), {
    onProgress,
    onCamera: (camera) => {
      el('angles').textContent =
        `azimuth ${deg(camera.azimuthDeg)} · elevation ${deg(camera.elevationDeg)}`;
      const across = camera.coneAzimuthDeg > 0
        ? camera.azimuthDeg / camera.coneAzimuthDeg : 0;
      el('thumb').style.left = `${50 + across * 50}%`;
      // The label cross-fades rather than switching, so approaching the edge
      // reads as arriving somewhere rather than as tripping something.
      const near = Math.max(0, Math.min(1, (camera.rho - 0.95) / 0.05));
      el('edge').style.opacity = String(near);
      el('trained').style.opacity = String(1 - near);
    },
    onTime: (time) => {
      el('clock').textContent =
        `${time.seconds.toFixed(2)} / ${time.clipSeconds.toFixed(2)} s`;
      if (!scrubbing) {
        el('scrub').value = String(Math.round(1000 * time.tau / Math.max(1, time.frameCount - 1)));
      }
      el('playIcon').innerHTML = time.playing
        ? '<path d="M3 1.5h4v13H3zM9 1.5h4v13H9z"/>'
        : '<path d="M3 1.5l11 6.5-11 6.5z"/>';
    },
    onBudget: (budget) => {
      drawing = budget.drawing;
      updateSub();
    },
  });
} catch (err) {
  errorBox.style.display = 'grid';
  errorBox.textContent = err.message;
  loader.classList.add('gone');
}

let scene = null;      // assigned by the loader below, fetched or inlined
let scrubbing = false;

function updateSub() {
  if (!scene) return;
  const parts = [
    `${scene.clip_seconds.toFixed(1)} s`,
    `${fmt(drawing || scene.gaussian_count)} gaussians`,
  ];
  if (drawing && drawing < scene.gaussian_count) {
    parts[1] = `${fmt(drawing)} of ${fmt(scene.gaussian_count)} gaussians`;
  }
  parts.push(`${fmt(scene.dynamic_count)} moving`);
  el('sub').textContent = parts.join(' · ');
}

if (viewer) {
  try {
    __LOADER__

    el('trained').textContent =
      `Trained view · ±${scene.cone.azimuth_deg.toFixed(0)}° × ±${scene.cone.elevation_deg.toFixed(0)}°`;

    const poseBadge = el('pose');
    poseBadge.textContent = scene.fixed_pose.source === 'verified'
      ? 'pose verified' : `pose ${scene.fixed_pose.source}`;
    poseBadge.classList.toggle('asserted', scene.fixed_pose.source !== 'verified');

    viewer.setRate(scene.rate || 0.5);
    viewer.setLoop(scene.loop || 'ping-pong');
    el('rate').textContent = `${(scene.rate || 0.5)}×`;
    el('loop').textContent = scene.loop || 'ping-pong';
    el('scrub').max = '1000';
    updateSub();

    // Facts the picture cannot show, printed rather than implied.
    const facts = [
      ['gaussians', fmt(scene.gaussian_count)],
      ['moving', fmt(scene.dynamic_count)],
      ['scaffold nodes', fmt(scene.node_count)],
      ['frames', `${fmt(scene.frame_count)} at ${scene.fps.toFixed(0)} fps`],
      ['trained cone', `±${scene.cone.azimuth_deg.toFixed(0)}° × ±${scene.cone.elevation_deg.toFixed(0)}°`],
      ['cone source', scene.cone.source],
      ['fixed pose', scene.fixed_pose.source],
      ['file', `${(scene.bytes / 1e6).toFixed(2)} MB`],
    ];
    if (scene.truncated) {
      facts.push(['truncated from', fmt(scene.trained_gaussian_count)]);
    }
    el('facts').innerHTML = facts
      .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

    if (scene.fixed_pose.source === 'asserted') {
      el('caveat').textContent += ' The fixed camera pose could not be verified against '
        + 'the moving frames, so it was inherited from the still frames and assumed. '
        + 'Treat the geometry as provisional.';
    }
    if (scene.truncated) {
      el('caveat').textContent += ' This export dropped gaussians to fit a size budget, '
        + 'so it is not the full trained scene.';
    }

    viewer.play();
    loader.classList.add('gone');
  } catch (err) {
    errorBox.style.display = 'grid';
    errorBox.textContent = err.message;
    loader.classList.add('gone');
  }
}

// ── transport ──────────────────────────────────────────────────────────────
el('play').addEventListener('click', () => viewer.togglePlay());

el('scrub').addEventListener('pointerdown', () => { scrubbing = true; viewer.pause(); });
el('scrub').addEventListener('input', (event) => {
  const last = Math.max(1, scene.frame_count - 1);
  viewer.seek(Number(event.target.value) / 1000 * last);
});
el('scrub').addEventListener('pointerup', () => { scrubbing = false; });
el('scrub').addEventListener('pointercancel', () => { scrubbing = false; });

const LOOPS = ['ping-pong', 'forward', 'once'];
el('loop').addEventListener('click', () => {
  const next = LOOPS[(LOOPS.indexOf(el('loop').textContent) + 1) % LOOPS.length];
  el('loop').textContent = next;
  viewer.setLoop(next);
});

const RATES = [0.25, 0.5, 1];
el('rate').addEventListener('click', () => {
  const current = parseFloat(el('rate').textContent);
  const next = RATES[(RATES.indexOf(current) + 1) % RATES.length];
  el('rate').textContent = `${next}×`;
  viewer.setRate(next);
});

el('infoButton').addEventListener('click', () => el('sheet').classList.toggle('on'));

window.addEventListener('keydown', (event) => {
  if (event.key === ' ') { event.preventDefault(); viewer.togglePlay(); }
  if (event.key === 'ArrowLeft') { viewer.pause(); viewer.seek(viewer.tau - 1); }
  if (event.key === 'ArrowRight') { viewer.pause(); viewer.seek(viewer.tau + 1); }
});
</script>
</body>
</html>
"""


__all__ = ["DEFAULT_CAVEAT", "build_scene4d_site"]
