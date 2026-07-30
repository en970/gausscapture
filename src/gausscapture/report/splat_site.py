"""A standalone page for looking at trained splats."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from gausscapture.export.splat_binary import write_splat_binary

_VIEWER_JS = Path(__file__).parent / "splat_viewer.js"


def build_splat_site(
    plys: list[Path],
    out_dir: Path,
    title: str = "GaussCapture",
    min_opacity: float = 0.02,
    max_gaussians: int | None = None,
    colmap_model: Path | None = None,
) -> Path:
    """Convert splats and write a self-contained viewer around them.

    ``colmap_model`` is the reconstruction the splats were trained from. Its
    cameras determine which way is up, and without it the scene keeps
    structure-from-motion's arbitrary frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[dict[str, Any]] = []
    for ply in plys:
        ply = Path(ply)
        if not ply.exists():
            continue
        packed = write_splat_binary(
            ply, out_dir / f"{ply.stem}.splat",
            min_opacity=min_opacity, max_gaussians=max_gaussians,
            colmap_model=colmap_model,
        )
        packed["label"] = _label(ply.stem)
        packed["source"] = ply.name
        scenes.append(packed)

    if not scenes:
        raise FileNotFoundError("No usable splats were given")

    shutil.copyfile(_VIEWER_JS, out_dir / "splat_viewer.js")
    (out_dir / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    index = out_dir / "index.html"
    index.write_text(_page(scenes, title), encoding="utf-8")
    return index


def _label(stem: str) -> str:
    """Turn `A_good_30000` into something a person would say."""
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return f"{' '.join(parts[:-1]).replace('_', ' ')} · {int(parts[-1]):,} steps"
    return stem.replace("_", " ")


def _page(scenes: list[dict[str, Any]], title: str) -> str:
    segments = "".join(
        f'<button class="segment{" on" if i == 0 else ""}" data-index="{i}">'
        f'{s["label"].split(" · ")[-1] if " · " in s["label"] else s["label"]}</button>'
        for i, s in enumerate(scenes)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#000000">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◍</text></svg>">
<style>
  :root {{
    --glass: rgba(28, 28, 30, 0.62);
    --stroke: rgba(255, 255, 255, 0.12);
    --ink: #f5f5f7;
    --dim: rgba(235, 235, 245, 0.6);
    --accent: #0a84ff;
    --radius: 20px;
    --safe-top: env(safe-area-inset-top, 0px);
    --safe-bottom: env(safe-area-inset-bottom, 0px);
  }}
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  html, body {{ height: 100%; margin: 0; overflow: hidden; background: #000; }}
  body {{
    font: 400 15px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text",
          "Segoe UI", Inter, system-ui, sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
  }}

  canvas {{
    position: fixed; inset: 0; width: 100%; height: 100%;
    display: block; touch-action: none; cursor: grab;
  }}
  canvas:active {{ cursor: grabbing; }}

  /* Frosted panels, the way iOS layers chrome over content. */
  .panel {{
    background: var(--glass);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    backdrop-filter: blur(24px) saturate(180%);
    border: 0.5px solid var(--stroke);
    border-radius: var(--radius);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }}

  header {{
    position: fixed; z-index: 5;
    top: calc(var(--safe-top) + 14px); left: 14px; right: 14px;
    display: flex; align-items: center; gap: 14px;
    padding: 12px 16px;
    pointer-events: none;
  }}
  header .panel-inner {{ pointer-events: auto; }}
  h1 {{
    margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -0.02em;
    font-family: -apple-system, "SF Pro Display", sans-serif;
  }}
  .sub {{ margin: 1px 0 0; font-size: 12.5px; color: var(--dim); font-variant-numeric: tabular-nums; }}
  .spacer {{ flex: 1; }}

  .stat {{
    text-align: right; font-variant-numeric: tabular-nums;
    font-size: 12.5px; color: var(--dim); line-height: 1.35;
  }}
  .stat b {{ display: block; color: var(--ink); font-size: 16px; font-weight: 600; }}

  /* Segmented control, the iOS pattern for switching between few options. */
  .dock {{
    position: fixed; z-index: 5;
    bottom: calc(var(--safe-bottom) + 18px); left: 50%;
    transform: translateX(-50%);
    display: flex; gap: 4px; padding: 4px;
    max-width: calc(100vw - 28px); overflow-x: auto;
    scrollbar-width: none;
  }}
  .dock::-webkit-scrollbar {{ display: none; }}
  .segment {{
    appearance: none; border: 0; cursor: pointer; white-space: nowrap;
    min-height: 40px; padding: 0 18px; border-radius: 16px;
    background: transparent; color: var(--dim);
    font: 500 14px/1 -apple-system, sans-serif;
    transition: background 220ms cubic-bezier(0.32, 0.72, 0, 1), color 180ms;
  }}
  .segment:hover {{ color: var(--ink); }}
  .segment.on {{ background: rgba(255, 255, 255, 0.16); color: var(--ink); }}

  .hint {{
    position: fixed; z-index: 4; left: 50%; transform: translateX(-50%);
    bottom: calc(var(--safe-bottom) + 74px);
    font-size: 12px; color: rgba(235, 235, 245, 0.45);
    pointer-events: none; text-align: center;
    transition: opacity 500ms;
  }}

  /* Loading, centred and quiet. */
  .loader {{
    position: fixed; inset: 0; z-index: 10;
    display: grid; place-content: center; justify-items: center; gap: 18px;
    background: #000; transition: opacity 450ms;
  }}
  .loader.gone {{ opacity: 0; pointer-events: none; }}
  .ring {{
    width: 34px; height: 34px; border-radius: 50%;
    border: 2.5px solid rgba(255, 255, 255, 0.14);
    border-top-color: var(--ink);
    animation: spin 800ms linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .loader p {{ margin: 0; font-size: 13.5px; color: var(--dim); font-variant-numeric: tabular-nums; }}
  .bar {{ width: 190px; height: 3px; border-radius: 2px; background: rgba(255,255,255,0.14); overflow: hidden; }}
  .bar i {{ display: block; height: 100%; width: 0; background: var(--ink); transition: width 140ms linear; }}

  .error {{
    position: fixed; inset: 0; z-index: 20; display: none;
    place-content: center; padding: 40px; text-align: center;
    background: #000; color: var(--dim); font-size: 14px;
  }}

  @media (max-width: 560px) {{
    header {{ padding: 10px 14px; gap: 10px; }}
    h1 {{ font-size: 15px; }}
    .stat b {{ font-size: 14px; }}
  }}
</style>
</head>
<body>

<canvas id="stage"></canvas>

<header>
  <div class="panel panel-inner" style="display:flex;align-items:center;gap:14px;padding:11px 16px;flex:1">
    <div>
      <h1>{title}</h1>
      <p class="sub" id="caption">—</p>
    </div>
    <div class="spacer"></div>
    <div class="stat"><b id="count">—</b>gaussians</div>
  </div>
</header>

<div class="hint" id="hint">drag to orbit · scroll to zoom · shift-drag to pan</div>

<div class="dock panel" id="dock">{segments}</div>

<div class="loader" id="loader">
  <div class="ring"></div>
  <div class="bar"><i id="bar"></i></div>
  <p id="status">Loading…</p>
</div>

<div class="error" id="error"></div>

<script type="module">
import {{ SplatViewer }} from './splat_viewer.js';

const scenes = await (await fetch('./scenes.json')).json();
const loader = document.getElementById('loader');
const bar = document.getElementById('bar');
const status = document.getElementById('status');
const caption = document.getElementById('caption');
const countEl = document.getElementById('count');
const hint = document.getElementById('hint');

const fmt = (n) => n.toLocaleString('en-US');

let viewer;
try {{
  viewer = new SplatViewer(document.getElementById('stage'), {{
    onProgress: (fraction, bytes) => {{
      bar.style.width = `${{Math.round(fraction * 100)}}%`;
      status.textContent = fraction
        ? `${{Math.round(fraction * 100)}}% · ${{(bytes / 1e6).toFixed(1)}} MB`
        : `${{(bytes / 1e6).toFixed(1)}} MB`;
    }},
  }});
}} catch (err) {{
  const box = document.getElementById('error');
  box.style.display = 'grid';
  box.textContent = err.message;
  loader.classList.add('gone');
}}

let current = -1;

async function show(index) {{
  if (!viewer || index === current) return;
  current = index;
  const scene = scenes[index];

  loader.classList.remove('gone');
  bar.style.width = '0%';
  status.textContent = 'Loading…';

  document.querySelectorAll('.segment').forEach((el, i) =>
    el.classList.toggle('on', i === index));

  const count = await viewer.load(`./${{scene.file}}`, scene);

  caption.textContent = scene.label;
  countEl.textContent = fmt(count);
  loader.classList.add('gone');

  // The hint has done its job once someone has actually moved the camera.
  setTimeout(() => {{ hint.style.opacity = '0'; }}, 6000);
}}

document.getElementById('dock').addEventListener('click', (event) => {{
  const button = event.target.closest('.segment');
  if (button) show(Number(button.dataset.index));
}});

if (viewer) await show(0);
</script>
</body>
</html>
"""
