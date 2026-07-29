"""Generating the local report page."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from gausscapture.eval.harness import load_results
from gausscapture.recon.dataset import _find_sparse_model
from gausscapture.report.scene import export_scene

_VIEWER_JS = Path(__file__).parent / "viewer.js"


def build_report(run_dir: Path, out_dir: Path) -> Path:
    """Turn a benchmark run into a folder of static files.

    Everything is written relative and self-contained so the result opens from
    disk. The reconstructions worth looking at are on the machine that produced
    them and far too large to upload, so a local page is the only place they
    can realistically be seen.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(run_dir / "results.jsonl")
    config = _read_json(run_dir / "run_config.json")

    scenes: list[dict[str, Any]] = []
    for row in rows:
        project_id = row.get("project_id")
        if not project_id:
            continue
        project_path = run_dir / "projects" / project_id
        model_dir = _find_sparse_model(project_path)
        if model_dir is None:
            continue
        name = _short_name(row.get("name", project_id))
        scene = export_scene(model_dir, out_dir, name=name)
        if not scene.get("points"):
            continue
        scene["capture"] = name
        scene["registered"] = row.get("images_registered")
        scene["total"] = row.get("frames_used")
        scene["ratio"] = row.get("registered_ratio")
        scene["status"] = row.get("pose_status")
        scene["seconds"] = row.get("seconds_pose")
        scenes.append(scene)

    shutil.copyfile(_VIEWER_JS, out_dir / "viewer.js")
    (out_dir / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(
        _page(scenes, rows, config, run_dir.name), encoding="utf-8"
    )
    return out_dir / "index.html"


def _short_name(name: str) -> str:
    return name.split("_2026")[0] if "_2026" in name else name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _signal_rows(rows: list[dict[str, Any]]) -> str:
    names = sorted({k[len("signal_") :] for row in rows for k in row if k.startswith("signal_")})
    interesting = [
        n for n in ("vol_p10", "vol_median", "motion_mean", "blurry_frame_ratio", "score")
        if n in names
    ]
    header = "".join(f"<th>{n}</th>" for n in interesting)
    body = ""
    for row in rows:
        cells = ""
        for n in interesting:
            value = row.get(f"signal_{n}")
            cells += f"<td>{value:.1f}</td>" if isinstance(value, (int, float)) else "<td>—</td>"
        body += f"<tr><td class=name>{_short_name(row.get('name', ''))}</td>{cells}</tr>"
    return f"<thead><tr><th>capture</th>{header}</tr></thead><tbody>{body}</tbody>"


def _result_rows(rows: list[dict[str, Any]]) -> str:
    body = ""
    for row in rows:
        ratio = row.get("registered_ratio") or 0
        status = row.get("pose_status", "—")
        klass = "ok" if status == "good" else "warn" if status == "warning" else "bad"
        body += (
            f"<tr><td class=name>{_short_name(row.get('name', ''))}</td>"
            f"<td>{row.get('frames_used', 0)}</td>"
            f"<td>{row.get('images_registered', 0)}</td>"
            f"<td>{ratio:.0%}</td>"
            f"<td>{row.get('sparse_points', 0):,}</td>"
            f"<td>{row.get('seconds_pose', 0):.0f}s</td>"
            f"<td><span class='tag {klass}'>{status}</span></td></tr>"
        )
    return body


def _page(
    scenes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    run_name: str,
) -> str:
    options = "".join(
        f'<option value="{i}">{s["capture"]} · {s["points"]:,} points · '
        f'{s.get("registered", 0)} cameras</option>'
        for i, s in enumerate(scenes)
    )
    empty = "" if scenes else "<p class=note>No reconstruction in this run produced points.</p>"
    preset = config.get("frame_preset", "?")
    seeded = "on" if config.get("seed_intrinsics") else "off"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GaussCapture — {run_name}</title>
<style>
  :root {{
    --bg:#0b0e17; --panel:#131826; --line:#252c3e; --ink:#e8ecf5;
    --muted:#8b93a7; --accent:#d99a6c; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;line-height:1.6}}
  header{{padding:22px 28px;border-bottom:1px solid var(--line);display:flex;
          justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:10px}}
  h1{{margin:0;font-size:19px;letter-spacing:-.01em}}
  .meta{{font-family:var(--mono);font-size:12px;color:var(--muted)}}
  main{{max-width:1500px;margin:0 auto;padding:22px 28px 60px}}
  .stage{{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;
          background:#080b12}}
  canvas{{display:block;width:100%;height:min(66vh,660px);cursor:grab;touch-action:none}}
  canvas:active{{cursor:grabbing}}
  .overlay{{position:absolute;top:14px;left:14px;display:flex;gap:8px;flex-wrap:wrap;
            align-items:center}}
  select,button{{background:var(--panel);color:var(--ink);border:1px solid var(--line);
                 border-radius:8px;padding:8px 12px;font-size:13px;font-family:inherit;
                 cursor:pointer;min-height:36px}}
  button:hover,select:hover{{border-color:var(--accent)}}
  .hint{{position:absolute;bottom:12px;left:14px;font-size:11.5px;color:var(--muted);
         font-family:var(--mono);background:#0b0e17cc;padding:6px 10px;border-radius:7px}}
  .caution{{margin:18px 0 0;padding:14px 18px;border-left:3px solid var(--accent);
            background:#d99a6c14;border-radius:0 10px 10px 0;font-size:14.5px}}
  .caution b{{color:var(--accent)}}
  h2{{font-size:12px;text-transform:uppercase;letter-spacing:.11em;color:var(--accent);
      margin:34px 0 12px}}
  .scroll{{overflow-x:auto}}
  table{{border-collapse:collapse;width:100%;font-size:14px;min-width:520px}}
  th,td{{text-align:left;padding:9px 14px;border-bottom:1px solid var(--line)}}
  th{{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.07em;
      color:var(--muted);font-weight:500}}
  td{{color:var(--muted);font-variant-numeric:tabular-nums}}
  td.name{{color:var(--ink);font-family:var(--mono);font-size:13px}}
  .tag{{font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:5px}}
  .tag.ok{{background:#4ade8022;color:var(--ok)}}
  .tag.warn{{background:#fbbf2422;color:var(--warn)}}
  .tag.bad{{background:#f8717122;color:var(--bad)}}
  .note{{color:var(--muted);font-size:14px}}
</style>
</head>
<body>
<header>
  <h1>GaussCapture — {run_name}</h1>
  <span class="meta">preset {preset} · intrinsic seeding {seeded} · {len(rows)} captures</span>
</header>
<main>
  <div class="stage">
    <canvas id="view"></canvas>
    <div class="overlay">
      <select id="scene">{options}</select>
      <button id="cameras">Hide cameras</button>
      <button id="reset">Reset view</button>
    </div>
    <div class="hint">drag to orbit · scroll to zoom · shift-drag to pan</div>
  </div>
  {empty}

  <p class="caution">
    <b>This is the structure-from-motion point cloud, not a Gaussian splat.</b>
    These are the sparse 3D points COLMAP triangulated, with the camera path it
    solved — the input a splat is trained <em>from</em>. Training the splat
    itself needs a CUDA GPU, which this machine does not have.
  </p>

  <h2>Reconstruction</h2>
  <div class="scroll"><table>
    <thead><tr><th>capture</th><th>frames</th><th>registered</th><th>ratio</th>
    <th>points</th><th>colmap</th><th>status</th></tr></thead>
    <tbody>{_result_rows(rows)}</tbody>
  </table></div>

  <h2>Capture telemetry</h2>
  <div class="scroll"><table>{_signal_rows(rows)}</table></div>
  <p class="note">Raw statistics order these captures correctly; the thresholded
  ratios derived from them do not. See <code>docs/EXPERIMENTS.md</code>.</p>
</main>

<script type="module">
import {{ Viewer }} from "./viewer.js";

const scenes = await (await fetch("./scenes.json")).json();
const canvas = document.getElementById("view");
const picker = document.getElementById("scene");

if (!scenes.length) {{
  canvas.style.display = "none";
}} else {{
  let viewer;
  try {{
    viewer = new Viewer(canvas);
  }} catch (error) {{
    canvas.outerHTML = `<p class="note" style="padding:40px">${{error.message}}</p>`;
  }}

  const show = async (index) => {{
    const scene = scenes[index];
    await viewer.load(scene, "./" + scene.buffer);
  }};

  picker.addEventListener("change", () => show(picker.value));
  document.getElementById("reset").addEventListener("click", () => show(picker.value));
  document.getElementById("cameras").addEventListener("click", (event) => {{
    viewer.showCameras = !viewer.showCameras;
    event.target.textContent = viewer.showCameras ? "Hide cameras" : "Show cameras";
    viewer.render();
  }});

  await show(0);
}}
</script>
</body>
</html>
"""
