from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gausscapture.errors import GaussCaptureError
from gausscapture.export.preview import preview_model_path

EXPORT_TYPES = ("raw", "web", "blender", "unity", "proxy_mesh")


def create_export(project_path: Path, export_type: str) -> dict[str, Any]:
    builders: dict[str, Callable[[Path, Path], None]] = {
        "raw": _raw_model,
        "web": _web_bundle,
        "blender": _blender_export,
        "unity": _unity_package,
        "proxy_mesh": _proxy_mesh,
    }
    if export_type not in builders:
        raise GaussCaptureError(
            f"Unsupported export type: {export_type}. Choose one of {', '.join(EXPORT_TYPES)}."
        )

    export_dir = project_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    work_dir = export_dir / export_type
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    builders[export_type](project_path, work_dir)

    zip_path = export_dir / f"{export_type}_export.zip"
    if zip_path.exists():
        zip_path.unlink()
    _zip_dir(work_dir, zip_path)

    return {
        "id": zip_path.stem,
        "type": export_type,
        "path": str(zip_path),
        "size_bytes": zip_path.stat().st_size,
    }


def list_exports(project_path: Path) -> list[dict[str, Any]]:
    export_dir = project_path / "export"
    if not export_dir.exists():
        return []
    return [
        {
            "id": file.stem,
            "name": file.name,
            "path": str(file),
            "size_bytes": file.stat().st_size,
        }
        for file in sorted(export_dir.glob("*.zip"))
    ]


def export_path(project_path: Path, export_id: str) -> Path:
    path = project_path / "export" / f"{export_id}.zip"
    if path.exists():
        return path
    matches = sorted((project_path / "export").glob(f"{export_id}*.zip"))
    if matches:
        return matches[0]
    raise FileNotFoundError(export_id)


def _raw_model(project_path: Path, out: Path) -> None:
    model = preview_model_path(project_path)
    shutil.copy2(model, out / model.name)
    shutil.copy2(project_path / "preview" / "preview_config.json", out / "metadata.json")


def _web_bundle(project_path: Path, out: Path) -> None:
    model = preview_model_path(project_path)
    shutil.copy2(model, out / model.name)
    shutil.copy2(project_path / "preview" / "preview_config.json", out / "metadata.json")
    (out / "index.html").write_text(_WEB_VIEWER.replace("__MODEL__", model.name), encoding="utf-8")
    (out / "README_WEB.md").write_text(
        "Serve this folder with any static server, for example `python -m http.server 8000`.\n\n"
        "The bundled viewer loads three.js from a CDN, so it needs an internet connection. "
        "It renders the model as a point cloud, which is a preview rather than a faithful "
        "splat render; for that, load the model into a Gaussian splat viewer such as Spark "
        "(https://github.com/sparkjsdev/spark) or SuperSplat.\n",
        encoding="utf-8",
    )


def _blender_export(project_path: Path, out: Path) -> None:
    model = preview_model_path(project_path)
    shutil.copy2(model, out / model.name)
    (out / "README_BLENDER.md").write_text(
        "Import `.ply` as a point cloud in Blender. Other containers need a conversion step; "
        "`splat-transform` (https://github.com/playcanvas/splat-transform, MIT) converts "
        "between PLY, SPZ, SOG and GLB.\n",
        encoding="utf-8",
    )


def _unity_package(project_path: Path, out: Path) -> None:
    model = preview_model_path(project_path)
    assets = out / "UnityAssets" / "Models"
    scripts = out / "UnityAssets" / "Scripts"
    assets.mkdir(parents=True)
    scripts.mkdir(parents=True)
    shutil.copy2(model, assets / model.name)
    (scripts / "GaussianSplatLoader.cs").write_text(
        "// Placeholder. GaussCapture prepares the model file; rendering it requires a\n"
        "// Unity Gaussian splat renderer package, which this export does not include.\n",
        encoding="utf-8",
    )
    (out / "README_UNITY.md").write_text(
        "Model files are prepared under `UnityAssets/Models`. A Unity Gaussian splat renderer "
        "integration is still required -- this export does not provide one.\n",
        encoding="utf-8",
    )


def _proxy_mesh(project_path: Path, out: Path) -> None:
    """Experimental Poisson reconstruction. Fails softly with a report.

    This is genuinely unreliable on splat point clouds -- Gaussian centres are
    not a surface sample -- so the failure is written to a report rather than
    raised, and the export still produces a zip explaining what happened.
    """
    report: dict[str, Any] = {"status": "error", "warnings": []}
    try:
        import open3d as o3d  # Optional dependency; see requirements-optional.txt

        model = preview_model_path(project_path)
        if model.suffix.lower() != ".ply":
            raise GaussCaptureError("Proxy mesh currently requires a .ply point cloud.")
        cloud = o3d.io.read_point_cloud(str(model))
        if len(cloud.points) == 0:
            raise GaussCaptureError("PLY contains no readable points.")
        cloud.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=7)
        o3d.io.write_triangle_mesh(str(out / "proxy_mesh.obj"), mesh)
        report = {
            "status": "success",
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "warnings": ["Poisson reconstruction of splat centres is approximate."],
        }
    except ImportError:
        report["warnings"].append(
            "Open3D is not installed. Install it with "
            "`python -m pip install -r requirements-optional.txt`."
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        report["warnings"].append(str(exc))
    (out / "proxy_mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _zip_dir(source: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(source))


_WEB_VIEWER = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>GaussCapture Web Bundle</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui}
#hint{position:fixed;top:12px;left:12px;background:#0008;padding:8px 10px;border-radius:6px;font-size:13px}</style></head>
<body><div id="hint">GaussCapture preview: __MODEL__ (point cloud approximation)</div><script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js';
const scene=new THREE.Scene();scene.background=new THREE.Color('#111318');
const camera=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,.01,1000);camera.position.set(0,0,3);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);
fetch('__MODEL__').then(r=>r.text()).then(t=>{const pts=parsePly(t);const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));const m=new THREE.PointsMaterial({size:.015,color:'#d7f5ff'});scene.add(new THREE.Points(g,m));});
function parsePly(t){const lines=t.split(/\\r?\\n/);let n=0,i=0;for(;i<lines.length;i++){if(lines[i].startsWith('element vertex'))n=+lines[i].split(' ')[2];if(lines[i]==='end_header'){i++;break}}const a=[];for(let j=0;j<n&&i+j<lines.length;j++){const p=lines[i+j].trim().split(/\\s+/).map(Number);a.push(p[0],p[1],p[2])}return a}
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera)}loop();
</script></body></html>"""
