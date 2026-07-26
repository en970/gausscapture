from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def create_export(project_path: Path, export_type: str) -> dict[str, Any]:
    export_dir = project_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "raw": _raw_model,
        "web": _web_bundle,
        "unity": _unity_package,
        "blender": _blender_export,
        "proxy_mesh": _proxy_mesh,
    }
    if export_type not in builders:
        raise ValueError(f"Unsupported export type: {export_type}")
    work_dir = export_dir / export_type
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    builders[export_type](project_path, work_dir)
    zip_path = export_dir / f"{export_type}_export.zip"
    if zip_path.exists():
        zip_path.unlink()
    _zip_dir(work_dir, zip_path)
    return {"id": zip_path.stem, "type": export_type, "path": str(zip_path), "size_bytes": zip_path.stat().st_size}


def list_exports(project_path: Path) -> list[dict[str, Any]]:
    exports = []
    for file in (project_path / "export").glob("*.zip"):
        exports.append({"id": file.stem, "name": file.name, "path": str(file), "size_bytes": file.stat().st_size})
    return exports


def export_path(project_path: Path, export_id: str) -> Path:
    path = project_path / "export" / f"{export_id}.zip"
    if not path.exists():
        matches = list((project_path / "export").glob(f"{export_id}*.zip"))
        if matches:
            return matches[0]
        raise FileNotFoundError(export_id)
    return path


def _model(project_path: Path) -> Path:
    config_path = project_path / "preview" / "preview_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Preview model is missing. Import a trained result first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = project_path / "preview" / config["model_file"]
    if not model.exists():
        raise FileNotFoundError("Configured preview model file is missing.")
    return model


def _raw_model(project_path: Path, out: Path) -> None:
    model = _model(project_path)
    shutil.copy2(model, out / model.name)
    config = project_path / "preview" / "preview_config.json"
    shutil.copy2(config, out / "metadata.json")


def _web_bundle(project_path: Path, out: Path) -> None:
    model = _model(project_path)
    shutil.copy2(model, out / model.name)
    (out / "metadata.json").write_text((project_path / "preview" / "preview_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "index.html").write_text(WEB_VIEWER_HTML.replace("__MODEL__", model.name), encoding="utf-8")
    (out / "README_WEB.md").write_text("Serve this folder with any static server, for example `python -m http.server 8000`.\n", encoding="utf-8")


def _unity_package(project_path: Path, out: Path) -> None:
    model = _model(project_path)
    assets = out / "UnityAssets" / "Models"
    scripts = out / "UnityAssets" / "Scripts"
    assets.mkdir(parents=True)
    scripts.mkdir(parents=True)
    shutil.copy2(model, assets / model.name)
    (scripts / "GaussianSplatLoader.cs").write_text("// Placeholder loader. Integrate with a Unity Gaussian Splat renderer package.\n", encoding="utf-8")
    (out / "README_UNITY.md").write_text("Model files are prepared. A Unity Gaussian Splat renderer integration is required.\n", encoding="utf-8")


def _blender_export(project_path: Path, out: Path) -> None:
    model = _model(project_path)
    shutil.copy2(model, out / model.name)
    (out / "README_BLENDER.md").write_text("Import `.ply` as a point cloud in Blender. `.splat` requires a compatible add-on or conversion tool.\n", encoding="utf-8")


def _proxy_mesh(project_path: Path, out: Path) -> None:
    report: dict[str, Any] = {"status": "error", "warnings": []}
    try:
        import open3d as o3d

        model = _model(project_path)
        if model.suffix.lower() != ".ply":
            raise RuntimeError("Proxy mesh currently requires a .ply point cloud.")
        pcd = o3d.io.read_point_cloud(str(model))
        if len(pcd.points) == 0:
            raise RuntimeError("PLY has no readable points.")
        pcd.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
        obj = out / "proxy_mesh.obj"
        o3d.io.write_triangle_mesh(str(obj), mesh)
        report = {"status": "success", "vertices": len(mesh.vertices), "triangles": len(mesh.triangles), "warnings": []}
    except Exception as exc:
        report["warnings"].append(str(exc))
    (out / "proxy_mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _zip_dir(source: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in source.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(source))


WEB_VIEWER_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>GaussCapture Web Bundle</title><style>body{margin:0;background:#111;color:#eee;font-family:system-ui}#hint{position:fixed;top:12px;left:12px;background:#0008;padding:8px 10px;border-radius:6px}</style></head>
<body><div id="hint">GaussCapture Web Bundle: __MODEL__</div><script type="module">
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

