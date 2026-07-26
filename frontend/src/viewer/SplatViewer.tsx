import { RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

type Props = {
  modelUrl: string;
  modelType: string;
};

export default function SplatViewer({ modelUrl, modelType }: Props) {
  const host = useRef<HTMLDivElement | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const [bg, setBg] = useState('#111318');
  const [message, setMessage] = useState('Loading model...');

  useEffect(() => {
    if (!host.current) return;
    const element = host.current;
    element.innerHTML = '';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(bg);
    const camera = new THREE.PerspectiveCamera(60, element.clientWidth / element.clientHeight, 0.01, 10000);
    camera.position.set(0, 0, 3);
    cameraRef.current = camera;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(element.clientWidth, element.clientHeight);
    element.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;
    scene.add(new THREE.AmbientLight(0xffffff, 1));

    let alive = true;
    loadModel(modelUrl, modelType)
      .then(({ positions, colors }) => {
        if (!alive) return;
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        if (colors.length) {
          geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }
        geometry.computeBoundingSphere();
        const material = new THREE.PointsMaterial({ size: 0.015, vertexColors: colors.length > 0, color: '#d7f5ff' });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        const sphere = geometry.boundingSphere;
        if (sphere) {
          controls.target.copy(sphere.center);
          camera.position.copy(sphere.center).add(new THREE.Vector3(0, 0, Math.max(1, sphere.radius * 3)));
          camera.near = Math.max(0.001, sphere.radius / 1000);
          camera.far = Math.max(1000, sphere.radius * 100);
          camera.updateProjectionMatrix();
        }
        setMessage(`${positions.length / 3} points loaded`);
      })
      .catch((err) => setMessage(err.message));

    const resize = () => {
      if (!element.clientWidth || !element.clientHeight) return;
      camera.aspect = element.clientWidth / element.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(element.clientWidth, element.clientHeight);
    };
    window.addEventListener('resize', resize);
    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      scene.background = new THREE.Color(bg);
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    return () => {
      alive = false;
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      renderer.dispose();
    };
  }, [modelUrl, modelType, bg]);

  function reset() {
    controlsRef.current?.reset();
  }

  return (
    <div className="viewer-shell">
      <div className="viewer-toolbar">
        <button onClick={reset}><RotateCcw size={17} /> Reset</button>
        <label>Background <input type="color" value={bg} onChange={(e) => setBg(e.target.value)} /></label>
        <span>{message}</span>
      </div>
      <div ref={host} className="three-host" />
    </div>
  );
}

async function loadModel(url: string, type: string): Promise<{ positions: number[]; colors: number[] }> {
  if (type === 'splat' || type === 'ksplat') {
    const buffer = await (await fetch(url)).arrayBuffer();
    return parseSplat(buffer);
  }
  const text = await (await fetch(url)).text();
  return parsePly(text);
}

function parsePly(text: string) {
  const lines = text.split(/\r?\n/);
  let count = 0;
  let headerEnd = 0;
  const props: string[] = [];
  let readingVertex = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('element vertex')) {
      count = Number(line.split(/\s+/)[2]);
      readingVertex = true;
    } else if (line.startsWith('element ') && !line.startsWith('element vertex')) {
      readingVertex = false;
    } else if (readingVertex && line.startsWith('property')) {
      props.push(line.split(/\s+/).pop() || '');
    } else if (line === 'end_header') {
      headerEnd = i + 1;
      break;
    }
  }
  const x = props.indexOf('x');
  const y = props.indexOf('y');
  const z = props.indexOf('z');
  const r = props.indexOf('red');
  const g = props.indexOf('green');
  const b = props.indexOf('blue');
  const positions: number[] = [];
  const colors: number[] = [];
  for (let i = 0; i < count && headerEnd + i < lines.length; i++) {
    const values = lines[headerEnd + i].trim().split(/\s+/).map(Number);
    if ([x, y, z].some((idx) => idx < 0 || Number.isNaN(values[idx]))) continue;
    positions.push(values[x], values[y], values[z]);
    if (r >= 0 && g >= 0 && b >= 0) {
      colors.push((values[r] || 255) / 255, (values[g] || 255) / 255, (values[b] || 255) / 255);
    }
  }
  return { positions, colors };
}

function parseSplat(buffer: ArrayBuffer) {
  const view = new DataView(buffer);
  const stride = 32;
  const count = Math.floor(buffer.byteLength / stride);
  const positions: number[] = [];
  const colors: number[] = [];
  for (let i = 0; i < count; i++) {
    const base = i * stride;
    positions.push(view.getFloat32(base, true), view.getFloat32(base + 4, true), view.getFloat32(base + 8, true));
    colors.push(view.getUint8(base + 24) / 255, view.getUint8(base + 25) / 255, view.getUint8(base + 26) / 255);
  }
  return { positions, colors };
}

