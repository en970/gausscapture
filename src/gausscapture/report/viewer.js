/*
 * Point-cloud viewer for GaussCapture reconstructions.
 *
 * Raw WebGL2, no libraries. A sparse cloud plus a few dozen camera frustums is
 * a few hundred lines of matrix maths, and vendoring a 600 KB engine to draw
 * coloured points would cost more than it saves -- especially since the whole
 * point of this page is that it opens from disk with nothing installed.
 */

const VERTEX_POINTS = `#version 300 es
in vec3 aPosition;
in vec3 aColor;
uniform mat4 uViewProjection;
uniform float uPointSize;
out vec3 vColor;
void main() {
  vColor = aColor;
  vec4 clip = uViewProjection * vec4(aPosition, 1.0);
  gl_Position = clip;
  // Scale with distance so near points read as substantial and far ones do not
  // dissolve, without the cost of real splatting.
  gl_PointSize = clamp(uPointSize / max(clip.w, 0.001), 1.0, 12.0);
}`;

const FRAGMENT_POINTS = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 outColor;
void main() {
  // Round points; square ones read as pixel noise at this density.
  vec2 offset = gl_PointCoord - vec2(0.5);
  if (dot(offset, offset) > 0.25) discard;
  outColor = vec4(vColor, 1.0);
}`;

const VERTEX_LINES = `#version 300 es
in vec3 aPosition;
in vec3 aColor;
uniform mat4 uViewProjection;
out vec3 vColor;
void main() {
  vColor = aColor;
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
}`;

const FRAGMENT_LINES = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }`;

// ---------------------------------------------------------------- matrix maths

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function normalise(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function lookAt(eye, target, up) {
  const forward = normalise([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  let side = cross(forward, up);
  if (Math.hypot(side[0], side[1], side[2]) < 1e-6) side = cross(forward, [0, 0, 1]);
  side = normalise(side);
  const trueUp = cross(side, forward);
  const out = new Float32Array(16);
  out[0] = side[0];  out[4] = side[1];  out[8]  = side[2];
  out[1] = trueUp[0]; out[5] = trueUp[1]; out[9] = trueUp[2];
  out[2] = -forward[0]; out[6] = -forward[1]; out[10] = -forward[2];
  out[12] = -(side[0] * eye[0] + side[1] * eye[1] + side[2] * eye[2]);
  out[13] = -(trueUp[0] * eye[0] + trueUp[1] * eye[1] + trueUp[2] * eye[2]);
  out[14] = forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2];
  out[15] = 1;
  return out;
}

// ---------------------------------------------------------------- gl helpers

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function program(gl, vertexSource, fragmentSource) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  return p;
}

// ---------------------------------------------------------------- viewer

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;

    this.pointProgram = program(gl, VERTEX_POINTS, FRAGMENT_POINTS);
    this.lineProgram = program(gl, VERTEX_LINES, FRAGMENT_LINES);

    this.pointCount = 0;
    this.lineCount = 0;
    this.pointSize = 260;
    this.showCameras = true;

    this.target = [0, 0, 0];
    this.radius = 5;
    this.theta = 0.6;
    this.phi = 1.15;

    this._bindControls();
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.043, 0.055, 0.09, 1);
  }

  _bindControls() {
    const canvas = this.canvas;
    let dragging = null;
    let lastX = 0;
    let lastY = 0;

    const start = (event) => {
      dragging = event.shiftKey || event.button === 2 ? "pan" : "orbit";
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    };
    const move = (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (dragging === "orbit") {
        this.theta -= dx * 0.006;
        // Clamped short of the poles, where the up vector degenerates and the
        // view flips disconcertingly.
        this.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.phi - dy * 0.006));
      } else {
        const scale = this.radius * 0.0016;
        const right = [Math.cos(this.theta), 0, -Math.sin(this.theta)];
        this.target[0] -= right[0] * dx * scale;
        this.target[2] -= right[2] * dx * scale;
        this.target[1] += dy * scale;
      }
      this.render();
    };
    const end = (event) => {
      dragging = null;
      try { canvas.releasePointerCapture(event.pointerId); } catch (_) { /* already gone */ }
    };

    canvas.addEventListener("pointerdown", start);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.radius *= Math.exp(event.deltaY * 0.0012);
      this.render();
    }, { passive: false });

    window.addEventListener("resize", () => this.render());
  }

  /** Loads the packed buffer: N float32 positions followed by N uint8 colours. */
  async load(scene, bufferUrl) {
    const gl = this.gl;
    const response = await fetch(bufferUrl);
    if (!response.ok) throw new Error(`Could not load ${bufferUrl}`);
    const raw = await response.arrayBuffer();

    const count = scene.points;
    const positions = new Float32Array(raw, 0, count * 3);
    const colors = new Uint8Array(raw, count * 12, count * 3);
    this.pointCount = count;

    this.pointVao = gl.createVertexArray();
    gl.bindVertexArray(this.pointVao);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const positionLocation = gl.getAttribLocation(this.pointProgram, "aPosition");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    const colorLocation = gl.getAttribLocation(this.pointProgram, "aColor");
    gl.enableVertexAttribArray(colorLocation);
    gl.vertexAttribPointer(colorLocation, 3, gl.UNSIGNED_BYTE, true, 0, 0);

    this._buildCameras(scene);

    this.target = scene.centre.slice();
    this.radius = scene.radius * 2.4;
    this.render();
  }

  /**
   * Builds the camera frustums and the path between them.
   *
   * Seeing where the operator actually walked is often more diagnostic than
   * the cloud: a capture that failed for lack of parallax looks like a cluster
   * of frustums in one spot, which no summary statistic conveys as directly.
   */
  _buildCameras(scene) {
    const gl = this.gl;
    const cameras = scene.cameras || [];
    const vertices = [];
    const colors = [];
    const size = scene.radius * 0.05;

    const push = (a, b, colour) => {
      vertices.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      colors.push(...colour, ...colour);
    };

    const frustum = [0.85, 0.55, 0.35];  // warm, so it reads against the scene
    const path = [0.35, 0.75, 0.55];

    cameras.forEach((camera, index) => {
      const c = camera.position;
      const R = camera.rotation;  // row-major 3x3, camera-to-world
      // COLMAP cameras look down +z; corners of the image plane at unit depth.
      const corner = (x, y) => [
        c[0] + (R[0] * x + R[1] * y + R[2]) * size,
        c[1] + (R[3] * x + R[4] * y + R[5]) * size,
        c[2] + (R[6] * x + R[7] * y + R[8]) * size,
      ];
      const tl = corner(-0.6, -0.4);
      const tr = corner(0.6, -0.4);
      const br = corner(0.6, 0.4);
      const bl = corner(-0.6, 0.4);
      push(c, tl, frustum); push(c, tr, frustum);
      push(c, br, frustum); push(c, bl, frustum);
      push(tl, tr, frustum); push(tr, br, frustum);
      push(br, bl, frustum); push(bl, tl, frustum);

      if (index > 0) push(cameras[index - 1].position, c, path);
    });

    this.lineCount = vertices.length / 3;
    if (this.lineCount === 0) return;

    this.lineVao = gl.createVertexArray();
    gl.bindVertexArray(this.lineVao);

    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    const positionLocation = gl.getAttribLocation(this.lineProgram, "aPosition");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    const colorLocation = gl.getAttribLocation(this.lineProgram, "aColor");
    gl.enableVertexAttribArray(colorLocation);
    gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
  }

  render() {
    const gl = this.gl;
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.floor(canvas.clientWidth * ratio);
    const height = Math.floor(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.pointCount) return;

    const eye = [
      this.target[0] + this.radius * Math.sin(this.phi) * Math.sin(this.theta),
      this.target[1] + this.radius * Math.cos(this.phi),
      this.target[2] + this.radius * Math.sin(this.phi) * Math.cos(this.theta),
    ];
    const view = lookAt(eye, this.target, [0, 1, 0]);
    const projection = perspective(
      (55 * Math.PI) / 180,
      canvas.width / Math.max(1, canvas.height),
      this.radius * 0.001,
      this.radius * 40,
    );
    const viewProjection = multiply(projection, view);

    gl.useProgram(this.pointProgram);
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.pointProgram, "uViewProjection"), false, viewProjection);
    gl.uniform1f(gl.getUniformLocation(this.pointProgram, "uPointSize"), this.pointSize);
    gl.bindVertexArray(this.pointVao);
    gl.drawArrays(gl.POINTS, 0, this.pointCount);

    if (this.showCameras && this.lineCount) {
      gl.useProgram(this.lineProgram);
      gl.uniformMatrix4fv(
        gl.getUniformLocation(this.lineProgram, "uViewProjection"), false, viewProjection);
      gl.bindVertexArray(this.lineVao);
      gl.drawArrays(gl.LINES, 0, this.lineCount);
    }
  }
}
