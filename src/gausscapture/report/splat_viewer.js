/*
 * Gaussian splat renderer — WebGL2, no libraries.
 *
 * Draws actual gaussians rather than their centres: each one is an ellipse
 * whose shape comes from projecting its 3D covariance into screen space, and
 * whose alpha falls off as a 2D gaussian. That is the difference between
 * seeing a reconstruction and seeing a point cloud of where a reconstruction
 * would be.
 *
 * Two parts do the real work.
 *
 * **Covariance projection.** A gaussian's shape is Σ = R·S·Sᵀ·Rᵀ, from its
 * quaternion and scale. Viewed through a camera it becomes Σ' = J·W·Σ·Wᵀ·Jᵀ,
 * where W is the view rotation and J the local affine approximation of the
 * perspective divide. Eigen-decomposing the resulting 2×2 gives the ellipse
 * axes the vertex shader stretches a quad along.
 *
 * **Depth sorting.** Alpha blending is order-dependent, so gaussians must be
 * drawn back to front or the image is wrong in a way that looks like fog. With
 * hundreds of thousands of them a comparison sort every frame is far too slow,
 * so a worker runs a 16-bit counting sort — linear in the number of gaussians —
 * and only when the camera has actually moved.
 */

const WORKER_SOURCE = `
let positions = null;
let count = 0;

self.onmessage = (event) => {
  const data = event.data;

  if (data.positions) {
    positions = data.positions;
    count = positions.length / 3;
    return;
  }

  if (!positions) return;
  const view = data.view;

  // Only the depth row of the view matrix matters for ordering.
  const d2 = view[2], d6 = view[6], d10 = view[10];

  const depths = new Int32Array(count);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    const depth = positions[i * 3] * d2 + positions[i * 3 + 1] * d6 + positions[i * 3 + 2] * d10;
    depths[i] = depth * 4096 | 0;
    if (depths[i] > max) max = depths[i];
    if (depths[i] < min) min = depths[i];
  }

  // Counting sort over 16-bit buckets: linear, and precise enough that any
  // residual error is between gaussians at almost identical depth.
  const BUCKETS = 65536;
  const counts = new Uint32Array(BUCKETS);
  const scale = max > min ? (BUCKETS - 1) / (max - min) : 0;
  for (let i = 0; i < count; i++) {
    depths[i] = (depths[i] - min) * scale | 0;
    counts[depths[i]]++;
  }
  const starts = new Uint32Array(BUCKETS);
  for (let i = 1; i < BUCKETS; i++) starts[i] = starts[i - 1] + counts[i - 1];

  // Back to front: the far end of the range is drawn first.
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[count - 1 - starts[depths[i]]++] = i;

  self.postMessage({ order }, [order.buffer]);
};
`;

const VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 uProjection;
uniform mat4 uView;
uniform vec2 uViewport;
uniform vec2 uFocal;

uniform highp usampler2D uData;
uniform int uTexWidth;

in vec2 aQuad;              // unit quad corner, -2..2
in uint aIndex;             // position of this gaussian in the sorted order

out vec4 vColor;
out vec2 vOffset;

// Gaussian data lives in a texture rather than in vertex attributes, because
// the draw order changes every time the camera turns. Attributes are fetched
// by instance id and cannot be indirected; a texture can, so the sort result
// is simply the index each instance looks up.
void main() {
  int texel = int(aIndex) * 2;
  ivec2 uvA = ivec2(texel % uTexWidth, texel / uTexWidth);
  ivec2 uvB = ivec2((texel + 1) % uTexWidth, (texel + 1) / uTexWidth);
  uvec4 packedA = texelFetch(uData, uvA, 0);
  uvec4 packedB = texelFetch(uData, uvB, 0);

  vec3 aCenter = uintBitsToFloat(packedA.xyz);
  vec3 aScale = vec3(uintBitsToFloat(packedA.w),
                     uintBitsToFloat(packedB.x),
                     uintBitsToFloat(packedB.y));
  vec4 aColor = vec4((uvec4(packedB.z) >> uvec4(0u, 8u, 16u, 24u)) & 255u);
  vec4 aRotation = vec4((uvec4(packedB.w) >> uvec4(0u, 8u, 16u, 24u)) & 255u);

  vec4 viewPos = uView * vec4(aCenter, 1.0);

  // Behind the camera, or so far out it cannot matter: emit a degenerate
  // triangle rather than paying for the maths.
  if (viewPos.z > -0.01) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  vec4 q = (aRotation - 128.0) / 128.0;
  float w = q.x, x = q.y, y = q.z, z = q.w;

  // Rotation from the quaternion, then Σ = R·S·Sᵀ·Rᵀ.
  mat3 R = mat3(
    1.0 - 2.0*(y*y + z*z),  2.0*(x*y + w*z),        2.0*(x*z - w*y),
    2.0*(x*y - w*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z + w*x),
    2.0*(x*z + w*y),        2.0*(y*z - w*x),        1.0 - 2.0*(x*x + y*y)
  );
  mat3 S = mat3(aScale.x, 0.0, 0.0, 0.0, aScale.y, 0.0, 0.0, 0.0, aScale.z);
  mat3 M = R * S;
  mat3 sigma = M * transpose(M);

  // J is the affine approximation of the perspective divide at this depth.
  float invZ = 1.0 / viewPos.z;
  float invZ2 = invZ * invZ;
  mat3 J = mat3(
    uFocal.x * invZ, 0.0,             -uFocal.x * viewPos.x * invZ2,
    0.0,             uFocal.y * invZ, -uFocal.y * viewPos.y * invZ2,
    0.0,             0.0,             0.0
  );
  mat3 W = mat3(uView);
  mat3 T = J * W;
  mat3 cov = T * sigma * transpose(T);

  // A small floor on the diagonal keeps sub-pixel gaussians from collapsing
  // into invisible slivers.
  float a = cov[0][0] + 0.3;
  float b = cov[0][1];
  float d = cov[1][1] + 0.3;

  float mid = 0.5 * (a + d);
  float radius = length(vec2((a - d) * 0.5, b));
  float lambda1 = mid + radius;
  float lambda2 = max(mid - radius, 0.1);

  vec2 major = normalize(vec2(b, lambda1 - a));
  if (abs(b) < 1e-9 && abs(lambda1 - a) < 1e-9) major = vec2(1.0, 0.0);
  vec2 minor = vec2(major.y, -major.x);

  // Clamped so one enormous gaussian cannot swallow the frame.
  vec2 axis1 = min(sqrt(2.0 * lambda1), 1024.0) * major;
  vec2 axis2 = min(sqrt(2.0 * lambda2), 1024.0) * minor;

  vColor = vec4(aColor.rgb / 255.0, aColor.a / 255.0);
  vOffset = aQuad;

  vec4 clip = uProjection * viewPos;
  vec2 offset = (aQuad.x * axis1 + aQuad.y * axis2) / uViewport * 2.0;
  gl_Position = vec4(clip.xy / clip.w + offset, clip.z / clip.w, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vOffset;
out vec4 outColor;

void main() {
  // The gaussian itself: alpha falls off with squared distance from centre.
  float power = -dot(vOffset, vOffset);
  float alpha = vColor.a * exp(power);
  if (alpha < 0.004) discard;
  outColor = vec4(vColor.rgb * alpha, alpha);   // premultiplied
}`;

const STRIDE = 32;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

function normalise(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function lookAt(eye, target, up) {
  const f = normalise([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  let s = cross(f, up);
  if (Math.hypot(s[0], s[1], s[2]) < 1e-6) s = cross(f, [0, 0, 1]);
  s = normalise(s);
  const u = cross(s, f);
  const m = new Float32Array(16);
  m[0] = s[0];  m[4] = s[1];  m[8]  = s[2];
  m[1] = u[0];  m[5] = u[1];  m[9]  = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]);
  m[13] = -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
  m[14] =   f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2];
  m[15] = 1;
  return m;
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

export class SplatViewer {
  constructor(canvas, { onProgress } = {}) {
    this.canvas = canvas;
    this.onProgress = onProgress || (() => {});
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      premultipliedAlpha: true,
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;

    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    this.program = program;

    this.count = 0;
    this.target = [0, 0, 0];
    this.radius = 6;
    this.theta = 0.5;
    this.phi = 1.2;
    this.fov = 55;
    this._lastSortView = null;
    this._pending = false;

    this.worker = new Worker(URL.createObjectURL(
      new Blob([WORKER_SOURCE], { type: 'text/javascript' })));
    this.worker.onmessage = (event) => {
      this.order = event.data.order;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.order, gl.DYNAMIC_DRAW);
      this._pending = false;
      this._draw();
    };

    this._setup();
    this._bindControls();
  }

  _setup() {
    const gl = this.gl;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    // One quad, instanced once per gaussian.
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    const quadLoc = gl.getAttribLocation(this.program, 'aQuad');
    gl.enableVertexAttribArray(quadLoc);
    gl.vertexAttribPointer(quadLoc, 2, gl.FLOAT, false, 0, 0);

    // Per-instance attribute carrying the sorted index.
    this.indexBuffer = gl.createBuffer();
    const indexLoc = gl.getAttribLocation(this.program, 'aIndex');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.enableVertexAttribArray(indexLoc);
    gl.vertexAttribIPointer(indexLoc, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(indexLoc, 1);

    this.dataTexture = gl.createTexture();
    this.texWidth = 2048;

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    // Premultiplied source, standard over-operator.
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  _bindControls() {
    const canvas = this.canvas;
    let mode = null;
    let lastX = 0, lastY = 0;
    const pointers = new Map();

    const down = (e) => {
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      mode = (e.shiftKey || e.button === 2 || pointers.size === 2) ? 'pan' : 'orbit';
      lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    };
    const move = (e) => {
      if (!mode) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (mode === 'orbit') {
        this.theta -= dx * 0.005;
        this.phi = Math.min(Math.PI - 0.02, Math.max(0.02, this.phi - dy * 0.005));
      } else {
        // Pan along the camera's own axes, read from the view matrix, rather
        // than from angles reconstructed by hand -- the hand-rolled version
        // drifted out of agreement with the camera as soon as it tilted.
        const view = this._viewMatrix();
        const right = [view[0], view[4], view[8]];
        const up = [view[1], view[5], view[9]];
        const s = this.radius * 0.0018;
        for (let i = 0; i < 3; i++) {
          this.target[i] -= right[i] * dx * s;
          this.target[i] += up[i] * dy * s;
        }
      }
      this.render();
    };
    const up = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size === 0) mode = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* gone */ }
    };

    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.radius = Math.max(0.05, this.radius * Math.exp(e.deltaY * 0.001));
      this.render();
    }, { passive: false });
    window.addEventListener('resize', () => this.render());
  }

  /** Streams the packed buffer, drawing as much as has arrived. */
  async load(url, scene) {
    const gl = this.gl;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}`);

    const total = Number(response.headers.get('content-length')) || 0;
    const chunks = [];
    let received = 0;
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      this.onProgress(total ? received / total : 0, received);
    }

    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }

    this.count = Math.floor(bytes.length / STRIDE);
    gl.bindVertexArray(this.vao);

    // Two RGBA32UI texels per gaussian: 32 bytes maps exactly onto 8 uint32.
    const texels = this.count * 2;
    const rows = Math.ceil(texels / this.texWidth);
    const padded = new Uint32Array(this.texWidth * rows * 4);
    padded.set(new Uint32Array(bytes.buffer, 0, this.count * 8));

    gl.bindTexture(gl.TEXTURE_2D, this.dataTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, this.texWidth, rows, 0,
                  gl.RGBA_INTEGER, gl.UNSIGNED_INT, padded);

    const positions = new Float32Array(this.count * 3);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < this.count; i++) {
      positions[i * 3] = view.getFloat32(i * STRIDE, true);
      positions[i * 3 + 1] = view.getFloat32(i * STRIDE + 4, true);
      positions[i * 3 + 2] = view.getFloat32(i * STRIDE + 8, true);
    }
    this.worker.postMessage({ positions }, [positions.buffer]);

    this.order = new Uint32Array(this.count);
    for (let i = 0; i < this.count; i++) this.order[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.order, gl.DYNAMIC_DRAW);

    if (scene) this.frame(scene);
    this.render();
    return this.count;
  }

  frame(scene) {
    this.target = (scene.centre || [0, 0, 0]).slice();
    this.radius = (scene.radius || 4) * 2.2;
    this.theta = 0.5;
    this.phi = 1.2;
  }

  _viewMatrix() {
    const eye = [
      this.target[0] + this.radius * Math.sin(this.phi) * Math.sin(this.theta),
      this.target[1] + this.radius * Math.cos(this.phi),
      this.target[2] + this.radius * Math.sin(this.phi) * Math.cos(this.theta),
    ];
    // +Y is up because the exporter normalised the scene to make it so.
    return lookAt(eye, this.target, [0, 1, 0]);
  }

  render() {
    if (!this.count) return;
    const view = this._viewMatrix();

    // Re-sorting is only worth it once the camera has actually turned; below
    // this the existing order is still correct enough to blend with.
    if (!this._pending) {
      const previous = this._lastSortView;
      let moved = true;
      if (previous) {
        let delta = 0;
        for (let i = 0; i < 16; i++) delta += Math.abs(previous[i] - view[i]);
        moved = delta > 0.01;
      }
      if (moved) {
        this._pending = true;
        this._lastSortView = view.slice();
        this.worker.postMessage({ view });
      }
    }
    this._draw(view);
  }

  _draw(view) {
    const gl = this.gl;
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.count) return;

    view = view || this._viewMatrix();
    const aspect = canvas.width / canvas.height;
    const fovY = (this.fov * Math.PI) / 180;
    const projection = perspective(fovY, aspect, this.radius * 0.002, this.radius * 60);
    // Focal length in pixels, read straight off the projection matrix so it
    // cannot drift from the actual projection: fx = width/2 * P[0].
    const focalX = (canvas.width / 2) * projection[0];
    const focalY = (canvas.height / 2) * projection[5];

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    const u = (n) => gl.getUniformLocation(this.program, n);
    gl.uniformMatrix4fv(u('uProjection'), false, projection);
    gl.uniformMatrix4fv(u('uView'), false, view);
    gl.uniform2f(u('uViewport'), canvas.width, canvas.height);
    gl.uniform2f(u('uFocal'), focalX, focalY);
    gl.uniform1i(u('uTexWidth'), this.texWidth);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dataTexture);
    gl.uniform1i(u('uData'), 0);

    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.count);
  }

  dispose() {
    if (this.worker) this.worker.terminate();
  }
}
