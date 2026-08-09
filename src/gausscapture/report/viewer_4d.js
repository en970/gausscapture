/*
 * Deformable gaussian splat renderer — WebGL2, no libraries, no build step.
 *
 * This is `splat_viewer.js` with a time axis. The covariance projection, the
 * eigen-decomposed ellipse axes, the premultiplied over-blend and the
 * instanced fan are the same arithmetic; what is new is that a gaussian's
 * position and orientation are functions of time, and that the camera is not
 * free.
 *
 * ── Deformation ──────────────────────────────────────────────────────────
 * The scene moves as a scaffold of at most 256 rigid nodes. Each dynamic
 * gaussian is bound to four of them with weights summing to one, and is
 * carried by their weighted rigid blend. The alternative — a per-gaussian
 * trajectory — would be a hundred times the bytes and would not fit in a page
 * you can put behind a link.
 *
 * Node transforms are interpolated on the CPU, once per frame, for K ≤ 256
 * nodes: an 8 KB texture upload against 400,000 gaussians of skinning that the
 * GPU does anyway. Interpolating per gaussian on the CPU would be the naive
 * reading of "play a 4D scene" and it is roughly 50,000 times more work.
 *
 * ── Sorting, honestly ────────────────────────────────────────────────────
 * Alpha blending is order-dependent: draw two overlapping gaussians the wrong
 * way round and the result is wrong in a way that reads as haze rather than as
 * a bug. Exact per-frame ordering of 400,000 *deformed* gaussians is not
 * affordable on integrated graphics, so this viewer makes three explicit
 * trades and says so rather than pretending it sorts exactly:
 *
 *   1. The sort is a 16-bit counting sort in a worker — linear in the gaussian
 *      count, not n·log n. Two gaussians landing in the same bucket may swap;
 *      a bucket spans 1/65536 of the scene's depth range, which is far below
 *      the size of a gaussian.
 *   2. Depth is computed without ever materialising a deformed position. The
 *      view's depth row is folded into the node transforms first, so a
 *      gaussian's depth is Σ wᵢ(gᵢ·p + hᵢ) — four dot products, about 34
 *      flops, against roughly 200 for skinning it properly and then projecting.
 *      The reduction is per node, so it costs 256 small matrix products per
 *      sort regardless of scene size.
 *   3. The worker is single-flight. If a sort is still running when the next
 *      frame is due, that frame is drawn with the previous order instead of
 *      queueing work that will already be stale. A one-frame-old order is
 *      wrong only for gaussians whose depth ordering changed within one
 *      thirtieth of a second of subject motion — which, for a head that moves
 *      centimetres in a scene that is metres deep, is a handful of adjacent
 *      pairs inside the subject and nothing at all in the background.
 *
 * When even that is too slow the viewer sheds gaussians rather than frames.
 * The file is ordered by importance within its dynamic and static groups, so
 * dropping the tail of each keeps the gaussians that cover the most screen.
 * The budget moves on a measured median frame time, never on a guess about the
 * device, and the page is told about it so the number on screen is the number
 * being drawn.
 *
 * ── The camera ───────────────────────────────────────────────────────────
 * A fixed-camera capture has parallax only across the angles the hand-held arc
 * actually swept. Outside that cone there is no evidence, and every method in
 * the literature hallucinates there. So the camera is parameterised relative
 * to the capture axis — (0, 0) *is* the pose the phone sat in — and is hard
 * clamped to ρ = √((a/a_max)² + (e/e_max)²) ≤ 1 with the half-angles read from
 * the file. The outer 15 % resists the radial component only, so the camera
 * slides along the rim rather than jamming in a corner. There is no warning
 * and no red: the boundary is a fact about the recording, not a user error,
 * and the page labels it instead.
 */

const INV_SQRT2 = 0.7071067811865476;
const MAGIC = 'GAUSS4D\0';
const HEADER_SIZE = 192;
const DIRECTORY_ENTRY_SIZE = 32;
const GEOM_STRIDE = 16;
const TRAJ_STRIDE = 14;
const ALIGNMENT = 16;

/* Any bit set in the low half of `flags` is a change this renderer cannot
 * fake. None are defined yet, so any of them is a refusal. */
const KNOWN_BREAKING_FLAGS = 0;
const FLAG_FIXED_POSE_VERIFIED = 1 << 16;
const FLAG_LOOP_PINGPONG = 1 << 17;

// ─────────────────────────────────────────────────────────────── the worker

export const WORKER_SOURCE = `
let positions = null;      // canonical means, Float32Array(N*3)
let skinIndex = null;      // Uint8Array(Nd*4)
let skinWeight = null;     // Float32Array(Nd*4)
let nodeRest = null;       // Float32Array(K*3)
let dynamicCount = 0;
let nodeCount = 0;
let depths = null;
let ids = null;
let counts = null;
let starts = null;

const BUCKETS = 65536;

self.onmessage = (event) => {
  const data = event.data;

  if (data.kind === 'scene') {
    positions = data.positions;
    skinIndex = data.skinIndex;
    skinWeight = data.skinWeight;
    nodeRest = data.nodeRest;
    dynamicCount = data.dynamicCount;
    nodeCount = data.nodeCount;
    counts = new Uint32Array(BUCKETS);
    starts = new Uint32Array(BUCKETS);
    return;
  }
  if (!positions) return;

  const { view, nodeQuat, nodeTrans, dynamicDraw, staticDraw } = data;
  const total = dynamicDraw + staticDraw;
  if (!depths || depths.length < total) {
    depths = new Int32Array(total);
    ids = new Uint32Array(total);
  }

  // Only the depth row of the view matrix affects ordering; its translation is
  // a constant offset and cannot reorder anything.
  const d0 = view[2], d1 = view[6], d2 = view[10];

  // Fold the view row into every node once: g = Rᵀd, h = d·(n+t) − g·n.
  const g = new Float32Array(nodeCount * 3);
  const h = new Float32Array(nodeCount);
  for (let j = 0; j < nodeCount; j++) {
    const w = nodeQuat[j * 4], x = nodeQuat[j * 4 + 1],
          y = nodeQuat[j * 4 + 2], z = nodeQuat[j * 4 + 3];
    // Columns of R are the rows of Rᵀ, so this is Rᵀ·d written out.
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
    const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
    const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);
    const gx = r00 * d0 + r10 * d1 + r20 * d2;
    const gy = r01 * d0 + r11 * d1 + r21 * d2;
    const gz = r02 * d0 + r12 * d1 + r22 * d2;
    g[j * 3] = gx; g[j * 3 + 1] = gy; g[j * 3 + 2] = gz;
    const nx = nodeRest[j * 3], ny = nodeRest[j * 3 + 1], nz = nodeRest[j * 3 + 2];
    h[j] = (nx + nodeTrans[j * 3]) * d0 + (ny + nodeTrans[j * 3 + 1]) * d1
         + (nz + nodeTrans[j * 3 + 2]) * d2 - (gx * nx + gy * ny + gz * nz);
  }

  let min = Infinity, max = -Infinity;
  let n = 0;

  for (let i = 0; i < dynamicDraw; i++) {
    const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
    let depth = 0;
    for (let k = 0; k < 4; k++) {
      const weight = skinWeight[i * 4 + k];
      if (weight === 0) continue;
      const j = skinIndex[i * 4 + k];
      depth += weight * (g[j * 3] * px + g[j * 3 + 1] * py + g[j * 3 + 2] * pz + h[j]);
    }
    const key = depth * 4096 | 0;
    depths[n] = key; ids[n] = i; n++;
    if (key > max) max = key;
    if (key < min) min = key;
  }

  for (let s = 0; s < staticDraw; s++) {
    const i = dynamicCount + s;
    const key = (positions[i * 3] * d0 + positions[i * 3 + 1] * d1
               + positions[i * 3 + 2] * d2) * 4096 | 0;
    depths[n] = key; ids[n] = i; n++;
    if (key > max) max = key;
    if (key < min) min = key;
  }

  counts.fill(0);
  const scale = max > min ? (BUCKETS - 1) / (max - min) : 0;
  for (let i = 0; i < n; i++) {
    depths[i] = (depths[i] - min) * scale | 0;
    counts[depths[i]]++;
  }
  starts[0] = 0;
  for (let i = 1; i < BUCKETS; i++) starts[i] = starts[i - 1] + counts[i - 1];

  // Back to front, and this is the one line where that is decided.
  //
  // The key is d·p with d taken from row 2 of the VIEW matrix, and lookAt
  // writes that row as -forward — so the key is -forward·p and it DECREASES
  // with distance from the camera. The lowest bucket is therefore the farthest
  // gaussian, and emitting ascending puts it at order[0], which is the first
  // instance drawn. With blendFuncSeparate(ONE, ONE_MINUS_SRC_ALPHA) over a
  // premultiplied fragment, later fragments composite OVER earlier ones, so
  // first-drawn must be farthest.
  //
  // This used to be the reversing form, order[n - 1 - starts[bucket]++], which
  // drew nearest-first and composited the background over the subject in every
  // scene — visible as haze and a washed-out subject, never as a crash. The
  // idiom is antimatter15's, where it is correct because the sort key comes
  // from viewProj: the projection's row 2 carries a negative (far+near)/(near-far)
  // factor that flips the sign. Dropping the projection dropped the sign flip.
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[starts[depths[i]]++] = ids[i];

  self.postMessage({ order }, [order.buffer]);
};
`;

// ────────────────────────────────────────────────────────────────── shaders

const VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 uProjection;
uniform mat4 uView;
uniform vec2 uViewport;
uniform vec2 uFocal;

uniform highp usampler2D uGeom;     // one RGBA32UI texel per gaussian
uniform highp sampler2D uColor;     // one RGBA8 texel per gaussian
uniform highp usampler2D uSkin;     // one RG32UI texel per dynamic gaussian
uniform highp sampler2D uNodes;     // K x 3 RGBA32F: quat, translation, rest

uniform int uGeomWidth;
uniform int uSkinWidth;
uniform int uDynamicCount;

uniform vec3 uPosOrigin;
uniform vec3 uPosSpan;
uniform float uLogMin;
uniform float uLogSpan;

in vec2 aQuad;
in uint aIndex;

out vec4 vColor;
out vec2 vOffset;

const float INV_SQRT2 = 0.7071067811865476;

vec3 quatRotate(vec4 q, vec3 v) {
  vec3 u = q.yzw;
  return v + 2.0 * cross(u, cross(u, v) + q.x * v);
}

vec4 quatMul(vec4 a, vec4 b) {
  return vec4(
    a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
    a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
    a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
    a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

void main() {
  int gid = int(aIndex);

  uvec4 packed = texelFetch(uGeom, ivec2(gid % uGeomWidth, gid / uGeomWidth), 0);

  vec3 posCode = vec3(uvec3(packed.x & 0xFFFFu, packed.x >> 16u, packed.y & 0xFFFFu));
  vec3 scaleCode = vec3(uvec3(packed.y >> 16u, packed.z & 0xFFFFu, packed.z >> 16u));
  vec3 centre = uPosOrigin + posCode / 65535.0 * uPosSpan;
  vec3 scale = exp(uLogMin + scaleCode / 65535.0 * uLogSpan);

  // Smallest-three: two bits name the component the unit norm reconstructs.
  // The 10-bit grid is signed about code 511, which is exactly zero — spread
  // evenly over the closed interval instead, zero falls at code 511.5 and a
  // gaussian at rest comes back 0.137° off. Must match SMALLEST_THREE_HALF in
  // gausscapture/export/quantise.py exactly.
  uint q32 = packed.w;
  uint largest = q32 >> 30u;
  vec3 kept = (vec3(uvec3((q32 >> 20u) & 1023u, (q32 >> 10u) & 1023u, q32 & 1023u)) - 511.0)
              / 511.0 * INV_SQRT2;
  float rest = sqrt(max(0.0, 1.0 - dot(kept, kept)));
  vec4 rotation =
      largest == 0u ? vec4(rest, kept.x, kept.y, kept.z)
    : largest == 1u ? vec4(kept.x, rest, kept.y, kept.z)
    : largest == 2u ? vec4(kept.x, kept.y, rest, kept.z)
                    : vec4(kept.x, kept.y, kept.z, rest);

  // Deformation runs here, before the covariance projection, because the
  // covariance is built from the *deformed* orientation.
  if (gid < uDynamicCount) {
    uvec2 skin = texelFetch(uSkin, ivec2(gid % uSkinWidth, gid / uSkinWidth), 0).xy;
    uvec4 ids = (uvec4(skin.x) >> uvec4(0u, 8u, 16u, 24u)) & 255u;
    vec4 weights = vec4((uvec4(skin.y) >> uvec4(0u, 8u, 16u, 24u)) & 255u) / 255.0;

    // Influence 0 carries the largest weight — the writer stores the row in
    // descending order — so it is the hemisphere reference, and no search is
    // needed to find one.
    vec4 reference = texelFetch(uNodes, ivec2(int(ids.x), 0), 0);

    vec3 blended = vec3(0.0);
    vec4 blendedRotation = vec4(0.0);
    for (int i = 0; i < 4; i++) {
      float weight = weights[i];
      if (weight <= 0.0) continue;
      int j = int(ids[i]);
      vec4 nodeQuat = texelFetch(uNodes, ivec2(j, 0), 0);
      vec3 nodeTrans = texelFetch(uNodes, ivec2(j, 1), 0).xyz;
      vec3 nodeRest = texelFetch(uNodes, ivec2(j, 2), 0).xyz;
      blended += weight * (quatRotate(nodeQuat, centre - nodeRest) + nodeRest + nodeTrans);
      blendedRotation += weight * (dot(nodeQuat, reference) < 0.0 ? -nodeQuat : nodeQuat);
    }
    centre = blended;
    rotation = quatMul(normalize(blendedRotation), rotation);
  }

  vec4 viewPos = uView * vec4(centre, 1.0);
  if (viewPos.z > -0.01) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  float w = rotation.x, x = rotation.y, y = rotation.z, z = rotation.w;
  mat3 R = mat3(
    1.0 - 2.0*(y*y + z*z),  2.0*(x*y + w*z),        2.0*(x*z - w*y),
    2.0*(x*y - w*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z + w*x),
    2.0*(x*z + w*y),        2.0*(y*z - w*x),        1.0 - 2.0*(x*x + y*y)
  );
  mat3 M = R * mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
  mat3 sigma = M * transpose(M);

  float invZ = 1.0 / viewPos.z;
  float invZ2 = invZ * invZ;
  mat3 J = mat3(
    uFocal.x * invZ, 0.0,             -uFocal.x * viewPos.x * invZ2,
    0.0,             uFocal.y * invZ, -uFocal.y * viewPos.y * invZ2,
    0.0,             0.0,             0.0
  );
  mat3 T = J * mat3(uView);
  mat3 cov = T * sigma * transpose(T);

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
  vec2 axis1 = min(sqrt(2.0 * lambda1), 1024.0) * major;
  vec2 axis2 = min(sqrt(2.0 * lambda2), 1024.0) * minor;

  vColor = texelFetch(uColor, ivec2(gid % uGeomWidth, gid / uGeomWidth), 0);
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
  float alpha = vColor.a * exp(-dot(vOffset, vOffset));
  if (alpha < 0.004) discard;
  outColor = vec4(vColor.rgb * alpha, alpha);
}`;

// ───────────────────────────────────────────────────────────── the container

/** Parse a `.g4d` buffer into typed views over the original bytes. */
export function parseG4D(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < HEADER_SIZE) throw new Error('Not a .g4d file: shorter than its header');
  const view = new DataView(buffer);

  let magic = '';
  for (let i = 0; i < 8; i++) magic += String.fromCharCode(bytes[i]);
  if (magic !== MAGIC) throw new Error('Not a .g4d file: wrong magic');

  const header = {
    versionMajor: view.getUint16(8, true),
    versionMinor: view.getUint16(10, true),
    flags: view.getUint32(12, true),
    headerSize: view.getUint32(16, true),
    directoryOffset: view.getUint32(20, true),
    chunkCount: view.getUint32(24, true),
    gaussianCount: view.getUint32(28, true),
    dynamicCount: view.getUint32(32, true),
    nodeCount: view.getUint32(36, true),
    frameCount: view.getUint32(40, true),
    skinK: view.getUint32(44, true),
    fps: view.getFloat32(48, true),
    clipSeconds: view.getFloat32(52, true),
    posOrigin: [view.getFloat32(56, true), view.getFloat32(60, true), view.getFloat32(64, true)],
    posSpan: [view.getFloat32(68, true), view.getFloat32(72, true), view.getFloat32(76, true)],
    logScaleMin: view.getFloat32(80, true),
    logScaleSpan: view.getFloat32(84, true),
    trajOrigin: [view.getFloat32(88, true), view.getFloat32(92, true), view.getFloat32(96, true)],
    trajSpan: [view.getFloat32(100, true), view.getFloat32(104, true), view.getFloat32(108, true)],
    captureEye: [view.getFloat32(112, true), view.getFloat32(116, true), view.getFloat32(120, true)],
    captureForward: [view.getFloat32(124, true), view.getFloat32(128, true), view.getFloat32(132, true)],
    captureUp: [view.getFloat32(136, true), view.getFloat32(140, true), view.getFloat32(144, true)],
    coneAzimuthDeg: view.getFloat32(148, true),
    coneElevationDeg: view.getFloat32(152, true),
    sceneCentre: [view.getFloat32(156, true), view.getFloat32(160, true), view.getFloat32(164, true)],
    sceneRadius: view.getFloat32(168, true),
  };

  if (header.versionMajor !== 1) {
    throw new Error(`This .g4d is version ${header.versionMajor}.${header.versionMinor}; ` +
                    'this viewer understands 1.x');
  }
  const unknown = header.flags & 0xffff & ~KNOWN_BREAKING_FLAGS;
  if (unknown) {
    throw new Error(`This .g4d needs features this viewer does not implement ` +
                    `(flags 0x${unknown.toString(16)}). Refusing rather than drawing it wrong.`);
  }
  header.fixedPoseVerified = Boolean(header.flags & FLAG_FIXED_POSE_VERIFIED);
  header.loop = (header.flags & FLAG_LOOP_PINGPONG) ? 'ping-pong' : 'forward';

  // The directory has to fit in the file before any of it is read, or a
  // malformed one surfaces as a raw DataView RangeError rather than as a
  // sentence about the file.
  const directoryBytes = header.chunkCount * DIRECTORY_ENTRY_SIZE;
  if (header.directoryOffset + directoryBytes > bytes.length) {
    throw new Error(`This .g4d declares ${header.chunkCount} chunks but its directory ` +
                    'does not fit in the file; it is truncated.');
  }

  const chunks = {};
  for (let i = 0; i < header.chunkCount; i++) {
    const at = header.directoryOffset + i * DIRECTORY_ENTRY_SIZE;
    let tag = '';
    for (let c = 0; c < 4; c++) tag += String.fromCharCode(bytes[at + c]);
    // Offsets are 64-bit in the file; a browser cannot address past 2^53 bytes
    // anyway, so the high word only ever needs checking, not keeping.
    if (view.getUint32(at + 12, true) || view.getUint32(at + 20, true)) {
      throw new Error('This .g4d is larger than a browser can address');
    }
    const offset = view.getUint32(at + 8, true);
    const length = view.getUint32(at + 16, true);
    if (offset + length > bytes.length) throw new Error(`Chunk ${tag} is truncated`);
    // Every typed view below is built from `offset` and a count taken from the
    // header, so an offset off the 16-byte grid is a view a browser refuses to
    // construct — and a length that disagrees with the header is a view over
    // whatever chunk happens to follow. read_scene4d checks both; this reader
    // did neither, so a file every Python tool refused rendered silently as
    // garbage. Same checks, same shape of message.
    if (offset % ALIGNMENT) {
      throw new Error(`Chunk ${tag} starts at ${offset}, which is not ${ALIGNMENT}-byte ` +
                      'aligned; a reader cannot take a typed view over it.');
    }
    chunks[tag] = { offset, length };
  }

  const need = (tag, implied) => {
    if (!chunks[tag]) throw new Error(`This .g4d has no ${tag} chunk, so it cannot be drawn`);
    const chunk = chunks[tag];
    if (chunk.length !== implied) {
      throw new Error(`Chunk ${tag} is ${chunk.length} bytes; the header implies ${implied}`);
    }
    return chunk;
  };

  const geom = need('GEOM', header.gaussianCount * GEOM_STRIDE);
  const colr = need('COLR', header.gaussianCount * 4);
  const node = need('NODE', header.nodeCount * 12);
  const traj = need('TRAJ', header.frameCount * header.nodeCount * TRAJ_STRIDE);

  const scene = {
    header,
    geom: new Uint32Array(buffer, geom.offset, header.gaussianCount * 4),
    colors: new Uint8Array(buffer, colr.offset, header.gaussianCount * 4),
    nodeRest: new Float32Array(buffer, node.offset, header.nodeCount * 3),
    traj: new Uint8Array(buffer, traj.offset, header.frameCount * header.nodeCount * TRAJ_STRIDE),
    meta: {},
  };

  if (chunks.META) {
    const text = new TextDecoder().decode(new Uint8Array(buffer, chunks.META.offset, chunks.META.length));
    try { scene.meta = JSON.parse(text); } catch { scene.meta = {}; }
  }

  // TRAJ is decoded once at load: 90 × 256 × 14 bytes is 322 KB, and decoding
  // it lazily per frame would put a parse in the animation loop for no gain.
  const frames = header.frameCount, nodes = header.nodeCount;
  scene.nodeQuats = new Float32Array(frames * nodes * 4);
  scene.nodeTrans = new Float32Array(frames * nodes * 3);
  const trajView = new DataView(buffer, traj.offset, traj.length);
  for (let i = 0; i < frames * nodes; i++) {
    const at = i * TRAJ_STRIDE;
    let nx = 0, ny = 0, nz = 0, nw = 0;
    nw = trajView.getInt16(at, true) / 32767;
    nx = trajView.getInt16(at + 2, true) / 32767;
    ny = trajView.getInt16(at + 4, true) / 32767;
    nz = trajView.getInt16(at + 6, true) / 32767;
    const norm = Math.hypot(nw, nx, ny, nz) || 1;
    scene.nodeQuats[i * 4] = nw / norm;
    scene.nodeQuats[i * 4 + 1] = nx / norm;
    scene.nodeQuats[i * 4 + 2] = ny / norm;
    scene.nodeQuats[i * 4 + 3] = nz / norm;
    for (let c = 0; c < 3; c++) {
      scene.nodeTrans[i * 3 + c] = header.trajOrigin[c]
        + trajView.getUint16(at + 8 + c * 2, true) / 65535 * header.trajSpan[c];
    }
  }

  // SKIN is padded to four influences so the shader has one fixed layout.
  const k = header.skinK;
  scene.skinIndex = new Uint8Array(header.dynamicCount * 4);
  scene.skinWeight = new Float32Array(header.dynamicCount * 4);
  scene.skinPacked = new Uint32Array(header.dynamicCount * 2);
  if (header.dynamicCount) {
    const skin = need('SKIN', header.dynamicCount * 2 * k);
    const raw = new Uint8Array(buffer, skin.offset, header.dynamicCount * 2 * k);
    for (let i = 0; i < header.dynamicCount; i++) {
      let indexWord = 0, weightWord = 0;
      for (let c = 0; c < k; c++) {
        const id = raw[i * 2 * k + c];
        const weight = raw[i * 2 * k + k + c];
        scene.skinIndex[i * 4 + c] = id;
        scene.skinWeight[i * 4 + c] = weight / 255;
        indexWord |= id << (8 * c);
        weightWord |= weight << (8 * c);
      }
      scene.skinPacked[i * 2] = indexWord >>> 0;
      scene.skinPacked[i * 2 + 1] = weightWord >>> 0;
    }
  }

  // Canonical positions, dequantised once for the sort worker.
  scene.positions = new Float32Array(header.gaussianCount * 3);
  for (let i = 0; i < header.gaussianCount; i++) {
    const w0 = scene.geom[i * 4], w1 = scene.geom[i * 4 + 1];
    const codes = [w0 & 0xffff, w0 >>> 16, w1 & 0xffff];
    for (let c = 0; c < 3; c++) {
      scene.positions[i * 3 + c] = header.posOrigin[c] + codes[c] / 65535 * header.posSpan[c];
    }
  }

  return scene;
}

// ───────────────────────────────────────────────────────────── linear algebra

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function normalise(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function rotateAbout(axis, angle, v) {
  // Rodrigues, because the camera rail is defined by two axes of the capture
  // frame rather than by world yaw and pitch.
  const [x, y, z] = normalise(axis);
  const c = Math.cos(angle), s = Math.sin(angle);
  const dot = x * v[0] + y * v[1] + z * v[2];
  const cr = cross([x, y, z], v);
  return [
    v[0] * c + cr[0] * s + x * dot * (1 - c),
    v[1] * c + cr[1] * s + y * dot * (1 - c),
    v[2] * c + cr[2] * s + z * dot * (1 - c),
  ];
}

export function lookAt(eye, target, up) {
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

// ────────────────────────────────────────────────────────────────── the view

/** Where the drag resistance starts, as a fraction of the cone radius. */
const RESIST_FROM = 0.85;
/** Pixels of drag that traverse the cone from one edge to the other. */
const TRAVERSAL_PX = 320;

export class Scene4DViewer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.onProgress = options.onProgress || (() => {});
    this.onCamera = options.onCamera || (() => {});
    this.onTime = options.onTime || (() => {});
    this.onBudget = options.onBudget || (() => {});

    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;

    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    this.program = program;
    this.uniforms = {};

    this.scene = null;
    this.count = 0;
    this.drawCount = 0;
    this.budget = Infinity;

    // Camera, in capture-relative coordinates. (0, 0) is the pose the phone
    // actually sat in, which is why it is also the initial view.
    this.azimuth = 0;
    this.elevation = 0;
    this.zoom = 1;
    this.coneAzimuth = 0.001;
    this.coneElevation = 0.001;

    // Playback. Tau is continuous, in frames, and driven by a wall clock —
    // a one-second clip stepped frame by frame is over before a user finishes
    // a single camera drag, and continuous tau is what makes this bullet time
    // rather than a thirty-frame flipbook.
    this.tau = 0;
    this.rate = 0.5;
    this.playing = false;
    this.loop = 'ping-pong';
    this.direction = 1;
    this._lastTick = 0;
    this._frameTimes = [];

    this._pendingSort = false;
    this._sortView = null;
    this._sortTau = -1e9;
    this._raf = 0;

    this.worker = new Worker(URL.createObjectURL(new Blob([WORKER_SOURCE], { type: 'text/javascript' })));
    this.worker.onmessage = (event) => {
      this.order = event.data.order;
      const gl2 = this.gl;
      gl2.bindBuffer(gl2.ARRAY_BUFFER, this.indexBuffer);
      gl2.bufferData(gl2.ARRAY_BUFFER, this.order, gl2.DYNAMIC_DRAW);
      this._pendingSort = false;
    };

    this._setup();
    this._bindControls();
  }

  _setup() {
    const gl = this.gl;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    const quadLoc = gl.getAttribLocation(this.program, 'aQuad');
    gl.enableVertexAttribArray(quadLoc);
    gl.vertexAttribPointer(quadLoc, 2, gl.FLOAT, false, 0, 0);

    this.indexBuffer = gl.createBuffer();
    const indexLoc = gl.getAttribLocation(this.program, 'aIndex');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.enableVertexAttribArray(indexLoc);
    gl.vertexAttribIPointer(indexLoc, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(indexLoc, 1);

    this.geomTexture = gl.createTexture();
    this.colorTexture = gl.createTexture();
    this.skinTexture = gl.createTexture();
    this.nodeTexture = gl.createTexture();
    this.texWidth = 2048;

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  _uniform(name) {
    if (!(name in this.uniforms)) this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
    return this.uniforms[name];
  }

  // ───────────────────────────────────────────────────────────── loading

  /** Fetch a `.g4d` and upload it. Streams so the page can show progress. */
  async load(url) {
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
    let at = 0;
    for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
    return this.loadBuffer(bytes.buffer);
  }

  /** Upload an already-resident `.g4d`, as the single-file page has. */
  loadBuffer(buffer) {
    const gl = this.gl;
    const scene = parseG4D(buffer);
    this.scene = scene;
    const header = scene.header;

    this.count = header.gaussianCount;
    this.drawCount = this.count;
    this.budget = this.count;
    this.loop = header.loop;
    this.coneAzimuth = Math.max(header.coneAzimuthDeg, 0.001) * Math.PI / 180;
    this.coneElevation = Math.max(header.coneElevationDeg, 0.001) * Math.PI / 180;

    gl.bindVertexArray(this.vao);

    const rows = Math.ceil(this.count / this.texWidth);
    const geomPadded = new Uint32Array(this.texWidth * rows * 4);
    geomPadded.set(scene.geom);
    this._upload(this.geomTexture, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT,
                 this.texWidth, rows, geomPadded);

    const colorPadded = new Uint8Array(this.texWidth * rows * 4);
    colorPadded.set(scene.colors);
    this._upload(this.colorTexture, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE,
                 this.texWidth, rows, colorPadded);

    this.skinWidth = 2048;
    const skinRows = Math.max(1, Math.ceil(header.dynamicCount / this.skinWidth));
    const skinPadded = new Uint32Array(this.skinWidth * skinRows * 2);
    skinPadded.set(scene.skinPacked);
    this._upload(this.skinTexture, gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT,
                 this.skinWidth, skinRows, skinPadded);

    // Three rows of K: rotation, translation, rest position. Rows 0 and 1 are
    // rewritten every frame; row 2 never changes.
    this.nodeBuffer = new Float32Array(header.nodeCount * 3 * 4);
    for (let j = 0; j < header.nodeCount; j++) {
      this.nodeBuffer[(2 * header.nodeCount + j) * 4] = scene.nodeRest[j * 3];
      this.nodeBuffer[(2 * header.nodeCount + j) * 4 + 1] = scene.nodeRest[j * 3 + 1];
      this.nodeBuffer[(2 * header.nodeCount + j) * 4 + 2] = scene.nodeRest[j * 3 + 2];
    }
    this._upload(this.nodeTexture, gl.RGBA32F, gl.RGBA, gl.FLOAT,
                 header.nodeCount, 3, this.nodeBuffer);

    this.order = new Uint32Array(this.count);
    for (let i = 0; i < this.count; i++) this.order[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.order, gl.DYNAMIC_DRAW);

    this.worker.postMessage({
      kind: 'scene',
      positions: scene.positions,
      skinIndex: scene.skinIndex,
      skinWeight: scene.skinWeight,
      nodeRest: scene.nodeRest,
      dynamicCount: header.dynamicCount,
      nodeCount: header.nodeCount,
    });

    this._frameCamera();
    this.tau = 0;
    this._sortTau = -1e9;
    this._emitTime();
    this.onBudget({ drawing: this.count, total: this.count, medianFrameMs: 0 });
    this.start();
    return scene;
  }

  _upload(texture, internal, format, type, width, height, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, data);
  }

  // ────────────────────────────────────────────────────────────── camera

  _frameCamera() {
    const header = this.scene.header;
    // The subject, not the scene: the background is a wall behind the person
    // and orbiting its centroid would swing the head across the frame.
    let target = header.sceneCentre.slice();
    if (header.dynamicCount) {
      let x = 0, y = 0, z = 0;
      const n = header.dynamicCount;
      for (let i = 0; i < n; i++) {
        x += this.scene.positions[i * 3];
        y += this.scene.positions[i * 3 + 1];
        z += this.scene.positions[i * 3 + 2];
      }
      target = [x / n, y / n, z / n];
    }
    this.target = target;

    let offset = [
      header.captureEye[0] - target[0],
      header.captureEye[1] - target[1],
      header.captureEye[2] - target[2],
    ];
    if (Math.hypot(offset[0], offset[1], offset[2]) < 1e-4) {
      // No usable capture pose recorded; fall back to a framing of the scene,
      // and say so, because (0,0) is then no longer the pose the phone sat in.
      offset = [0, 0, Math.max(header.sceneRadius, 0.1) * 2.5];
      this.captureEyeKnown = false;
    } else {
      this.captureEyeKnown = true;
    }
    this.captureOffset = offset;

    const forward = normalise(header.captureForward);
    let up = normalise(header.captureUp);
    if (Math.abs(forward[0] * up[0] + forward[1] * up[1] + forward[2] * up[2]) > 0.98) {
      up = [0, 1, 0];
    }
    this.axisUp = normalise(cross(cross(forward, up), forward));
    this.axisRight = normalise(cross(forward, this.axisUp));
    this.azimuth = 0;
    this.elevation = 0;
    this.zoom = 1;
    this._emitCamera();
  }

  /** ρ: how far out in the trained cone the camera currently is, 0 at capture. */
  get rho() {
    return Math.hypot(this.azimuth / this.coneAzimuth, this.elevation / this.coneElevation);
  }

  _emitCamera() {
    this.onCamera({
      azimuthDeg: this.azimuth * 180 / Math.PI,
      elevationDeg: this.elevation * 180 / Math.PI,
      coneAzimuthDeg: this.coneAzimuth * 180 / Math.PI,
      coneElevationDeg: this.coneElevation * 180 / Math.PI,
      rho: this.rho,
      captureEyeKnown: this.captureEyeKnown !== false,
    });
  }

  /**
   * Move the camera by a drag, respecting the cone.
   *
   * The whole clamp lives here. Working in units of the cone half-angles turns
   * an ellipse into a unit disc, which makes "slide along the rim" a two-line
   * decomposition instead of a special case per corner.
   */
  nudge(deltaAzimuth, deltaElevation) {
    let u = this.azimuth / this.coneAzimuth;
    let v = this.elevation / this.coneElevation;
    let du = deltaAzimuth / this.coneAzimuth;
    let dv = deltaElevation / this.coneElevation;

    const rho = Math.hypot(u, v);
    if (rho > RESIST_FROM) {
      const ru = u / rho, rv = v / rho;
      const radial = du * ru + dv * rv;
      if (radial > 0) {
        // Resist only outward motion, and only its radial part: the tangential
        // part is untouched, so the camera keeps sliding around the rim.
        const slack = Math.max(0, 1 - (rho - RESIST_FROM) / (1 - RESIST_FROM));
        du -= radial * ru * (1 - slack);
        dv -= radial * rv * (1 - slack);
      }
    }

    u += du;
    v += dv;
    const out = Math.hypot(u, v);
    if (out > 1) { u /= out; v /= out; }

    this.azimuth = u * this.coneAzimuth;
    this.elevation = v * this.coneElevation;
    this._emitCamera();
  }

  setZoom(factor) {
    this.zoom = Math.min(3.5, Math.max(0.35, factor));
  }

  _viewMatrix() {
    let offset = rotateAbout(this.axisRight, this.elevation, this.captureOffset);
    offset = rotateAbout(this.axisUp, this.azimuth, offset);
    const eye = [
      this.target[0] + offset[0] * this.zoom,
      this.target[1] + offset[1] * this.zoom,
      this.target[2] + offset[2] * this.zoom,
    ];
    return lookAt(eye, this.target, this.axisUp);
  }

  _bindControls() {
    const canvas = this.canvas;
    let dragging = false;
    let lastX = 0, lastY = 0;

    const down = (e) => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    };
    const move = (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      // A full traversal of the cone is one 320-pixel drag, whatever the cone
      // happens to be, so a narrow cone does not feel like a stuck control.
      this.nudge(
        -dx * (2 * this.coneAzimuth) / TRAVERSAL_PX,
        -dy * (2 * this.coneElevation) / TRAVERSAL_PX,
      );
    };
    const up = (e) => {
      dragging = false;
      try { canvas.releasePointerCapture(e.pointerId); } catch { /* already gone */ }
    };

    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.setZoom(this.zoom * Math.exp(e.deltaY * 0.001));
    }, { passive: false });
  }

  // ──────────────────────────────────────────────────────────── playback

  start() {
    if (this._raf) return;
    this._lastTick = performance.now();
    const tick = (now) => {
      this._raf = requestAnimationFrame(tick);
      const dt = Math.min((now - this._lastTick) / 1000, 0.25);
      this._lastTick = now;
      if (this.playing) this._advance(dt);
      this._render(now);
    };
    this._raf = requestAnimationFrame(tick);
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
  }

  play() { this.playing = true; }
  pause() { this.playing = false; }
  togglePlay() { this.playing = !this.playing; }

  setRate(rate) { this.rate = rate; }
  setLoop(mode) { this.loop = mode; this.direction = 1; }

  seek(tau) {
    const last = Math.max(this.scene.header.frameCount - 1, 0);
    this.tau = Math.min(last, Math.max(0, tau));
    this._emitTime();
  }

  _advance(dt) {
    const header = this.scene.header;
    const last = Math.max(header.frameCount - 1, 0);
    if (last === 0) return;
    this.tau += this.direction * dt * header.fps * this.rate;

    if (this.loop === 'ping-pong') {
      // A one-to-three second clip has no reason to be cyclic — the subject
      // does not end where they began — so cutting back to frame zero is a
      // visible jump every loop. Ping-pong replays the motion backwards
      // instead, which is wrong about time and right about continuity, and
      // the transport says which mode is on.
      if (this.tau > last) { this.tau = last - (this.tau - last); this.direction = -1; }
      if (this.tau < 0) { this.tau = -this.tau; this.direction = 1; }
    } else if (this.loop === 'forward') {
      if (this.tau > last) this.tau -= last;
      if (this.tau < 0) this.tau += last;
    } else {
      if (this.tau >= last) { this.tau = last; this.playing = false; }
    }
    this.tau = Math.min(last, Math.max(0, this.tau));
    this._emitTime();
  }

  _emitTime() {
    const header = this.scene.header;
    this.onTime({
      tau: this.tau,
      frame: Math.round(this.tau),
      seconds: this.tau / header.fps,
      clipSeconds: header.clipSeconds,
      frameCount: header.frameCount,
      playing: this.playing,
    });
  }

  /** Interpolate the scaffold to the current tau and upload it. */
  _uploadNodes() {
    const gl = this.gl;
    const header = this.scene.header;
    const K = header.nodeCount;
    const last = Math.max(header.frameCount - 1, 0);
    const tau = Math.min(last, Math.max(0, this.tau));
    const lower = Math.floor(tau);
    const upper = Math.min(lower + 1, last);
    const alpha = tau - lower;

    const q = this.scene.nodeQuats, t = this.scene.nodeTrans;
    for (let j = 0; j < K; j++) {
      const a = (lower * K + j) * 4, b = (upper * K + j) * 4;
      let bw = q[b], bx = q[b + 1], by = q[b + 2], bz = q[b + 3];
      // Hemisphere alignment per node: two nodes in the same frame can sit on
      // opposite sides, so a whole-frame flip would fix one and break another.
      if (q[a] * bw + q[a + 1] * bx + q[a + 2] * by + q[a + 3] * bz < 0) {
        bw = -bw; bx = -bx; by = -by; bz = -bz;
      }
      let nw = q[a] + (bw - q[a]) * alpha;
      let nx = q[a + 1] + (bx - q[a + 1]) * alpha;
      let ny = q[a + 2] + (by - q[a + 2]) * alpha;
      let nz = q[a + 3] + (bz - q[a + 3]) * alpha;
      const inv = 1 / (Math.hypot(nw, nx, ny, nz) || 1);
      this.nodeBuffer[j * 4] = nw * inv;
      this.nodeBuffer[j * 4 + 1] = nx * inv;
      this.nodeBuffer[j * 4 + 2] = ny * inv;
      this.nodeBuffer[j * 4 + 3] = nz * inv;

      const at = (lower * K + j) * 3, bt = (upper * K + j) * 3;
      for (let c = 0; c < 3; c++) {
        this.nodeBuffer[(K + j) * 4 + c] = t[at + c] + (t[bt + c] - t[at + c]) * alpha;
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, this.nodeTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, K, 2, gl.RGBA, gl.FLOAT,
                     this.nodeBuffer.subarray(0, K * 2 * 4));
  }

  // ───────────────────────────────────────────────────────────── drawing

  _adaptBudget(elapsed) {
    // A rolling median rather than the last frame: one long frame is a browser
    // doing something else, and reacting to it would make the count flicker.
    this._frameTimes.push(elapsed);
    if (this._frameTimes.length < 30) return;
    const sorted = this._frameTimes.slice().sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    this._frameTimes.length = 0;

    const before = this.budget;
    if (median > 32 && this.budget > 40000) {
      this.budget = Math.max(40000, Math.floor(this.budget * 0.82));
    } else if (median < 14 && this.budget < this.count) {
      this.budget = Math.min(this.count, Math.floor(this.budget * 1.15) + 1000);
    }
    if (before !== this.budget) {
      this.onBudget({ drawing: Math.min(this.budget, this.count), total: this.count,
                      medianFrameMs: median });
    }
  }

  _drawCounts() {
    const header = this.scene.header;
    const budget = Math.min(this.budget, this.count);
    if (budget >= this.count) {
      return [header.dynamicCount, this.count - header.dynamicCount];
    }
    // Shed from both groups in proportion: dropping only the background leaves
    // a subject floating in nothing, and dropping only the subject defeats the
    // point of the recording.
    const fraction = budget / this.count;
    const dynamic = Math.min(header.dynamicCount, Math.max(1, Math.round(header.dynamicCount * fraction)));
    const staticCount = Math.max(0, Math.min(this.count - header.dynamicCount, budget - dynamic));
    return [dynamic, staticCount];
  }

  _render(now) {
    if (!this.scene) return;
    const started = performance.now();

    this._uploadNodes();
    const view = this._viewMatrix();
    const [dynamicDraw, staticDraw] = this._drawCounts();
    this.drawCount = dynamicDraw + staticDraw;

    if (!this._pendingSort) {
      let moved = !this._sortView;
      if (this._sortView) {
        let delta = 0;
        for (let i = 0; i < 16; i++) delta += Math.abs(this._sortView[i] - view[i]);
        moved = delta > 0.004;
      }
      // Time changes the order too, but far more slowly than the camera does,
      // so it only earns a re-sort once the subject has moved a whole frame's
      // worth. Between those the previous order is stale only inside the
      // subject, and only between gaussians at nearly equal depth.
      const timeMoved = Math.abs(this.tau - this._sortTau) > 1;
      if (moved || timeMoved) {
        this._pendingSort = true;
        this._sortView = view.slice();
        this._sortTau = this.tau;
        this.worker.postMessage({
          view,
          nodeQuat: this.nodeBuffer.slice(0, this.scene.header.nodeCount * 4),
          nodeTrans: this._nodeTransFlat(),
          dynamicDraw, staticDraw,
        });
      }
    }

    this._draw(view);
    this._adaptBudget(performance.now() - started);
  }

  _nodeTransFlat() {
    const K = this.scene.header.nodeCount;
    const out = new Float32Array(K * 3);
    for (let j = 0; j < K; j++) {
      out[j * 3] = this.nodeBuffer[(K + j) * 4];
      out[j * 3 + 1] = this.nodeBuffer[(K + j) * 4 + 1];
      out[j * 3 + 2] = this.nodeBuffer[(K + j) * 4 + 2];
    }
    return out;
  }

  _draw(view) {
    const gl = this.gl;
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.043, 0.043, 0.051, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.count) return;

    const header = this.scene.header;
    const radius = Math.max(header.sceneRadius, 0.05);
    const projection = perspective(55 * Math.PI / 180, canvas.width / canvas.height,
                                   radius * 0.01, radius * 60);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this._uniform('uProjection'), false, projection);
    gl.uniformMatrix4fv(this._uniform('uView'), false, view);
    gl.uniform2f(this._uniform('uViewport'), canvas.width, canvas.height);
    gl.uniform2f(this._uniform('uFocal'),
                 (canvas.width / 2) * projection[0], (canvas.height / 2) * projection[5]);
    gl.uniform1i(this._uniform('uGeomWidth'), this.texWidth);
    gl.uniform1i(this._uniform('uSkinWidth'), this.skinWidth);
    gl.uniform1i(this._uniform('uDynamicCount'), header.dynamicCount);
    gl.uniform3fv(this._uniform('uPosOrigin'), header.posOrigin);
    gl.uniform3fv(this._uniform('uPosSpan'), header.posSpan);
    gl.uniform1f(this._uniform('uLogMin'), header.logScaleMin);
    gl.uniform1f(this._uniform('uLogSpan'), header.logScaleSpan);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.geomTexture);
    gl.uniform1i(this._uniform('uGeom'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTexture);
    gl.uniform1i(this._uniform('uColor'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.skinTexture);
    gl.uniform1i(this._uniform('uSkin'), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.nodeTexture);
    gl.uniform1i(this._uniform('uNodes'), 3);

    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, Math.min(this.order.length, this.drawCount));
  }

  dispose() {
    this.stop();
    if (this.worker) this.worker.terminate();
  }
}

/**
 * The deformation contract, in JavaScript, for the cross-implementation check.
 *
 * The shader is the implementation that matters, but a shader cannot be run
 * from a test runner. This is the same arithmetic written straight, and
 * `tests/test_viewer_contract.py` runs it under Node against the NumPy
 * reference in `export/deform_reference.py` — over a real `.g4d` this repository
 * wrote, and over hand-built inputs for the hemisphere alignment, which a
 * well-formed file never exercises. If the two disagree, one of the three
 * implementations has drifted, and the browser is the one nobody would notice.
 */
export function deformPoint(point, quat, nodeRest, nodeQuats, nodeTrans, indices, weights) {
  const acc = [0, 0, 0];
  const blend = [0, 0, 0, 0];
  const reference = nodeQuats[indices[0]];

  for (let i = 0; i < indices.length; i++) {
    const weight = weights[i];
    if (weight === 0) continue;
    const j = indices[i];
    const q = nodeQuats[j], n = nodeRest[j], t = nodeTrans[j];
    const local = [point[0] - n[0], point[1] - n[1], point[2] - n[2]];
    const u = [q[1], q[2], q[3]];
    const c1 = cross(u, local);
    const c2 = cross(u, [c1[0] + q[0] * local[0], c1[1] + q[0] * local[1], c1[2] + q[0] * local[2]]);
    for (let c = 0; c < 3; c++) {
      acc[c] += weight * (local[c] + 2 * c2[c] + n[c] + t[c]);
    }
    const sign = (q[0] * reference[0] + q[1] * reference[1]
                + q[2] * reference[2] + q[3] * reference[3]) < 0 ? -1 : 1;
    for (let c = 0; c < 4; c++) blend[c] += weight * sign * q[c];
  }

  const inv = 1 / (Math.hypot(blend[0], blend[1], blend[2], blend[3]) || 1);
  const b = blend.map((v) => v * inv);
  const rotated = [
    b[0] * quat[0] - b[1] * quat[1] - b[2] * quat[2] - b[3] * quat[3],
    b[0] * quat[1] + b[1] * quat[0] + b[2] * quat[3] - b[3] * quat[2],
    b[0] * quat[2] - b[1] * quat[3] + b[2] * quat[0] + b[3] * quat[1],
    b[0] * quat[3] + b[1] * quat[2] - b[2] * quat[1] + b[3] * quat[0],
  ];
  return { position: acc, quaternion: rotated };
}

/** Interpolate a scaffold to continuous frame index `tau`, as the viewer does. */
export function sampleNodes(nodeQuats, nodeTrans, tau) {
  const frames = nodeQuats.length;
  const clamped = Math.min(frames - 1, Math.max(0, tau));
  const lower = Math.floor(clamped);
  const upper = Math.min(lower + 1, frames - 1);
  const alpha = clamped - lower;

  const quats = [];
  const trans = [];
  for (let j = 0; j < nodeQuats[lower].length; j++) {
    const a = nodeQuats[lower][j];
    let b = nodeQuats[upper][j];
    if (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] < 0) b = b.map((v) => -v);
    const blended = a.map((v, c) => v + (b[c] - v) * alpha);
    const inv = 1 / (Math.hypot(blended[0], blended[1], blended[2], blended[3]) || 1);
    quats.push(blended.map((v) => v * inv));
    trans.push(nodeTrans[lower][j].map((v, c) => v + (nodeTrans[upper][j][c] - v) * alpha));
  }
  return { quats, trans };
}
