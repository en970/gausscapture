const els = {
  status: document.getElementById("status"),
  preview: document.getElementById("preview"),
  timer: document.getElementById("timer"),
  apiBase: document.getElementById("apiBase"),
  projectId: document.getElementById("projectId"),
  sessionName: document.getElementById("sessionName"),
  targetType: document.getElementById("targetType"),
  recordAudio: document.getElementById("recordAudio"),
  startCamera: document.getElementById("startCamera"),
  motionPermission: document.getElementById("motionPermission"),
  startRecord: document.getElementById("startRecord"),
  stopRecord: document.getElementById("stopRecord"),
  downloadPack: document.getElementById("downloadPack"),
  uploadVideo: document.getElementById("uploadVideo"),
  videoInfo: document.getElementById("videoInfo"),
  motionInfo: document.getElementById("motionInfo"),
  orientationInfo: document.getElementById("orientationInfo"),
  projectInfo: document.getElementById("projectInfo"),
  log: document.getElementById("log")
};

let stream = null;
let recorder = null;
let chunks = [];
let recordedBlob = null;
let recordedMime = "";
let startedAt = 0;
// Sensor samples must share a clock with the video, and Date.now() is not it:
// it is wall time and can jump. performance.now() is monotonic, so recording
// start is stamped on both clocks and every sample is expressed as
// milliseconds since that instant. Without this the logs are unalignable.
let recordingPerfStart = 0;
let timerId = 0;
let motionLog = [];
let orientationLog = [];
let cameraSettings = {};

els.apiBase.value = location.origin;

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(() => undefined);
}

function setStatus(value) {
  els.status.textContent = value;
}

function log(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  els.log.textContent = `${line}\n${els.log.textContent}`.slice(0, 5000);
}

function updateTimer() {
  const sec = Math.floor((Date.now() - startedAt) / 1000);
  const min = Math.floor(sec / 60).toString().padStart(2, "0");
  const rem = (sec % 60).toString().padStart(2, "0");
  els.timer.textContent = `${min}:${rem}`;
}

els.startCamera.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30 }
      },
      audio: els.recordAudio.checked
    });
    els.preview.srcObject = stream;
    const videoTrack = stream.getVideoTracks()[0];
    cameraSettings = videoTrack ? videoTrack.getSettings() : {};
    els.videoInfo.textContent = `${cameraSettings.width || "?"}x${cameraSettings.height || "?"} ${cameraSettings.frameRate || "?"}fps`;
    els.startRecord.disabled = false;
    setStatus("Camera Ready");
    log("Camera started");
  } catch (error) {
    log(`Camera error: ${error.message}`);
    setStatus("Camera Error");
  }
});

els.motionPermission.addEventListener("click", async () => {
  try {
    if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
      const result = await DeviceMotionEvent.requestPermission();
      log(`Motion permission: ${result}`);
    }
    if (typeof DeviceOrientationEvent !== "undefined" && typeof DeviceOrientationEvent.requestPermission === "function") {
      const result = await DeviceOrientationEvent.requestPermission();
      log(`Orientation permission: ${result}`);
    }
    window.addEventListener("devicemotion", onMotion);
    window.addEventListener("deviceorientation", onOrientation);
    log("Motion/orientation logging enabled");
  } catch (error) {
    log(`Motion permission error: ${error.message}`);
  }
});

els.startRecord.addEventListener("click", () => {
  if (!stream) return;
  chunks = [];
  motionLog = [];
  orientationLog = [];
  recordedBlob = null;
  recordedMime = bestMimeType();
  recorder = new MediaRecorder(stream, recordedMime ? { mimeType: recordedMime } : undefined);
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.onstop = () => {
    recordedBlob = new Blob(chunks, { type: recordedMime || "video/webm" });
    els.downloadPack.disabled = false;
    els.uploadVideo.disabled = false;
    els.videoInfo.textContent = `${formatBytes(recordedBlob.size)} ${recordedBlob.type || "video"}`;
    setStatus("Recorded");
    log(`Recording ready: ${formatBytes(recordedBlob.size)}`);
  };
  recorder.start(1000);
  startedAt = Date.now();
  recordingPerfStart = performance.now();
  timerId = window.setInterval(updateTimer, 250);
  els.startRecord.disabled = true;
  els.stopRecord.disabled = false;
  setStatus("Recording");
  log(`Recording started as ${recordedMime || "browser default"}`);
});

els.stopRecord.addEventListener("click", () => {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  window.clearInterval(timerId);
  updateTimer();
  els.startRecord.disabled = false;
  els.stopRecord.disabled = true;
  log("Recording stopped");
});

els.downloadPack.addEventListener("click", async () => {
  if (!recordedBlob) return;
  const pack = await buildCapturePack();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(pack);
  a.download = `${slug(els.sessionName.value)}.capturepack`;
  a.click();
  URL.revokeObjectURL(a.href);
  log("CapturePack downloaded");
});

els.uploadVideo.addEventListener("click", async () => {
  if (!recordedBlob) return;
  try {
    setStatus("Uploading");
    const apiBase = els.apiBase.value.replace(/\/$/, "");
    let projectId = els.projectId.value.trim();
    if (!projectId) {
      const created = await jsonFetch(`${apiBase}/api/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: els.sessionName.value || "Phone Capture", target_type: els.targetType.value })
      });
      projectId = created.id;
      els.projectId.value = projectId;
      els.projectInfo.textContent = `${created.name} (${created.id})`;
    }
    const form = new FormData();
    form.append("file", recordedBlob, `phone_capture${videoExtension()}`);
    const result = await jsonFetch(`${apiBase}/api/projects/${projectId}/import/video`, { method: "POST", body: form });
    els.projectInfo.textContent = `${result.project.name} · ${result.project.status}`;
    setStatus("Uploaded");
    log("Video uploaded to desktop backend");
  } catch (error) {
    setStatus("Upload Error");
    log(`Upload error: ${error.message}`);
  }
});

function onMotion(event) {
  if (!recorder || recorder.state !== "recording") return;
  motionLog.push({
    // Milliseconds since recording started, so t=0 is video frame 0.
    t_ms: Math.round(performance.now() - recordingPerfStart),
    acceleration: event.acceleration,
    accelerationIncludingGravity: event.accelerationIncludingGravity,
    rotationRate: event.rotationRate,
    interval: event.interval
  });
  els.motionInfo.textContent = String(motionLog.length);
}

function onOrientation(event) {
  if (!recorder || recorder.state !== "recording") return;
  orientationLog.push({
    t_ms: Math.round(performance.now() - recordingPerfStart),
    alpha: event.alpha,
    beta: event.beta,
    gamma: event.gamma,
    absolute: event.absolute
  });
  els.orientationInfo.textContent = String(orientationLog.length);
}

function bestMimeType() {
  const candidates = [
    "video/mp4;codecs=h264",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function videoExtension() {
  return recordedMime.includes("mp4") ? ".mp4" : ".webm";
}

async function buildCapturePack() {
  const now = new Date().toISOString();
  const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`;
  const videoName = `video/main_video${videoExtension()}`;
  const manifest = {
    capturepack_version: "0.1",
    session_id: id,
    session_name: els.sessionName.value || "Phone Capture",
    capture_type: "static_scene",
    target_type: els.targetType.value,
    created_at: now,
    device: {
      manufacturer: navigator.vendor || "unknown",
      model: navigator.userAgent,
      os: navigator.platform || "unknown",
      app_version: "mobile-pwa-0.1"
    },
    video: {
      main_file: videoName,
      duration_sec: Math.round((Date.now() - startedAt) / 100) / 10,
      width: cameraSettings.width || null,
      height: cameraSettings.height || null,
      fps: cameraSettings.frameRate || null,
      codec: recordedMime || "browser_default",
      bitrate: null,
      has_audio: els.recordAudio.checked
    },
    capture_settings: {
      exposure_locked: null,
      white_balance_locked: null,
      focus_locked: null,
      storage_mode: "phone"
    },
    metadata_files: {
      intrinsics: "camera/intrinsics.json",
      poses: null,
      imu: "motion/imu_gyro.json",
      light: null,
      audio: null
    }
  };
  const intrinsics = {
    source: "browser_media_track_settings",
    camera_settings: cameraSettings,
    note: "Browser APIs do not expose calibrated intrinsics in this minimal app."
  };
  const imu = {
    source: "DeviceMotionEvent",
    // Declared explicitly so a consumer never has to guess which clock these
    // timestamps came from.
    time_base: "milliseconds_since_recording_start",
    recording_started_at: new Date(startedAt).toISOString(),
    sample_rate_note:
      "DeviceMotionEvent fires at 10-60 Hz depending on browser and device, and iOS requires " +
      "an explicit permission grant. Treat the rate as variable and read event.interval.",
    samples: motionLog,
    orientation_samples: orientationLog
  };
  const warnings = {
    warnings: [
      "Browser capture does not provide calibrated camera intrinsics.",
      "ARKit/ARCore camera poses are not captured in this minimal PWA.",
      "COLMAP will be required for reconstruction unless pose metadata is added later."
    ]
  };
  const payload = [
    ["manifest.json", jsonBlob(manifest)],
    [videoName, recordedBlob],
    ["camera/intrinsics.json", jsonBlob(intrinsics)],
    ["motion/imu_gyro.json", jsonBlob(imu)],
    ["quality/capture_warnings.json", jsonBlob(warnings)]
  ];

  const checksumEntries = {};
  for (const [name, blob] of payload) {
    checksumEntries[name] = await sha256(blob);
  }
  payload.push(["checksums/sha256.json", jsonBlob(checksumEntries)]);

  return new Blob([await zipStore(await buildBag(payload))], { type: "application/zip" });
}

// Wraps the payload as a BagIt bag (RFC 8493) so a pack written on a phone is
// verifiable by bagit-python, an institutional repository, or Zenodo -- none
// of which will ever learn a bespoke zip dialect. See docs/CAPTUREPACK_SPEC.md.
async function buildBag(payload) {
  const entries = [];
  const manifestLines = [];
  let octets = 0;

  for (const [name, blob] of payload) {
    const path = `data/${name}`;
    entries.push([path, blob]);
    manifestLines.push(`${await sha256(blob)}  ${path}`);
    octets += blob.size;
  }

  const bagit = new Blob(["BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"], {
    type: "text/plain"
  });
  const bagInfo = new Blob(
    [
      [
        "Bag-Software-Agent: GaussCapture mobile PWA (https://github.com/en970/gausscapture)",
        `Bagging-Date: ${new Date().toISOString().slice(0, 10)}`,
        // Payload-Oxum is BagIt's cheap integrity check: total octets and file
        // count, verifiable without hashing anything.
        `Payload-Oxum: ${octets}.${payload.length}`,
        `Internal-Sender-Identifier: ${els.sessionName.value || "Phone Capture"}`,
        "Internal-Sender-Description: GaussCapture CapturePack captured in a browser"
      ].join("\n") + "\n"
    ],
    { type: "text/plain" }
  );
  const payloadManifest = new Blob([manifestLines.join("\n") + "\n"], { type: "text/plain" });

  const tagFiles = [
    ["bagit.txt", bagit],
    ["bag-info.txt", bagInfo],
    ["manifest-sha256.txt", payloadManifest]
  ];
  const tagLines = [];
  for (const [name, blob] of tagFiles) {
    tagLines.push(`${await sha256(blob)}  ${name}`);
  }
  tagFiles.push(["tagmanifest-sha256.txt", new Blob([tagLines.join("\n") + "\n"])]);

  return [...tagFiles, ...entries];
}

function jsonBlob(value) {
  return new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
}

async function jsonFetch(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function slug(value) {
  return (value || "phone_capture").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "phone_capture";
}

async function sha256(blob) {
  const hash = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(hash)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function zipStore(files) {
  const encoder = new TextEncoder();
  const chunks = [];
  const central = [];
  let offset = 0;
  for (const [name, blob] of files) {
    const nameBytes = encoder.encode(name);
    const data = new Uint8Array(await blob.arrayBuffer());
    const crc = crc32(data);
    const local = new ArrayBuffer(30 + nameBytes.length);
    const l = new DataView(local);
    l.setUint32(0, 0x04034b50, true);
    l.setUint16(4, 20, true);
    l.setUint16(6, 0, true);
    l.setUint16(8, 0, true);
    l.setUint16(10, 0, true);
    l.setUint16(12, 0, true);
    l.setUint32(14, crc, true);
    l.setUint32(18, data.length, true);
    l.setUint32(22, data.length, true);
    l.setUint16(26, nameBytes.length, true);
    l.setUint16(28, 0, true);
    new Uint8Array(local, 30).set(nameBytes);
    chunks.push(local, data);

    const cent = new ArrayBuffer(46 + nameBytes.length);
    const c = new DataView(cent);
    c.setUint32(0, 0x02014b50, true);
    c.setUint16(4, 20, true);
    c.setUint16(6, 20, true);
    c.setUint16(8, 0, true);
    c.setUint16(10, 0, true);
    c.setUint16(12, 0, true);
    c.setUint16(14, 0, true);
    c.setUint32(16, crc, true);
    c.setUint32(20, data.length, true);
    c.setUint32(24, data.length, true);
    c.setUint16(28, nameBytes.length, true);
    c.setUint16(30, 0, true);
    c.setUint16(32, 0, true);
    c.setUint16(34, 0, true);
    c.setUint16(36, 0, true);
    c.setUint32(38, 0, true);
    c.setUint32(42, offset, true);
    new Uint8Array(cent, 46).set(nameBytes);
    central.push(cent);
    offset += local.byteLength + data.length;
  }
  const centralSize = central.reduce((sum, item) => sum + item.byteLength, 0);
  const centralOffset = offset;
  const end = new ArrayBuffer(22);
  const e = new DataView(end);
  e.setUint32(0, 0x06054b50, true);
  e.setUint16(8, files.length, true);
  e.setUint16(10, files.length, true);
  e.setUint32(12, centralSize, true);
  e.setUint32(16, centralOffset, true);
  e.setUint16(20, 0, true);
  return new Blob([...chunks, ...central, end]);
}

function crc32(data) {
  let crc = -1;
  for (let i = 0; i < data.length; i++) {
    crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ data[i]) & 0xff];
  }
  return (crc ^ -1) >>> 0;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();
