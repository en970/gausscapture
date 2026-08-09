import 'dart:async';

import 'package:flutter/services.dart';

/// The Dart side of the capture engine.
///
/// Recording lives in Java, and not for historical reasons. Flutter's camera plugin does not expose
/// a per-frame `SENSOR_TIMESTAMP`, and its sensor plugins do not expose `SensorEvent.timestamp` at
/// all. Those two values, on one hardware clock, are what let a frame's camera pose be paired with
/// the accelerometer reading taken at that instant — which is how the world's up direction is
/// recovered, at 0.998 agreement across a capture, against heuristics that were wrong by as much as
/// 41 degrees. A pure-Dart rewrite would look better and record data that could not be used.
///
/// So this is a thin, honest wrapper: commands out, telemetry in, and no logic that belongs on the
/// other side of the boundary.
class CaptureEngine {
  CaptureEngine() {
    _commands.setMethodCallHandler(_onNativeCall);
  }

  static const _commands = MethodChannel('gausscapture/commands');
  static const _telemetry = EventChannel('gausscapture/telemetry');

  /// The platform view id the preview widget instantiates.
  static const previewViewType = 'gausscapture/preview';

  final _problems = StreamController<String>.broadcast();
  final _status = StreamController<CaptureStatus>.broadcast();
  final _finished = StreamController<FinishedTake>.broadcast();

  /// Failures the operator did not cause and cannot see: a full card, the camera taken by another
  /// app, an encoder fault. These used to reach nobody.
  Stream<String> get problems => _problems.stream;

  /// Emitted whenever the engine's own state changes underneath the interface.
  Stream<CaptureStatus> get statusChanges => _status.stream;

  /// A scripted take that ended by itself.
  ///
  /// Not a problem and not a stop the operator asked for. The dynamic phase auto-stops precisely so
  /// that nobody reaches for the phone — reaching for it moves it, and the frames that would
  /// corrupt are the last ones, the ones closest to the motion the take exists to record. Which
  /// means nothing on this side knows the take is over unless it is told.
  Stream<FinishedTake> get finished => _finished.stream;

  /// Live capture signals, at roughly 10 Hz.
  Stream<Telemetry> get telemetry =>
      _telemetry.receiveBroadcastStream().map((e) => Telemetry.fromMap(Map<String, dynamic>.from(e as Map)));

  Future<void> _onNativeCall(MethodCall call) async {
    switch (call.method) {
      case 'problem':
        _problems.add(call.arguments as String);
        break;
      case 'stateChanged':
        _status.add(CaptureStatus.fromMap(Map<String, dynamic>.from(call.arguments as Map)));
        break;
      case 'finished':
        final map = Map<String, dynamic>.from(call.arguments as Map);
        if (map['status'] != null) {
          _status.add(CaptureStatus.fromMap(Map<String, dynamic>.from(map['status'] as Map)));
        }
        _finished.add(FinishedTake.fromMap(map));
        break;
    }
  }

  Future<CaptureStatus> status() async {
    final map = await _commands.invokeMapMethod<String, dynamic>('status');
    return CaptureStatus.fromMap(map ?? const {});
  }

  /// Begins a take. Throws with a readable reason rather than returning false, because every
  /// failure here has a specific cause the operator can act on.
  Future<void> start() => _commands.invokeMethod<void>('start');

  /// Ends a take and returns its manifest, or null when nothing usable was produced.
  Future<Map<String, dynamic>?> stop() async {
    final result = await _commands.invokeMapMethod<String, dynamic>('stop');
    return result;
  }

  Future<CaptureStatus> setPreset(String id) async {
    final map = await _commands.invokeMapMethod<String, dynamic>('setPreset', {'id': id});
    return CaptureStatus.fromMap(map ?? const {});
  }

  Future<List<Preset>> presets() async {
    final rows = await _commands.invokeListMethod<Object?>('presets') ?? const [];
    return rows.map((r) => Preset.fromMap(Map<String, dynamic>.from(r! as Map))).toList();
  }

  Future<List<Take>> takes() async {
    final rows = await _commands.invokeListMethod<Object?>('takes') ?? const [];
    return rows.map((r) => Take.fromMap(Map<String, dynamic>.from(r! as Map))).toList();
  }

  Future<bool> deleteTake(String name) async =>
      await _commands.invokeMethod<bool>('deleteTake', {'name': name}) ?? false;

  /// Opens the system screen for all-files access. Returns true when it was already granted.
  Future<bool> requestSharedStorage() async =>
      await _commands.invokeMethod<bool>('requestSharedStorage') ?? false;
}

/// A capture protocol. Two of the five are meant to fail: a predictor trained only on successes
/// has never seen the thing it is supposed to warn about.
class Preset {
  const Preset({
    required this.id,
    required this.title,
    required this.hint,
    required this.targetSeconds,
    required this.expectedToFail,
    this.scripted = false,
    this.captureType = 'static_scene',
    this.phaseCount = 1,
  });

  factory Preset.fromMap(Map<String, dynamic> map) => Preset(
        id: map['id'] as String? ?? '',
        title: map['title'] as String? ?? '',
        hint: map['hint'] as String? ?? '',
        targetSeconds: map['targetSeconds'] as int? ?? 60,
        expectedToFail: map['expectedToFail'] as bool? ?? false,
        scripted: map['scripted'] as bool? ?? false,
        captureType: map['captureType'] as String? ?? 'static_scene',
        phaseCount: map['phaseCount'] as int? ?? 1,
      );

  final String id;
  final String title;
  final String hint;
  final int targetSeconds;
  final bool expectedToFail;

  /// Whether this take is several phases inside one recording rather than one undivided stretch.
  ///
  /// One file, not several, because the phase that carries the geometry and the phase that carries
  /// the motion have to share an exposure, a white balance and a clock — and because stopping the
  /// recorder to start it again moves the phone, which is the one thing the fixed-camera half of
  /// the take assumes did not happen.
  final bool scripted;

  /// What the manifest declares, and therefore which pipeline the desktop dispatches to.
  final String captureType;
  final int phaseCount;

  /// The letter an operator actually refers to a shot by.
  String get letter => id.isEmpty ? '?' : id[0];
}

/// A take that ended without anyone pressing stop.
class FinishedTake {
  const FinishedTake({required this.reason, required this.aborted, this.manifest});

  factory FinishedTake.fromMap(Map<String, dynamic> map) => FinishedTake(
        reason: map['reason'] as String? ?? 'complete',
        aborted: map['aborted'] as bool? ?? false,
        manifest: map['manifest'] == null
            ? null
            : Map<String, dynamic>.from(map['manifest'] as Map),
      );

  /// `complete`, or a sentence saying what went wrong.
  final String reason;

  /// True when the script gave up rather than finished. The footage is still on disk and the
  /// manifest still says what was measured — a refused take is a record, not a deleted file.
  final bool aborted;
  final Map<String, dynamic>? manifest;

  bool get completed => !aborted;
}

/// What the engine is and where it will write, in one snapshot.
class CaptureStatus {
  const CaptureStatus({
    this.recording = false,
    this.preset = '',
    this.presetTitle = '',
    this.presetHint = '',
    this.targetSeconds = 60,
    this.expectedToFail = false,
    this.storagePath = '',
    this.storageVisible = false,
    this.minutesRemaining = -1,
    this.clockAligned = false,
    this.clockDescription = '',
    this.cameraRealtime = false,
    this.sensorClock = 'UNKNOWN',
    this.focalPixels = 0,
    this.cameraReady = false,
    this.imuRateHz = 0,
    this.scripted = false,
    this.phaseIds = const [],
    this.arcTargetDeg = 0,
  });

  factory CaptureStatus.fromMap(Map<String, dynamic> map) => CaptureStatus(
        recording: map['recording'] as bool? ?? false,
        preset: map['preset'] as String? ?? '',
        presetTitle: map['presetTitle'] as String? ?? '',
        presetHint: map['presetHint'] as String? ?? '',
        targetSeconds: map['targetSeconds'] as int? ?? 60,
        expectedToFail: map['expectedToFail'] as bool? ?? false,
        storagePath: map['storagePath'] as String? ?? '',
        storageVisible: map['storageVisible'] as bool? ?? false,
        minutesRemaining: map['minutesRemaining'] as int? ?? -1,
        clockAligned: map['clockAligned'] as bool? ?? false,
        clockDescription: map['clockDescription'] as String? ?? '',
        cameraRealtime: map['cameraRealtime'] as bool? ?? false,
        sensorClock: map['sensorClock'] as String? ?? 'UNKNOWN',
        focalPixels: (map['focalPixels'] as num?)?.toDouble() ?? 0,
        cameraReady: map['cameraReady'] as bool? ?? false,
        imuRateHz: map['imuRateHz'] as int? ?? 0,
        scripted: map['scripted'] as bool? ?? false,
        phaseIds: (map['phaseIds'] as List?)?.map((e) => e as String).toList() ?? const [],
        arcTargetDeg: map['arcTargetDeg'] as int? ?? 0,
      );

  final bool recording;
  final String preset;
  final String presetTitle;
  final String presetHint;
  final int targetSeconds;
  final bool expectedToFail;

  final String storagePath;

  /// False means captures are somewhere macOS cannot see, and `gausscapture pull` is the only
  /// way to get them off. Worth saying out loud before a day of recording, not after.
  final bool storageVisible;
  final int minutesRemaining;

  /// The check that decides whether a take is scientifically worth anything.
  final bool clockAligned;
  final String clockDescription;
  final bool cameraRealtime;
  final String sensorClock;

  final double focalPixels;
  final bool cameraReady;
  final int imuRateHz;

  /// True when the selected preset is a script rather than one stretch of recording.
  final bool scripted;

  /// The phases in order, known before the take starts so the rail can be drawn empty rather than
  /// growing a segment at a time as each phase is entered.
  final List<String> phaseIds;

  /// Degrees the sweep is aiming for, or zero for a preset with no arc.
  final int arcTargetDeg;

  /// Motion warnings need a focal length to convert angular rate into pixels of smear.
  bool get canWarnAboutMotion => focalPixels > 0;
}

/// One sample of what is happening right now.
class Telemetry {
  const Telemetry({
    this.recording = false,
    this.elapsedMs = 0,
    this.blurPixels = 0,
    this.angularRate = 0,
    this.imuSamples = 0,
    this.imuRateHz = 0,
    this.frames = 0,
    this.exposureNs = 0,
    this.phaseId = '',
    this.phaseIndex = 0,
    this.phaseCount = 1,
    this.phaseCue = '',
    this.phaseDetail = '',
    this.phaseWarning,
    this.phaseElapsedMs = 0,
    this.phaseTargetMs = 0,
    this.phaseTransitions = 0,
    this.countdownMs = -1,
    this.atRest = false,
    this.restForMs = 0,
    this.restDisturbed = false,
    this.arcSweptDeg = 0,
    this.arcDirection = 0,
  });

  factory Telemetry.fromMap(Map<String, dynamic> map) => Telemetry(
        recording: map['recording'] as bool? ?? false,
        elapsedMs: (map['elapsedMs'] as num?)?.toInt() ?? 0,
        blurPixels: (map['blurPixels'] as num?)?.toDouble() ?? 0,
        angularRate: (map['angularRate'] as num?)?.toDouble() ?? 0,
        imuSamples: (map['imuSamples'] as num?)?.toInt() ?? 0,
        imuRateHz: (map['imuRateHz'] as num?)?.toInt() ?? 0,
        frames: (map['frames'] as num?)?.toInt() ?? 0,
        exposureNs: (map['exposureNs'] as num?)?.toInt() ?? 0,
        phaseId: map['phaseId'] as String? ?? '',
        phaseIndex: (map['phaseIndex'] as num?)?.toInt() ?? 0,
        phaseCount: (map['phaseCount'] as num?)?.toInt() ?? 1,
        phaseCue: map['phaseCue'] as String? ?? '',
        phaseDetail: map['phaseDetail'] as String? ?? '',
        phaseWarning: map['phaseWarning'] as String?,
        phaseElapsedMs: (map['phaseElapsedMs'] as num?)?.toInt() ?? 0,
        phaseTargetMs: (map['phaseTargetMs'] as num?)?.toInt() ?? 0,
        phaseTransitions: (map['phaseTransitions'] as num?)?.toInt() ?? 0,
        countdownMs: (map['countdownMs'] as num?)?.toInt() ?? -1,
        atRest: map['atRest'] as bool? ?? false,
        restForMs: (map['restForMs'] as num?)?.toInt() ?? 0,
        restDisturbed: map['restDisturbed'] as bool? ?? false,
        arcSweptDeg: (map['arcSweptDeg'] as num?)?.toDouble() ?? 0,
        arcDirection: (map['arcDirection'] as num?)?.toInt() ?? 0,
      );

  final bool recording;
  final int elapsedMs;

  /// Predicted smear in pixels: angular rate times exposure times focal length. Geometry, not a
  /// fitted score — which is why it is honest to act on while this project's own quality model is
  /// still unvalidated.
  final double blurPixels;
  final double angularRate;
  final int imuSamples;
  final int imuRateHz;
  final int frames;
  final int exposureNs;

  // --- the script -----------------------------------------------------------
  //
  // Measured on the other side of the channel rather than inferred here. A phase boundary is the
  // moment the phone was measured to have stopped moving, not a point on a stopwatch, and an
  // interface that timed the phases itself would be wrong exactly at the transitions — which are
  // the only moments it exists to signal.

  /// `perch`, `arc`, `reseat`, `hold`, or `free` for an unscripted take.
  final String phaseId;
  final int phaseIndex;
  final int phaseCount;

  /// The one word for this phase. Neutral: an instruction, never a fault.
  final String phaseCue;
  final String phaseDetail;

  /// The one thing that is wrong, or null. Never populated during the dynamic phase — the subject
  /// is being asked to look natural, and there is nothing useful they could do about the phone now.
  final String? phaseWarning;

  final int phaseElapsedMs;
  final int phaseTargetMs;

  /// Counts phase changes, so the interface can buzz exactly once on each without timing them.
  final int phaseTransitions;

  /// Milliseconds until the dynamic phase begins, or −1 when nothing is counting down.
  final int countdownMs;

  /// Whether the phone is measurably still right now — gyroscope and accelerometer together, not
  /// "the operator put it down".
  final bool atRest;
  final int restForMs;

  /// Latched: rest was confirmed and then lost. The take can no longer claim a fixed camera.
  final bool restDisturbed;

  final double arcSweptDeg;

  /// Which way the sweep still needs to go: −1 left, +1 right, 0 when both sides have their share.
  final int arcDirection;

  double get phaseProgress =>
      phaseTargetMs <= 0 ? 0 : (phaseElapsedMs / phaseTargetMs).clamp(0.0, 1.0);
}

/// A take on disk.
class Take {
  const Take({
    required this.name,
    required this.path,
    required this.bytes,
    required this.offloaded,
    required this.incomplete,
    this.manifest,
  });

  factory Take.fromMap(Map<String, dynamic> map) => Take(
        name: map['name'] as String? ?? '',
        path: map['path'] as String? ?? '',
        bytes: (map['bytes'] as num?)?.toInt() ?? 0,
        offloaded: map['offloaded'] as bool? ?? false,
        incomplete: map['incomplete'] as bool? ?? false,
        manifest: map['manifest'] == null
            ? null
            : Map<String, dynamic>.from(map['manifest'] as Map),
      );

  final String name;
  final String path;
  final int bytes;

  /// True once the desktop side has recorded that it copied this take. Recording all day produces
  /// dozens of takes, and without this there is no way to tell which are already safe.
  final bool offloaded;

  /// Has a directory but no manifest: interrupted by process death, a flat battery or a camera
  /// error. Shown rather than hidden, so it can be deleted rather than silently holding storage.
  final bool incomplete;
  final Map<String, dynamic>? manifest;

  String get presetId => manifest?['preset'] as String? ?? name.split('_').first;
  int get frames => (manifest?['video']?['frames_recorded'] as num?)?.toInt() ?? 0;
  int get imuSamples => (manifest?['imu_samples'] as num?)?.toInt() ?? 0;
  int get droppedImu =>
      (manifest?['stream_health']?['imu_lines_dropped'] as num?)?.toInt() ?? 0;
  bool get clockAligned => manifest?['clocks']?['camera_imu_same_clock'] as bool? ?? false;
  String get sensorClock => manifest?['clocks']?['sensor_clock'] as String? ?? 'UNKNOWN';
  int? get firstFrameNs =>
      (manifest?['clocks']?['first_encoded_frame_t_ns'] as num?)?.toInt();

  /// How exposure was decided. `manual_capped` means blur was bounded; `auto_locked` means the
  /// device offered no manual control and the smear is whatever the light dictated.
  String get exposurePolicy =>
      manifest?['capture_settings']?['exposure']?['policy'] as String? ?? 'unknown';

  /// Predicted smear at a normal orbit rate, in pixels. The number the whole exposure policy
  /// exists to hold down.
  double? get blurAtNormalPace =>
      (manifest?['capture_settings']?['exposure']?['blur_px_at_0p35_rad_s'] as num?)?.toDouble();

  int? get exposureNs =>
      (manifest?['capture_settings']?['exposure']?['exposure_ns'] as num?)?.toInt();

  double get seconds {
    final fps = (manifest?['video']?['fps'] as num?)?.toDouble() ?? 30;
    return fps > 0 ? frames / fps : 0;
  }

  /// The single question worth asking of a take before it leaves the phone.
  bool get usable => !incomplete && clockAligned && frames > 0;
}
