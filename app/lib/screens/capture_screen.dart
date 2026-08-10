import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../capture_engine.dart';
import '../design.dart';
import '../widgets/coach.dart';
import '../widgets/perimeter.dart';
import '../widgets/phase_rail.dart';
import '../widgets/record_button.dart';
import '../widgets/shot_picker.dart';
import 'takes_screen.dart';

/// The only screen used while moving.
///
/// Its whole design follows from one fact: the operator is walking, holding the phone out, and
/// looking at the subject rather than at the display. So the layers are ordered by how little
/// attention each demands — haptics first, then a border readable in peripheral vision, then one
/// large word, and only then anything that needs a deliberate glance.
///
/// Silence is the success state for an unscripted take. There is no "good" message, because a
/// signal that is always present is one the operator learns to stop seeing.
///
/// A scripted take is the deliberate exception, and it is worth naming why. Preset F is four phases
/// inside one recording — hold still, sweep around, put the phone back, now move — and the operator
/// is also the subject, facing their own front camera with their hands busy. There is no moment to
/// consult anything. So during a scripted take the guidance slot always holds *something*: a
/// warning when one applies, and otherwise the current phase's cue. That is not a persistent
/// quality signal of the kind this app refuses to show; it is an instruction that changes every few
/// seconds and is the entire content of the take.
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key, required this.engine});

  final CaptureEngine engine;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  /// Warn above this many pixels of predicted smear, clear below the other.
  ///
  /// Two thresholds rather than one, because a warning that appears and clears at the same value
  /// flickers along the boundary, and a flickering warning teaches the operator to ignore it.
  static const _warnPixels = 2.0;
  static const _clearPixels = 1.4;

  /// Continuous rest the phase machine requires before it will accept the phone as still. Mirrored
  /// here only to scale the perimeter's brightness; the decision itself is made on the other side
  /// of the channel, from the sensors.
  static const _restConfirmMs = 700;

  CaptureStatus _status = const CaptureStatus();
  Telemetry _telemetry = const Telemetry();
  StreamSubscription<Telemetry>? _telemetrySubscription;
  StreamSubscription<String>? _problemSubscription;
  StreamSubscription<CaptureStatus>? _statusSubscription;
  StreamSubscription<FinishedTake>? _finishedSubscription;

  String? _coach;
  IconData _coachIcon = Icons.speed;
  DateTime? _coachShownAt;
  int _pacingTicked = 0;
  int _transitionsFelt = 0;
  int _countdownFelt = -1;
  bool _busy = false;
  bool _diagnostics = false;

  @override
  void initState() {
    super.initState();
    _refresh();
    _telemetrySubscription = widget.engine.telemetry.listen(_onTelemetry);
    _statusSubscription = widget.engine.statusChanges.listen((s) {
      if (mounted) setState(() => _status = s);
    });
    _problemSubscription = widget.engine.problems.listen(_onProblem);
    _finishedSubscription = widget.engine.finished.listen(_onFinished);
  }

  @override
  void dispose() {
    _telemetrySubscription?.cancel();
    _statusSubscription?.cancel();
    _problemSubscription?.cancel();
    _finishedSubscription?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    final status = await widget.engine.status();
    if (mounted) setState(() => _status = status);
  }

  // --------------------------------------------------------------- guidance

  void _onTelemetry(Telemetry telemetry) {
    if (!mounted) return;
    setState(() => _telemetry = telemetry);
    if (!telemetry.recording) {
      _setCoach(null, Icons.speed, immediate: true);
      return;
    }
    _updateCoach(telemetry);
    _updateHaptics(telemetry);
  }

  /// A strict single-slot priority queue: one message, ever.
  ///
  /// Only signals that are geometry or measurement appear here. The composed quality score this
  /// project is working towards is explicitly unvalidated, and putting an uncalibrated number on
  /// the viewfinder would both mislead the operator and contaminate the study meant to validate it.
  ///
  /// The order is not arbitrary. A phase warning outranks the blur warning because it describes
  /// something that invalidates the take rather than degrading it — a phone that never settled
  /// makes the fixed-camera claim false, while smear makes it worse. The cue is last because it is
  /// what to do when nothing is wrong.
  void _updateCoach(Telemetry telemetry) {
    final warning = telemetry.phaseWarning;
    if (warning != null && warning.isNotEmpty) {
      _setCoach(warning, _warningGlyph(warning), immediate: true);
      return;
    }

    // Smear only matters while the phone is being carried. During the phases that hold it still it
    // is zero by construction, and warning about it there would be noise at the moment the operator
    // most needs the screen quiet.
    final handHeld = !_status.scripted || telemetry.phaseId == 'arc';
    if (handHeld && _status.canWarnAboutMotion) {
      final showing = _coach == 'Slow down';
      final threshold = showing ? _clearPixels : _warnPixels;
      if (telemetry.blurPixels > threshold) {
        _setCoach('Slow down', Icons.speed);
        return;
      }
    }

    if (_status.scripted && telemetry.phaseCue.isNotEmpty) {
      _setCoach(telemetry.phaseCue, _phaseGlyph(telemetry.phaseId), immediate: true);
      return;
    }
    _setCoach(null, Icons.speed);
  }

  /// [immediate] skips the minimum-dwell rule.
  ///
  /// The dwell exists so a warning that clears the instant it appears is still readable. A phase
  /// cue is the opposite case: it must change the moment the phase does, because the operator is
  /// about to be asked to do something different and a stale instruction is worse than none.
  void _setCoach(String? message, IconData icon, {bool immediate = false}) {
    if (_coach == message) return;

    if (!immediate && message == null && _coachShownAt != null) {
      final visible = DateTime.now().difference(_coachShownAt!);
      if (visible < Motion.coachMinimumVisible) return;
    }

    final wasWarning = _coach != null && _coach != _telemetry.phaseCue;
    setState(() {
      _coach = message;
      _coachIcon = icon;
      _coachShownAt = message == null ? null : DateTime.now();
    });

    // One buzz on entering a warning, never while it persists, and never for a cue — the phase
    // transition has already buzzed, and buzzing twice for one event teaches the operator that the
    // buzz means nothing.
    if (message != null && message != _telemetry.phaseCue && !wasWarning) {
      HapticFeedback.mediumImpact();
    }
  }

  /// Everything that can be felt rather than seen, which is the layer that costs no attention.
  void _updateHaptics(Telemetry telemetry) {
    if (_status.scripted) {
      // One buzz per phase change, driven by the engine's own counter rather than by timing the
      // phases here — a transition happens when the sensors say so, not when a stopwatch does.
      while (_transitionsFelt < telemetry.phaseTransitions) {
        _transitionsFelt++;
        HapticFeedback.heavyImpact();
      }
      // A tick a second through the hands-off countdown, so the operator can keep their eyes on
      // their own face and still know when to start moving.
      if (telemetry.countdownMs >= 0) {
        final second = (telemetry.countdownMs / 1000).ceil();
        if (second != _countdownFelt) {
          _countdownFelt = second;
          HapticFeedback.selectionClick();
        }
      } else {
        _countdownFelt = -1;
      }
      return;
    }
    _updatePacing(telemetry);
  }

  /// Ticks at each quarter of the target, so "am I done yet" needs no glance at all.
  void _updatePacing(Telemetry telemetry) {
    final target = _status.targetSeconds * 1000;
    if (target <= 0) return;
    final quarter = (telemetry.elapsedMs * 4 ~/ target).clamp(0, 4);
    while (_pacingTicked < quarter) {
      _pacingTicked++;
      if (_pacingTicked == 4) {
        HapticFeedback.heavyImpact();
      } else {
        HapticFeedback.selectionClick();
      }
    }
  }

  static IconData _phaseGlyph(String phaseId) {
    switch (phaseId) {
      case 'perch':
        return Icons.center_focus_strong;
      case 'arc':
        return Icons.threesixty;
      case 'reseat':
        return Icons.smartphone;
      case 'hold':
        return Icons.record_voice_over;
      default:
        return Icons.videocam;
    }
  }

  static IconData _warningGlyph(String warning) {
    switch (warning) {
      case 'Hands off':
        return Icons.pan_tool;
      case 'Move around':
      case 'Walk around it':
        return Icons.directions_walk;
      case 'Sweep wider':
        return Icons.open_in_full;
      default:
        return Icons.priority_high;
    }
  }

  // ---------------------------------------------------------------- actions

  Future<void> _toggleRecording() async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      if (_status.recording) {
        final manifest = await widget.engine.stop();
        HapticFeedback.mediumImpact();
        _announce(manifest == null
            ? 'Nothing was recorded'
            : 'Saved ${manifest['session_name']} · '
                '${(manifest['video']?['frames_recorded'] ?? 0)} frames');
      } else {
        _pacingTicked = 0;
        _transitionsFelt = 0;
        _countdownFelt = -1;
        await widget.engine.start();
        HapticFeedback.heavyImpact();
      }
      await _refresh();
    } on PlatformException catch (error) {
      _announce(error.message ?? 'Could not start');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// A scripted take that ended by itself. Nothing here asked for it, which is the point.
  void _onFinished(FinishedTake take) {
    if (!mounted) return;
    _setCoach(null, Icons.speed, immediate: true);
    HapticFeedback.heavyImpact();
    if (take.aborted) {
      // Not a crash and not a deleted file: the footage and the manifest are on disk, and the
      // manifest says what was measured and why the script gave up on it.
      _announce('Take stopped: ${take.reason}');
    } else {
      final manifest = take.manifest;
      _announce(manifest == null
          ? 'Take finished'
          : 'Saved ${manifest['session_name']} · '
              '${(manifest['video']?['frames_recorded'] ?? 0)} frames');
    }
    _refresh();
  }

  void _onProblem(String message) {
    _announce(message);
    _refresh();
  }

  void _announce(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(
        content: Text(message, style: Fonts.bodyLarge),
        backgroundColor: Palette.raised,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 4),
      ));
  }

  Future<void> _pickShot() async {
    final chosen = await showShotPicker(context, widget.engine, _status.preset);
    if (chosen != null) {
      final status = await widget.engine.setPreset(chosen);
      if (mounted) setState(() => _status = status);
    }
  }

  // ------------------------------------------------------------------ build

  @override
  Widget build(BuildContext context) {
    final recording = _status.recording;
    final scripted = _status.scripted;
    final padding = MediaQuery.paddingOf(context);
    final showingWarning = _coach != null && _coach != _telemetry.phaseCue;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          const _Preview(),

          // Peripheral vision reads motion and luminance well and detail badly, so the border is
          // the signal that works when the screen is not being looked at.
          Perimeter(
            active: recording,
            warning: showingWarning,
            rest: _restConfidence(),
            direction: recording && scripted && _telemetry.phaseId == 'arc'
                ? _telemetry.arcDirection
                : 0,
          ),

          if (!recording) _brief(padding),
          if (recording) _letterChip(padding),
          if (!recording) _rotateButton(padding),
          if (recording && scripted) _rail(padding),
          if (!recording) _readiness(padding),

          Align(
            alignment: const Alignment(0, -0.2),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: Space.gutter),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Coach(message: _coach, icon: _coachIcon),
                  if (_showPhaseDetail) ...[
                    const SizedBox(height: Space.md),
                    _phaseDetailLine(),
                  ],
                ],
              ),
            ),
          ),

          if (recording && _telemetry.countdownMs >= 0)
            Align(
              alignment: const Alignment(0, 0.25),
              child: Countdown(millis: _telemetry.countdownMs),
            ),

          Positioned(
            left: 0,
            right: 0,
            bottom: padding.bottom + Space.gutter,
            child: _controls(recording),
          ),

          if (_diagnostics)
            Positioned(
              left: Space.gutter,
              bottom: padding.bottom + 160,
              child: _diagnosticsLine(),
            ),
        ],
      ),
    );
  }

  /// How settled the phone is, for the perimeter, and zero wherever the question is meaningless.
  ///
  /// Only the phases that keep the phone in its support ask it. During the sweep the phone is
  /// supposed to be moving, and a border that dimmed as the operator did the right thing would be
  /// reporting failure at the one moment everything was going well.
  ///
  /// [Telemetry.atRest] gates it rather than merely informing it. `restForMs` is an accumulated
  /// counter, and between a nudge and the next telemetry tick it still holds the count from before
  /// — so a phone that has just been knocked reads as settled for one frame. The border should
  /// drop the instant the phone moves; that is the only thing it is for.
  double _restConfidence() {
    if (!_status.recording || !_status.scripted) return 0;
    const stillPhases = {'perch', 'reseat', 'hold'};
    if (!stillPhases.contains(_telemetry.phaseId)) return 0;
    if (!_telemetry.atRest) return 0;
    return (_telemetry.restForMs / _restConfirmMs).clamp(0.0, 1.0);
  }

  /// The supporting sentence for this phase, shown while the instruction is still news.
  ///
  /// `Preset.java` documents each detail string as "read once at the start of the phase", and the
  /// engine sends it on every tick — but nothing rendered it, so the four sentences the preset
  /// authors wrote ("Put the phone back exactly where it was, and let go.") never reached anyone.
  /// The operator saw only the one-word cue in the Coach pill.
  bool get _showPhaseDetail =>
      _status.recording &&
      _status.scripted &&
      _telemetry.phaseDetail.isNotEmpty &&
      _telemetry.phaseElapsedMs < _phaseDetailMs;

  /// How long the supporting line stays. Long enough to read while holding a phone, short enough
  /// that it is gone before it becomes something to look past.
  static const int _phaseDetailMs = 2500;

  Widget _phaseDetailLine() => Container(
        padding: const EdgeInsets.symmetric(horizontal: Space.lg, vertical: Space.sm),
        decoration: BoxDecoration(
          color: Palette.scrim,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          _telemetry.phaseDetail,
          textAlign: TextAlign.center,
          style: Fonts.body.copyWith(color: Palette.primary),
        ),
      );

  Widget _brief(EdgeInsets padding) => Positioned(
        top: 0,
        left: 0,
        right: 0,
        child: Container(
          padding: EdgeInsets.fromLTRB(
              Space.gutter, padding.top + Space.lg, Space.gutter, Space.lg),
          color: Palette.scrim,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(_status.presetTitle, style: Fonts.title),
              const SizedBox(height: Space.xs),
              Text(_status.presetHint, style: Fonts.bodyLarge.copyWith(color: Palette.secondary)),
              if (_status.scripted) ...[
                const SizedBox(height: Space.md),
                Text(
                  '${_status.phaseIds.length} phases in one recording · '
                  '${_status.phaseIds.join(" → ")} · stops by itself',
                  style: Fonts.label,
                ),
              ],
            ],
          ),
        ),
      );

  /// Turns the preview a quarter at a time.
  ///
  /// The correct turn depends on how the platform composites the preview surface, and the app
  /// cannot read that: every derived angle was tried on a device and each one left the image on
  /// its side. So the operator sets it, once, and it is remembered across launches. Display only
  /// -- the recording, its orientation hint and the intrinsics are untouched. Hidden during a
  /// take, because nothing that changes what is on screen belongs within reach while recording.
  Widget _rotateButton(EdgeInsets padding) => Positioned(
        top: padding.top + Space.md,
        right: Space.gutter,
        child: GestureDetector(
          onTap: () async {
            await widget.engine.nudgePreviewTurn();
            HapticFeedback.selectionClick();
          },
          child: Container(
            width: Touch.pill,
            height: Touch.pill,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: Palette.scrim,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.screen_rotation_outlined, size: 22, color: Palette.secondary),
          ),
        ),
      );

  /// During a take the brief collapses: it has been read, and a block of text at the top would
  /// compete with the warning channel for the whole recording.
  Widget _letterChip(EdgeInsets padding) => Positioned(
        top: padding.top + Space.md,
        left: Space.gutter,
        child: Container(
          width: 40,
          height: 40,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: Palette.scrim,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            _status.preset.isEmpty ? '?' : _status.preset[0],
            style: Fonts.bodyLarge.copyWith(fontWeight: FontWeight.w700, fontSize: 20),
          ),
        ),
      );

  /// Four segments, so "which part am I in" costs a glance rather than a reading.
  Widget _rail(EdgeInsets padding) => Positioned(
        top: padding.top + Space.md + 48,
        left: Space.gutter,
        right: Space.gutter,
        child: PhaseRail(
          index: _telemetry.phaseIndex,
          count: _telemetry.phaseCount,
          progress: _telemetry.phaseProgress,
        ),
      );

  /// Whether this take will be worth anything, said before it is recorded rather than discovered
  /// on a Mac a week later.
  Widget _readiness(EdgeInsets padding) {
    final ready = _status.clockAligned;
    return Positioned(
      top: padding.top + 96 + (_status.scripted ? 28 : 0),
      right: Space.gutter,
      child: GestureDetector(
        onTap: () => setState(() => _diagnostics = !_diagnostics),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: Space.md, vertical: Space.sm),
          decoration: BoxDecoration(
            color: Palette.scrim,
            borderRadius: BorderRadius.circular(999),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                ready ? Icons.check_circle : Icons.error_outline,
                size: 16,
                color: ready ? Palette.ok : Palette.warn,
              ),
              const SizedBox(width: Space.sm),
              Text(
                ready ? 'Ready' : 'Check',
                style: Fonts.label.copyWith(color: ready ? Palette.secondary : Palette.warn),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _controls(bool recording) {
    final scripted = _status.scripted;
    final target = _status.targetSeconds;
    final seconds = _telemetry.elapsedMs ~/ 1000;

    // A scripted take's ring follows the phase, not the whole recording. Two of its four phases
    // end on a measurement rather than a clock, so a ring drawn against the total would be a
    // prediction rather than a report.
    final progress = scripted
        ? _telemetry.phaseProgress
        : target > 0
            ? (_telemetry.elapsedMs / (target * 1000)).clamp(0.0, 1.0)
            : 0.0;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (recording) ...[
          // Elapsed and target are not equals. Rendering them as one string of the same weight
          // forces reading two numbers at half the size needed while walking.
          Text(_formatTime(seconds), style: Fonts.counter),
          Text(
            scripted
                ? '${_telemetry.phaseId} · ${_telemetry.phaseIndex + 1} of '
                    '${_telemetry.phaseCount}'
                : 'of ${_formatTime(target)}',
            style: Fonts.body.copyWith(color: Palette.secondary),
          ),
          // Progress towards the sweep target, during the one phase where it is
          // actionable. The perimeter shows which direction still owes its share
          // but not how far the sweep has got, and the number lived only in the
          // diagnostics line — whose toggle is not rendered while recording, so
          // during a take it was unreachable.
          if (scripted && _telemetry.phaseId == 'arc' && _status.arcTargetDeg > 0)
            Text(
              '${_telemetry.arcSweptDeg.toStringAsFixed(0)}° of ${_status.arcTargetDeg}°',
              style: Fonts.body.copyWith(color: Palette.secondary),
            ),
          const SizedBox(height: Space.lg),
        ],
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(
              width: 110,
              child: recording
                  ? const SizedBox.shrink()
                  : _pill('Takes', Icons.folder_outlined, _openTakes),
            ),
            RecordButton(
              recording: recording,
              progress: progress,
              enabled: !_busy && _status.cameraReady,
              onPressed: _toggleRecording,
            ),
            SizedBox(
              width: 110,
              child: recording
                  ? const SizedBox.shrink()
                  : _pill('Shot ${_status.preset.isEmpty ? "" : _status.preset[0]}',
                      Icons.tune, _pickShot),
            ),
          ],
        ),
      ],
    );
  }

  Widget _pill(String label, IconData icon, VoidCallback onTap) => Material(
        color: Palette.scrim,
        borderRadius: BorderRadius.circular(Touch.pill / 2),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(Touch.pill / 2),
          child: Container(
            height: Touch.pill,
            alignment: Alignment.center,
            padding: const EdgeInsets.symmetric(horizontal: Space.lg),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 18, color: Palette.secondary),
                const SizedBox(width: Space.sm),
                Flexible(
                  child: Text(label,
                      style: Fonts.body.copyWith(color: Palette.primary),
                      overflow: TextOverflow.ellipsis),
                ),
              ],
            ),
          ),
        ),
      );

  Widget _diagnosticsLine() => Text(
        '${_status.focalPixels.toStringAsFixed(0)} px · '
        '${_status.sensorClock} · ${_telemetry.imuRateHz} imu/s · '
        '${_telemetry.frames} frames · ${_telemetry.blurPixels.toStringAsFixed(1)} px blur'
        '${_status.scripted ? ' · ${_telemetry.arcSweptDeg.toStringAsFixed(0)}° swept'
            ' of ${_status.arcTargetDeg}°'
            '${_telemetry.restDisturbed ? ' · rest lost' : ''}' : ''}',
        style: Fonts.mono.copyWith(color: Palette.muted.withValues(alpha: 0.7)),
      );

  Future<void> _openTakes() async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => TakesScreen(engine: widget.engine),
    ));
    if (mounted) _refresh();
  }

  static String _formatTime(int seconds) =>
      '${seconds ~/ 60}:${(seconds % 60).toString().padLeft(2, '0')}';
}

/// The camera preview: a platform view, because the transform that makes it upright and correctly
/// proportioned is written against a real TextureView and is the most error-prone code in the app.
class _Preview extends StatelessWidget {
  const _Preview();

  @override
  Widget build(BuildContext context) => const AndroidView(
        viewType: CaptureEngine.previewViewType,
        creationParamsCodec: StandardMessageCodec(),
      );
}
