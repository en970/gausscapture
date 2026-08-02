import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../capture_engine.dart';
import '../design.dart';
import '../widgets/coach.dart';
import '../widgets/perimeter.dart';
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
/// Silence is the success state. There is no "good" message, because a signal that is always
/// present is one the operator learns to stop seeing.
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

  CaptureStatus _status = const CaptureStatus();
  Telemetry _telemetry = const Telemetry();
  StreamSubscription<Telemetry>? _telemetrySubscription;
  StreamSubscription<String>? _problemSubscription;
  StreamSubscription<CaptureStatus>? _statusSubscription;

  String? _coach;
  DateTime? _coachShownAt;
  int _pacingTicked = 0;
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
  }

  @override
  void dispose() {
    _telemetrySubscription?.cancel();
    _statusSubscription?.cancel();
    _problemSubscription?.cancel();
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
      _setCoach(null);
      return;
    }
    _updateCoach(telemetry);
    _updatePacing(telemetry);
  }

  /// A strict single-slot priority queue: one message, ever.
  ///
  /// Only signals that are geometry rather than guesswork appear here. The composed quality score
  /// this project is working towards is explicitly unvalidated, and putting an uncalibrated number
  /// on the viewfinder would both mislead the operator and contaminate the study meant to
  /// validate it.
  void _updateCoach(Telemetry telemetry) {
    if (!_status.canWarnAboutMotion) return;

    final showing = _coach != null;
    final threshold = showing ? _clearPixels : _warnPixels;

    if (telemetry.blurPixels > threshold) {
      _setCoach('Slow down');
    } else {
      _setCoach(null);
    }
  }

  void _setCoach(String? message) {
    if (_coach == message) return;

    // A warning that has just appeared stays up long enough to be read, even if the condition
    // clears immediately.
    if (message == null && _coachShownAt != null) {
      final visible = DateTime.now().difference(_coachShownAt!);
      if (visible < Motion.coachMinimumVisible) return;
    }

    setState(() {
      _coach = message;
      _coachShownAt = message == null ? null : DateTime.now();
    });

    if (message != null) {
      // One buzz on entering the state, never while it persists. Repetition is how an alert
      // teaches the person to stop noticing it.
      HapticFeedback.mediumImpact();
    }
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
    final padding = MediaQuery.paddingOf(context);

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
            warning: _coach != null,
          ),

          if (!recording) _brief(padding),
          if (recording) _letterChip(padding),
          if (!recording) _readiness(padding),

          Align(
            alignment: const Alignment(0, -0.2),
            child: Coach(message: _coach),
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
            ],
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

  /// Whether this take will be worth anything, said before it is recorded rather than discovered
  /// on a Mac a week later.
  Widget _readiness(EdgeInsets padding) {
    final ready = _status.clockAligned;
    return Positioned(
      top: padding.top + 96,
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
    final target = _status.targetSeconds;
    final seconds = _telemetry.elapsedMs ~/ 1000;
    final progress = target > 0 ? (_telemetry.elapsedMs / (target * 1000)).clamp(0.0, 1.0) : 0.0;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (recording) ...[
          // Elapsed and target are not equals. Rendering them as one string of the same weight
          // forces reading two numbers at half the size needed while walking.
          Text(_formatTime(seconds), style: Fonts.counter),
          Text('of ${_formatTime(target)}',
              style: Fonts.body.copyWith(color: Palette.secondary)),
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
        '${_telemetry.frames} frames · ${_telemetry.blurPixels.toStringAsFixed(1)} px blur',
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
