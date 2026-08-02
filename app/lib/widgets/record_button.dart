import 'package:flutter/material.dart';

import '../design.dart';

/// Start and stop, with the progress ring around it.
///
/// 84 logical pixels, which is well above any minimum, because during a take the priority inverts:
/// starting no longer matters and stopping has to be trivial while walking with the phone held
/// out. The ring stays neutral even at 100% — completion is carried by a haptic tick, so it does
/// not need a colour that would compete with the warning channel.
class RecordButton extends StatelessWidget {
  const RecordButton({
    super.key,
    required this.recording,
    required this.progress,
    required this.enabled,
    required this.onPressed,
  });

  final bool recording;
  final double progress;
  final bool enabled;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: recording ? 'Stop recording' : 'Start recording',
      child: SizedBox(
        width: Touch.record + 12,
        height: Touch.record + 12,
        child: Stack(
          alignment: Alignment.center,
          children: [
            if (recording)
              SizedBox(
                width: Touch.record + 12,
                height: Touch.record + 12,
                child: CircularProgressIndicator(
                  value: progress,
                  strokeWidth: 6,
                  backgroundColor: const Color(0x33FFFFFF),
                  valueColor: const AlwaysStoppedAnimation(Color(0xE6FFFFFF)),
                ),
              ),
            GestureDetector(
              onTap: enabled ? onPressed : null,
              child: AnimatedContainer(
                duration: Motion.morph,
                curve: Curves.decelerate,
                width: Touch.record,
                height: Touch.record,
                decoration: BoxDecoration(
                  color: enabled ? Palette.record : Palette.disabled,
                  borderRadius: BorderRadius.circular(recording ? 20 : Touch.record / 2),
                  border: Border.all(color: Colors.white.withValues(alpha: 0.9), width: 4),
                ),
                child: recording
                    ? const Icon(Icons.stop, color: Colors.white, size: 34)
                    : null,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
