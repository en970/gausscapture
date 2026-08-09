import 'package:flutter/material.dart';

import '../design.dart';

/// Where the take is in its script, as a strip of segments.
///
/// A scripted take is the one situation in this app where the operator has to know something the
/// perimeter cannot say: not "is this going well" but "which of the four things am I doing, and how
/// much of it is left". The strip answers both without any text — filled segments are done, the
/// current one fills as it runs, and the ones ahead are empty.
///
/// It is neutral white and grey on purpose. Progress is not a state that needs acting on, and the
/// moment progress borrows amber is the moment amber stops meaning "act now".
class PhaseRail extends StatelessWidget {
  const PhaseRail({
    super.key,
    required this.index,
    required this.count,
    required this.progress,
  });

  /// Zero-based index of the phase running now.
  final int index;
  final int count;

  /// How far through the current phase, 0 to 1. A phase that ends on a measurement rather than a
  /// clock will sit at 1 until the measurement arrives, which is honest: the clock really is spent.
  final double progress;

  @override
  Widget build(BuildContext context) {
    if (count <= 1) return const SizedBox.shrink();
    return Row(
      children: [
        for (var i = 0; i < count; i++) ...[
          if (i > 0) const SizedBox(width: Space.xs),
          Expanded(
            child: SizedBox(
              height: 4,
              child: Stack(
                children: [
                  Container(color: Colors.white.withValues(alpha: 0.2)),
                  FractionallySizedBox(
                    widthFactor: i < index
                        ? 1.0
                        : i == index
                            ? progress.clamp(0.0, 1.0)
                            : 0.0,
                    child: Container(color: Colors.white.withValues(alpha: 0.9)),
                  ),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }
}

/// The seconds between letting go of the phone and the subject being asked to move.
///
/// It exists as its own large number rather than as words inside the coach pill because it is the
/// one moment in the take where the operator's hands must be nowhere near the phone and their
/// attention must be on their own face. A countdown that has to be read cannot do that job; a digit
/// that changes once a second can be caught at the edge of vision.
///
/// It restarts whenever the phone is touched. A countdown that survives being touched is a
/// countdown that means nothing, and the phase after it gives every one of its frames the same
/// camera pose.
class Countdown extends StatelessWidget {
  const Countdown({super.key, required this.millis});

  /// Milliseconds remaining, or a negative number when nothing is counting down.
  final int millis;

  @override
  Widget build(BuildContext context) {
    if (millis < 0) return const SizedBox.shrink();
    final seconds = (millis / 1000).ceil().clamp(0, 99);
    return Container(
      width: 96,
      height: 96,
      alignment: Alignment.center,
      decoration: const BoxDecoration(color: Palette.scrim, shape: BoxShape.circle),
      child: Text(
        '$seconds',
        style: Fonts.glance.copyWith(color: Palette.primary, fontSize: 56),
      ),
    );
  }
}
