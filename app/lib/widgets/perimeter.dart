import 'package:flutter/material.dart';

import '../design.dart';

/// A border around the whole preview, and the highest-value signal in the app.
///
/// At arm's length with the operator's gaze on the subject, the display sits in peripheral vision.
/// Peripheral vision is sensitive to motion and luminance and poor at detail and colour, which is
/// exactly the wrong profile for reading a word and exactly the right one for noticing a bright
/// edge start pulsing. A centred message requires foveating; this does not.
///
/// Calm while recording is presence rather than alarm — hue stays reserved for the state that
/// needs acting on.
class Perimeter extends StatefulWidget {
  const Perimeter({super.key, required this.active, required this.warning});

  final bool active;
  final bool warning;

  @override
  State<Perimeter> createState() => _PerimeterState();
}

class _PerimeterState extends State<Perimeter> with SingleTickerProviderStateMixin {
  late final AnimationController _pulse = AnimationController(
    vsync: this,
    duration: Motion.pulse,
  );

  @override
  void didUpdateWidget(Perimeter old) {
    super.didUpdateWidget(old);
    if (widget.warning && !_pulse.isAnimating) {
      _pulse.repeat(reverse: true);
    } else if (!widget.warning) {
      _pulse.stop();
      _pulse.value = 0;
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.active) return const SizedBox.shrink();

    // Honour the platform's reduced-motion setting: the border still reports the state, it just
    // stops moving.
    final reduceMotion = MediaQuery.disableAnimationsOf(context);

    return IgnorePointer(
      child: AnimatedBuilder(
        animation: _pulse,
        builder: (context, _) {
          final colour = widget.warning
              ? Palette.warn.withValues(
                  alpha: reduceMotion ? 1.0 : 1.0 - 0.55 * _pulse.value)
              : Palette.perimeterCalm;
          return Container(
            decoration: BoxDecoration(
              border: Border.all(color: colour, width: 8),
            ),
          );
        },
      ),
    );
  }
}
