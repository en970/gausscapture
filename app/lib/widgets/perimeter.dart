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
///
/// The scripted protocol asks two more things of it, and both are answered without introducing a
/// second colour, because the whole point of the perimeter is that it works in a part of the visual
/// field that reads colour badly.
///
///   * **Rest.** The transition ends on a measurement, not a clock, and the operator has to be able
///     to tell whether the phone has been accepted as still without looking at it. That is
///     *luminance*: the calm white border brightens as rest accumulates. Brightness is not hue, so
///     it costs nothing from the amber budget.
///   * **Direction.** During the sweep, one edge glows to say which way still owes its share. It is
///     an edge rather than an arrow because an arrow has to be foveated to be read and an edge does
///     not, and because the answer is literally "over there".
class Perimeter extends StatefulWidget {
  const Perimeter({
    super.key,
    required this.active,
    required this.warning,
    this.rest = 0,
    this.direction = 0,
  });

  final bool active;
  final bool warning;

  /// How settled the phone is, 0 to 1. Zero for any phase that does not care.
  final double rest;

  /// Which way the sweep still needs to go: −1 left, +1 right, 0 for neither.
  final int direction;

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
          final Color colour;
          if (widget.warning) {
            colour = Palette.warn
                .withValues(alpha: reduceMotion ? 1.0 : 1.0 - 0.55 * _pulse.value);
          } else {
            // 0.18 is the calm alpha; a fully settled phone reaches 0.55, which is a clear change
            // in the periphery and still nowhere near competing with amber.
            final rest = widget.rest.clamp(0.0, 1.0);
            colour = Colors.white.withValues(alpha: 0.18 + 0.37 * rest);
          }
          return Stack(
            fit: StackFit.expand,
            children: [
              Container(
                decoration: BoxDecoration(border: Border.all(color: colour, width: 8)),
              ),
              if (widget.direction != 0 && !widget.warning)
                Align(
                  alignment:
                      widget.direction < 0 ? Alignment.centerLeft : Alignment.centerRight,
                  child: Container(width: 8, height: 200, color: Colors.white),
                ),
            ],
          );
        },
      ),
    );
  }
}
