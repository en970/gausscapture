import 'package:flutter/material.dart';

import '../design.dart';

/// The one word.
///
/// Three rules, each of which the previous version broke. It is 44sp, because at arm's length
/// while walking the old 16sp subtended about a third of the angular size needed to read it. It
/// sits on an *opaque* shape, because contrast against a live camera image cannot be guaranteed
/// when the background is whatever the operator is pointing at — frosted glass is the fashionable
/// choice and the wrong one here. And it carries a glyph as well as a colour, so the state is
/// never encoded in hue alone.
///
/// Two words maximum, imperative, no units and no jargon. "Slow down", not "ω 1.8 rad/s". The
/// physics is computed and the verb is spoken.
///
/// It carries two kinds of message, and deliberately in the same slot and the same colour. A
/// warning ("Hands off") and a scripted cue ("Sweep around") are both *act now*, and hue is
/// reserved for exactly that — so what separates them is the glyph and the word, never the colour,
/// which is also what keeps them legible to an operator who cannot distinguish them. One slot
/// rather than two because two amber shapes competing for the same glance is how both get ignored.
class Coach extends StatelessWidget {
  const Coach({super.key, required this.message, this.icon = Icons.speed});

  final String? message;

  /// The glyph beside the word. State is never encoded in hue alone.
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    // Honour the platform's reduced-motion setting: the message still appears, it just does not
    // slide or fade its way in.
    final instant = MediaQuery.disableAnimationsOf(context);

    return AnimatedSwitcher(
      duration: instant ? Duration.zero : Motion.appear,
      reverseDuration: instant ? Duration.zero : Motion.vanish,
      // Deliberately not a cross-fade: two messages briefly overlapping is unreadable, so one
      // leaves before the next arrives.
      switchInCurve: Curves.decelerate,
      switchOutCurve: Curves.easeIn,
      child: message == null
          ? const SizedBox.shrink(key: ValueKey('none'))
          : Container(
              key: ValueKey(message),
              padding: const EdgeInsets.symmetric(horizontal: Space.xl, vertical: Space.lg),
              decoration: BoxDecoration(
                color: Palette.warn,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, size: 40, color: Palette.onWarn),
                  const SizedBox(width: Space.md),
                  Flexible(
                    child: Text(message!, style: Fonts.glance, textAlign: TextAlign.center),
                  ),
                ],
              ),
            ),
    );
  }
}
