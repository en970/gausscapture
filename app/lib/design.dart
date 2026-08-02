import 'package:flutter/material.dart';

/// The visual system, derived from the task rather than from a style guide.
///
/// Two constraints shaped every value here, and both come from where the phone actually is during
/// a capture: held out at roughly arm's length, in daylight, with the operator walking and looking
/// at the subject rather than at the screen.
///
/// The first is angular size. At 550 mm a 16sp word subtends about 0.05 degrees of x-height, which
/// is roughly a third of what is readable in motion. That single number is why the guidance word
/// is 44sp and not "a bit bigger".
///
/// The second is that contrast over a live camera image cannot be guaranteed. Frosted glass and
/// translucent scrims are the obvious modern choice and are wrong here, because the background is
/// whatever the operator happens to be pointing at. Everything critical sits on an opaque shape.
///
/// Hue is reserved for state: amber means act now, red means recording and nothing else, green
/// means verified — and green never appears over the viewfinder. Progress bars, selection and
/// every other affordance are neutral. That is what keeps red-against-amber from having to be
/// distinguished at the one moment it would matter.
class Palette {
  const Palette._();

  // Surfaces
  static const base = Color(0xFF0B0F14);
  static const surface = Color(0xFF141A21);
  static const raised = Color(0xFF1B222B);
  static const hairline = Color(0xFF2A333D);

  // Text, with contrast against `base`
  static const primary = Color(0xFFFFFFFF); // 19.3:1 — anything read in motion
  static const secondary = Color(0xFFC3CDD8); // 11.9:1
  static const muted = Color(0xFF8A97A5); //  6.5:1 — never in motion
  static const disabled = Color(0xFF56606B);

  // State
  static const warn = Color(0xFFFFB020); //  10.5:1, the only warning colour
  static const onWarn = Color(0xFF0B0F14);
  static const ok = Color(0xFF30D158); // static screens only
  static const record = Color(0xFFFF3B30); // recording, nothing else

  /// Recording and healthy. Presence, not alarm.
  static const perimeterCalm = Color(0x2EFFFFFF);

  /// Over the preview: opaque enough that an unknown background cannot defeat it.
  static const scrim = Color(0xEB060A0F);
}

/// Type scale, in logical pixels, chosen from angular size at arm's length.
class Fonts {
  const Fonts._();

  /// The one guidance word. Three times the size of what it replaced.
  static const glance = TextStyle(
    fontSize: 44,
    fontWeight: FontWeight.w900,
    height: 1.15,
    color: Palette.onWarn,
    letterSpacing: -0.5,
  );

  /// Elapsed time and any large count. Tabular so digits do not jitter as they change.
  static const counter = TextStyle(
    fontSize: 32,
    fontWeight: FontWeight.w700,
    height: 1.15,
    fontFamily: 'monospace',
    fontFamilyFallback: ['RobotoMono', 'monospace'],
    color: Palette.primary,
  );

  static const title = TextStyle(
    fontSize: 22,
    fontWeight: FontWeight.w700,
    height: 1.3,
    color: Palette.primary,
  );

  static const bodyLarge = TextStyle(fontSize: 17, height: 1.3, color: Palette.primary);
  static const body = TextStyle(fontSize: 15, height: 1.45, color: Palette.secondary);

  static const label = TextStyle(
    fontSize: 13,
    fontWeight: FontWeight.w600,
    height: 1.45,
    letterSpacing: 0.6,
    color: Palette.muted,
  );

  /// Diagnostics and paths. Never smaller: these are read standing still, but they are still read.
  static const mono = TextStyle(
    fontSize: 12,
    height: 1.4,
    fontFamily: 'monospace',
    fontFamilyFallback: ['RobotoMono', 'monospace'],
    color: Palette.muted,
  );
}

/// A 4-point grid. Gaps between adjacent targets are 12 rather than the usual 8, because a finger
/// arriving on a moving phone is less precise than one arriving on a phone sitting still.
class Space {
  const Space._();

  static const xs = 4.0;
  static const sm = 8.0;
  static const md = 12.0;
  static const lg = 16.0;
  static const gutter = 20.0;
  static const xl = 24.0;
  static const xxl = 32.0;
}

/// Touch targets. The floor is 56 rather than 48: 48 is the standing-still minimum.
class Touch {
  const Touch._();

  static const record = 84.0;
  static const pill = 56.0;
  static const row = 72.0;
  static const sheetRow = 88.0;
}

/// Durations. Nothing exceeds 300 ms, and at most two things animate at once.
class Motion {
  const Motion._();

  static const appear = Duration(milliseconds: 120);
  static const vanish = Duration(milliseconds: 180);
  static const press = Duration(milliseconds: 90);
  static const morph = Duration(milliseconds: 160);
  static const sheet = Duration(milliseconds: 220);
  static const pulse = Duration(milliseconds: 1200);

  /// The real motion design of this app is hysteresis, not easing. A warning that appears and
  /// clears at the same threshold flickers at the boundary, which trains the operator to ignore it.
  static const coachMinimumVisible = Duration(milliseconds: 900);
}

ThemeData buildTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: Palette.base,
    colorScheme: base.colorScheme.copyWith(
      surface: Palette.surface,
      primary: Palette.primary,
      error: Palette.warn,
    ),
    dividerColor: Palette.hairline,
    splashFactory: InkSparkle.splashFactory,
  );
}
