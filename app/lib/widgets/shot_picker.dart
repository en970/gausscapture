import 'package:flutter/material.dart';

import '../capture_engine.dart';
import '../design.dart';

/// Choosing which protocol this take follows.
///
/// Some of the shots are *meant* to fail, and they are labelled as such in amber rather than
/// hidden. That is the scientific point: a model that predicts capture quality has to be trained
/// on captures that went wrong, so recording bad ones deliberately is part of the protocol and
/// should read as intentional rather than as a mistake about to be made.
///
/// The list scrolls, and that is not a detail. `showModalBottomSheet` caps an unconstrained child
/// at half the screen height, and a `Column` past its bounds overflows rather than scrolling — so
/// once a sixth protocol was added, the sixth row sat below the fold with no way to reach it. The
/// sixth is `F_bullet`, the fixed-camera 4D capture, which meant the whole 4D path shipped and
/// could never be selected on the phone. A picker must be able to show every option it has.
Future<String?> showShotPicker(
  BuildContext context,
  CaptureEngine engine,
  String current,
) async {
  final presets = await engine.presets();
  if (!context.mounted) return null;

  return showModalBottomSheet<String>(
    context: context,
    backgroundColor: Palette.raised,
    isScrollControlled: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
    ),
    builder: (context) => SafeArea(
      child: ConstrainedBox(
        // Tall enough for every protocol, short enough that the sheet still
        // reads as a sheet rather than a screen.
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.85,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: Space.md),
            Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: Palette.hairline,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: Space.gutter),
            // The handle stays put; only the protocols move. Dragging a list
            // that carries its own grab handle away feels like the sheet is
            // closing.
            Flexible(
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    for (final preset in presets)
                      _ShotRow(
                        preset: preset,
                        selected: preset.id == current,
                        onTap: () => Navigator.of(context).pop(preset.id),
                      ),
                    const SizedBox(height: Space.sm),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    ),
  );
}

class _ShotRow extends StatelessWidget {
  const _ShotRow({required this.preset, required this.selected, required this.onTap});

  final Preset preset;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        height: Touch.sheetRow,
        padding: const EdgeInsets.symmetric(horizontal: Space.gutter),
        color: selected ? Colors.white.withValues(alpha: 0.06) : null,
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: selected ? Palette.primary : Palette.surface,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                preset.letter,
                style: Fonts.bodyLarge.copyWith(
                  fontWeight: FontWeight.w700,
                  color: selected ? Palette.base : Palette.secondary,
                ),
              ),
            ),
            const SizedBox(width: Space.lg),
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(preset.title, style: Fonts.bodyLarge),
                  const SizedBox(height: 2),
                  Text(
                    preset.expectedToFail
                        ? 'expected to fail'
                        // A scripted take behaves differently enough to say so before it is
                        // chosen: it runs itself, and it stops itself.
                        : preset.scripted
                            ? '${preset.phaseCount} phases · stops by itself'
                            : preset.hint,
                    style: Fonts.body.copyWith(
                      color: preset.expectedToFail ? Palette.warn : Palette.muted,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            const SizedBox(width: Space.md),
            Text('${preset.targetSeconds}s', style: Fonts.mono.copyWith(color: Palette.secondary)),
          ],
        ),
      ),
    );
  }
}
