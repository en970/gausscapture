import 'package:flutter/material.dart';

import '../capture_engine.dart';
import '../design.dart';

/// Choosing which protocol this take follows.
///
/// Two of the five shots are *meant* to fail, and they are labelled as such in amber rather than
/// hidden. That is the scientific point: a model that predicts capture quality has to be trained
/// on captures that went wrong, so recording bad ones deliberately is part of the protocol and
/// should read as intentional rather than as a mistake about to be made.
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
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
    ),
    builder: (context) => SafeArea(
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
                    preset.expectedToFail ? 'expected to fail' : preset.hint,
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
