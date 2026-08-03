import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../capture_engine.dart';
import '../design.dart';

/// The library, and the answer to a question a modal list could not carry.
///
/// Recording all day produces dozens of takes, and the only two things worth knowing about each
/// are whether it is scientifically usable and whether it has already been copied off. Neither fits
/// in a dialog of plain names, which is what this replaces — and a dialog also puts delete one tap
/// away from the take you were looking at.
///
/// Newest first. The previous version sorted by name ascending, which buried the take just recorded
/// at the bottom of the list.
class TakesScreen extends StatefulWidget {
  const TakesScreen({super.key, required this.engine});

  final CaptureEngine engine;

  @override
  State<TakesScreen> createState() => _TakesScreenState();
}

class _TakesScreenState extends State<TakesScreen> {
  List<Take> _takes = const [];
  CaptureStatus _status = const CaptureStatus();
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final takes = await widget.engine.takes();
    final status = await widget.engine.status();
    if (!mounted) return;
    setState(() {
      _takes = takes;
      _status = status;
      _loading = false;
    });
  }

  int get _bytes => _takes.fold(0, (sum, t) => sum + t.bytes);
  int get _pending => _takes.where((t) => !t.offloaded).length;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Palette.base,
      appBar: AppBar(
        backgroundColor: Palette.base,
        surfaceTintColor: Colors.transparent,
        title: const Text('Takes', style: Fonts.title),
        actions: [
          IconButton(
            icon: const Icon(Icons.ios_share),
            tooltip: 'How to copy these off',
            onPressed: _showTransfer,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Palette.muted))
          : _takes.isEmpty
              ? _empty()
              : RefreshIndicator(
                  onRefresh: _load,
                  backgroundColor: Palette.surface,
                  color: Palette.primary,
                  child: ListView.separated(
                    padding: const EdgeInsets.only(bottom: Space.xxl),
                    itemCount: _takes.length + 1,
                    separatorBuilder: (_, _) =>
                        const Divider(height: 1, color: Palette.hairline, indent: Space.gutter),
                    itemBuilder: (context, index) =>
                        index == 0 ? _summary() : _row(_takes[index - 1]),
                  ),
                ),
    );
  }

  Widget _summary() => Padding(
        padding: const EdgeInsets.fromLTRB(Space.gutter, Space.sm, Space.gutter, Space.lg),
        child: Text(
          '${_takes.length} takes · ${_formatBytes(_bytes)}'
          '${_pending > 0 ? " · $_pending not copied" : ""}',
          style: Fonts.body,
        ),
      );

  Widget _empty() => Center(
        child: Padding(
          padding: const EdgeInsets.all(Space.xxl),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.videocam_outlined, size: 48, color: Palette.disabled),
              const SizedBox(height: Space.gutter),
              const Text('No takes yet', style: Fonts.title),
              const SizedBox(height: Space.sm),
              Text(
                'Pick a shot and press the red button. Start with A.',
                style: Fonts.body,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: Space.xl),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(),
                style: FilledButton.styleFrom(
                  minimumSize: const Size(200, Touch.pill),
                  backgroundColor: Palette.surface,
                ),
                child: const Text('Go to capture', style: Fonts.bodyLarge),
              ),
            ],
          ),
        ),
      );

  Widget _row(Take take) {
    final health = _health(take);
    return InkWell(
      onTap: () => _showDetail(take),
      child: Container(
        constraints: const BoxConstraints(minHeight: Touch.row),
        padding: const EdgeInsets.symmetric(horizontal: Space.gutter, vertical: Space.md),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: Palette.surface,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                take.presetId.isEmpty ? '?' : take.presetId[0],
                style: Fonts.bodyLarge.copyWith(fontWeight: FontWeight.w700),
              ),
            ),
            const SizedBox(width: Space.lg),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(take.name, style: Fonts.bodyLarge, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 2),
                  Text(
                    '${take.seconds.toStringAsFixed(0)}s · ${take.frames} frames · '
                    '${take.imuSamples} imu · ${_formatBytes(take.bytes)}',
                    style: Fonts.body.copyWith(color: Palette.muted),
                  ),
                  if (health != null) ...[
                    const SizedBox(height: Space.xs),
                    Row(children: [
                      const Icon(Icons.error_outline, size: 14, color: Palette.warn),
                      const SizedBox(width: Space.xs),
                      Text(health, style: Fonts.body.copyWith(color: Palette.warn, fontSize: 13)),
                    ]),
                  ],
                ],
              ),
            ),
            const SizedBox(width: Space.md),
            // Filled means the desktop has it; hollow means this take exists only on the phone.
            Icon(
              take.offloaded ? Icons.circle : Icons.circle_outlined,
              size: 12,
              color: take.offloaded ? Palette.ok : Palette.disabled,
            ),
          ],
        ),
      ),
    );
  }

  /// The one thing worth flagging in a list: whether this take can be used at all.
  String? _health(Take take) {
    if (take.incomplete) return 'interrupted — no manifest';
    if (!take.clockAligned) return 'clock mismatch — up not recoverable';
    if (take.droppedImu > 0) return '${take.droppedImu} imu samples lost';
    if (take.frames == 0) return 'no frames recorded';
    return null;
  }

  Future<void> _showDetail(Take take) async {
    await showModalBottomSheet<void>(
      context: context,
      backgroundColor: Palette.raised,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (context) => _TakeDetail(
        take: take,
        onDelete: () async {
          Navigator.of(context).pop();
          await _confirmDelete(take);
        },
      ),
    );
  }

  Future<void> _confirmDelete(Take take) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Palette.raised,
        title: const Text('Delete this take?', style: Fonts.title),
        content: Text(
          take.offloaded
              ? '${take.name} has been copied to a computer.'
              : '${take.name} has not been copied anywhere yet. '
                  'This is the only copy.',
          style: Fonts.body,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Keep'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete', style: TextStyle(color: Palette.warn)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await widget.engine.deleteTake(take.name);
      await _load();
    }
  }

  void _showTransfer() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: Palette.raised,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(Space.gutter),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Getting takes onto a computer', style: Fonts.title),
              const SizedBox(height: Space.md),
              Text(
                _status.storageVisible
                    ? 'Plug the phone in and open this folder. It appears in Finder '
                        'and in any file manager.'
                    : 'Takes are in the app’s private folder, which macOS cannot browse. '
                        'Either grant file access so they move to Documents, or copy them '
                        'with the command below.',
                style: Fonts.body,
              ),
              const SizedBox(height: Space.lg),
              _copyable(_status.storagePath),
              const SizedBox(height: Space.sm),
              _copyable('gausscapture pull'),
              if (!_status.storageVisible) ...[
                const SizedBox(height: Space.lg),
                FilledButton(
                  onPressed: () async {
                    await widget.engine.requestSharedStorage();
                    if (context.mounted) Navigator.of(context).pop();
                  },
                  style: FilledButton.styleFrom(
                    minimumSize: const Size(double.infinity, Touch.pill),
                    backgroundColor: Palette.surface,
                  ),
                  child: const Text('Grant file access', style: Fonts.bodyLarge),
                ),
              ],
              const SizedBox(height: Space.lg),
            ],
          ),
        ),
      ),
    );
  }

  Widget _copyable(String text) => InkWell(
        onLongPress: () {
          Clipboard.setData(ClipboardData(text: text));
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Copied'), backgroundColor: Palette.surface),
          );
        },
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(Space.md),
          decoration: BoxDecoration(
            color: Palette.base,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(text, style: Fonts.mono.copyWith(color: Palette.secondary)),
        ),
      );

  static String _formatBytes(int bytes) {
    if (bytes >= 1 << 30) return '${(bytes / (1 << 30)).toStringAsFixed(1)} GB';
    if (bytes >= 1 << 20) return '${bytes >> 20} MB';
    return '${bytes >> 10} KB';
  }
}

/// What a researcher actually wants to know about a take, in the order they want to know it.
class _TakeDetail extends StatelessWidget {
  const _TakeDetail({required this.take, required this.onDelete});

  final Take take;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(Space.gutter),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(take.name, style: Fonts.title),
            const SizedBox(height: Space.xl),

            // The validation card comes first because it is the only question that decides
            // whether the rest of the numbers mean anything.
            _check('Camera and IMU clock', take.clockAligned,
                take.clockAligned ? take.sensorClock : 'not aligned'),
            _check('Frames recorded', take.frames > 0, '${take.frames}'),
            _check('IMU samples', take.imuSamples > 0, '${take.imuSamples}'),
            _check('No samples lost', take.droppedImu == 0,
                take.droppedImu == 0 ? 'none' : '${take.droppedImu} dropped'),
            _check('Video anchored', take.firstFrameNs != null,
                take.firstFrameNs != null ? 'first frame stamped' : 'ordinal only'),
            _check(
              'Motion blur bounded',
              (take.blurAtNormalPace ?? 99) < 2.0,
              take.blurAtNormalPace == null
                  ? take.exposurePolicy
                  : '${take.blurAtNormalPace!.toStringAsFixed(1)} px at a normal pace',
            ),
            _check('Copied to a computer', take.offloaded,
                take.offloaded ? 'yes' : 'not yet'),

            const SizedBox(height: Space.xl),
            Text(take.path, style: Fonts.mono),
            const SizedBox(height: Space.xl),
            OutlinedButton(
              onPressed: onDelete,
              style: OutlinedButton.styleFrom(
                minimumSize: const Size(double.infinity, Touch.pill),
                side: const BorderSide(color: Palette.warn),
              ),
              child: const Text('Delete take', style: TextStyle(color: Palette.warn, fontSize: 17)),
            ),
          ],
        ),
      ),
    );
  }

  /// Glyph and word together, never colour alone.
  Widget _check(String label, bool ok, String value) => Padding(
        padding: const EdgeInsets.only(bottom: Space.md),
        child: Row(
          children: [
            Icon(ok ? Icons.check_circle : Icons.error_outline,
                size: 18, color: ok ? Palette.ok : Palette.warn),
            const SizedBox(width: Space.md),
            Expanded(child: Text(label, style: Fonts.bodyLarge)),
            Text(value, style: Fonts.body.copyWith(color: ok ? Palette.secondary : Palette.warn)),
          ],
        ),
      );
}
