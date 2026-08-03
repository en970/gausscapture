import 'package:flutter_test/flutter_test.dart';

import 'package:gausscapture/capture_engine.dart';

/// The mapping between what the engine reports and what the interface believes.
///
/// These are the only pure functions in the app worth testing without a device: everything else
/// needs a camera. What they protect is the judgement a take gets in the library — a capture whose
/// clocks disagree is scientifically worthless, and mislabelling one as fine is the single most
/// expensive mistake this app can make.
void main() {
  group('Take', () {
    Take build(Map<String, dynamic> manifest, {bool incomplete = false}) => Take(
          name: 'A_good_20260803_120000',
          path: '/sdcard/Documents/GaussCapture/A_good_20260803_120000',
          bytes: 180 << 20,
          offloaded: false,
          incomplete: incomplete,
          manifest: manifest,
        );

    final healthy = {
      'preset': 'A_good',
      'video': {'frames_recorded': 1800, 'fps': 30},
      'imu_samples': 24000,
      'clocks': {'camera_imu_same_clock': true, 'sensor_clock': 'BOOTTIME',
                 'first_encoded_frame_t_ns': 2150390835896080},
      'stream_health': {'imu_lines_dropped': 0},
    };

    test('reads what the manifest reports', () {
      final take = build(healthy);
      expect(take.frames, 1800);
      expect(take.imuSamples, 24000);
      expect(take.seconds, 60);
      expect(take.clockAligned, isTrue);
      expect(take.firstFrameNs, isNotNull);
      expect(take.usable, isTrue);
    });

    test('a clock mismatch makes a take unusable', () {
      final take = build({...healthy, 'clocks': {'camera_imu_same_clock': false}});
      expect(take.usable, isFalse);
    });

    test('an interrupted take is unusable even with a manifest', () {
      expect(build(healthy, incomplete: true).usable, isFalse);
    });

    test('a take with no frames is unusable', () {
      final take = build({...healthy, 'video': {'frames_recorded': 0, 'fps': 30}});
      expect(take.usable, isFalse);
    });

    test('missing fields degrade rather than throw', () {
      const take = Take(
        name: 'B_normal_20260803_130000',
        path: '/tmp/x',
        bytes: 0,
        offloaded: false,
        incomplete: true,
      );
      expect(take.frames, 0);
      expect(take.presetId, 'B');
      expect(take.clockAligned, isFalse);
      expect(take.usable, isFalse);
    });
  });

  group('Exposure policy', () {
    Take withExposure(Map<String, dynamic> exposure) => Take(
          name: 'A_good_20260804_090000',
          path: '/tmp/x',
          bytes: 0,
          offloaded: false,
          incomplete: false,
          manifest: {
            'capture_settings': {'exposure': exposure},
          },
        );

    test('reports the smear the chosen exposure admits', () {
      // 1/120 s at 1350 px and 0.35 rad/s is about 1.6 px -- under the warning threshold, which
      // is the whole point of capping.
      final take = withExposure({
        'policy': 'manual_capped',
        'exposure_ns': 8333333,
        'blur_px_at_0p35_rad_s': 1.575,
      });
      expect(take.exposurePolicy, 'manual_capped');
      expect(take.blurAtNormalPace, closeTo(1.575, 0.001));
      expect(take.exposureNs, 8333333);
    });

    test('a device with no manual control says so rather than implying a cap held', () {
      final take = withExposure({'policy': 'auto_locked', 'exposure_ns': null});
      expect(take.exposurePolicy, 'auto_locked');
      expect(take.exposureNs, isNull);
    });

    test('an older manifest without the field degrades to unknown', () {
      const take = Take(
        name: 'x', path: '/tmp/x', bytes: 0, offloaded: false, incomplete: false,
        manifest: {},
      );
      expect(take.exposurePolicy, 'unknown');
      expect(take.blurAtNormalPace, isNull);
    });
  });

  group('CaptureStatus', () {
    test('an empty map yields safe defaults', () {
      const status = CaptureStatus();
      expect(status.recording, isFalse);
      expect(status.clockAligned, isFalse);
      // No focal length means blur cannot be computed, so motion warnings must stay off rather
      // than report zero smear and read as "all clear".
      expect(status.canWarnAboutMotion, isFalse);
    });

    test('a focal length enables motion warnings', () {
      final status = CaptureStatus.fromMap({'focalPixels': 1357.0});
      expect(status.canWarnAboutMotion, isTrue);
    });
  });
}
