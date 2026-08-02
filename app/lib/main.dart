import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'capture_engine.dart';
import 'design.dart';
import 'screens/capture_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  runApp(const GaussCaptureApp());
}

/// A capture instrument that happens to be an app.
///
/// The interface is Flutter; the recording is not, and that split is the whole architecture.
/// Flutter's camera plugin cannot report a per-frame SENSOR_TIMESTAMP and its sensor plugins do
/// not expose SensorEvent.timestamp, and those two values on one hardware clock are what let a
/// frame's pose be paired with the accelerometer reading taken at that instant. That pairing is
/// how the world's up direction is recovered, at 0.998 agreement across a capture. Writing the
/// recording in Dart would have produced a better-looking app that gathered unusable data.
class GaussCaptureApp extends StatelessWidget {
  const GaussCaptureApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GaussCapture',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(),
      home: CaptureScreen(engine: CaptureEngine()),
    );
  }
}
