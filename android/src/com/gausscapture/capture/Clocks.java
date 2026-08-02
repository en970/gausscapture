package com.gausscapture.capture;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.os.SystemClock;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Deciding whether the camera and the motion sensors are on the same clock, by measuring.
 *
 * <p>This is the reason the app is native rather than a screen recorder. Pairing a frame's camera
 * pose with the accelerometer reading taken at that instant recovers which way is up, and it works
 * to an agreement of 0.998 across a capture — but only if the two timestamp streams refer to the
 * same clock. If they do not, the error is not a wobble; it is the accumulated deep-sleep time
 * since boot, which can be hours.
 *
 * <p>Android gives half the answer directly: {@code SENSOR_INFO_TIMESTAMP_SOURCE} says whether the
 * camera stamps frames on {@code CLOCK_BOOTTIME}. The other half is not exposed at all. Sensor
 * events are documented to use {@code CLOCK_BOOTTIME}, and several OEM builds — Samsung among them
 * — have shipped sensors on {@code CLOCK_MONOTONIC} instead. Nothing reports this. The difference
 * only becomes visible if you look, and it is one subtraction: an event's timestamp is compared
 * against both {@code elapsedRealtimeNanos} (boottime) and {@code nanoTime} (monotonic), and
 * whichever it sits closer to is the clock it is on.
 *
 * <p>The previous version asserted alignment and printed "imu offset recorded" without ever
 * computing an offset (audit D11, D12). Everything here is measured, and where a value cannot be
 * established it is recorded as unknown rather than as true.
 */
final class Clocks {

    /** Beyond this, a sample is not plausibly on the clock being tested. */
    private static final long PLAUSIBLE_NS = 5_000_000_000L;

    enum SensorClock {
        /** elapsedRealtimeNanos: keeps counting through suspend. What the camera uses. */
        BOOTTIME,
        /** nanoTime: stops during suspend. Alignment is wrong by the sleep since boot. */
        MONOTONIC,
        /** Neither matched; the stream cannot be trusted against video. */
        UNKNOWN
    }

    private SensorClock sensorClock = SensorClock.UNKNOWN;
    private long sensorOffsetNs;
    private long bootMinusMonotonicNs;
    private boolean measured;

    private Integer cameraTimestampSource;

    /**
     * Work out which clock a sensor event is on, from the first event of a capture.
     *
     * <p>Both references are read as close to each other as possible so the comparison is not
     * polluted by the time this method itself takes.
     */
    synchronized void observeSensorEvent(long eventTimestampNs) {
        if (measured) {
            return;
        }
        long boottime = SystemClock.elapsedRealtimeNanos();
        long monotonic = System.nanoTime();
        bootMinusMonotonicNs = boottime - monotonic;

        long fromBoottime = Math.abs(boottime - eventTimestampNs);
        long fromMonotonic = Math.abs(monotonic - eventTimestampNs);

        if (fromBoottime < fromMonotonic && fromBoottime < PLAUSIBLE_NS) {
            sensorClock = SensorClock.BOOTTIME;
            sensorOffsetNs = 0;
        } else if (fromMonotonic < PLAUSIBLE_NS) {
            // Usable, but every sensor timestamp needs this added before it can be compared with a
            // camera timestamp. Recording it is what makes such a capture salvageable.
            sensorClock = SensorClock.MONOTONIC;
            sensorOffsetNs = bootMinusMonotonicNs;
        } else {
            sensorClock = SensorClock.UNKNOWN;
            sensorOffsetNs = 0;
        }
        measured = true;
    }

    /** Record what the camera says about its own timestamps. */
    void observeCamera(CameraCharacteristics characteristics) {
        cameraTimestampSource = characteristics.get(
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
    }

    boolean cameraUsesRealtime() {
        return cameraTimestampSource != null
                && cameraTimestampSource
                == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME;
    }

    synchronized SensorClock sensorClock() {
        return sensorClock;
    }

    /** Nanoseconds to add to a sensor timestamp to put it on the camera's clock. */
    synchronized long sensorOffsetNs() {
        return sensorOffsetNs;
    }

    /**
     * Whether a capture taken now can have its up direction recovered.
     *
     * <p>True needs both halves: the camera on boottime, and sensors on a clock whose relationship
     * to it is known. A MONOTONIC sensor clock still qualifies, because the offset is measured.
     */
    synchronized boolean alignable() {
        return cameraUsesRealtime() && sensorClock != SensorClock.UNKNOWN;
    }

    /** One short phrase for the preflight sheet. Names the consequence, not the mechanism. */
    synchronized String describe() {
        if (!cameraUsesRealtime()) {
            return "Camera and motion sensors disagree — up will not be recoverable";
        }
        switch (sensorClock) {
            case BOOTTIME:
                return "Camera and motion sensors share a clock";
            case MONOTONIC:
                return "Clocks differ by a measured offset, recorded with the take";
            default:
                return "Sensor clock not yet established";
        }
    }

    synchronized JSONObject toJson(long recordingStartElapsedNs,
                                   long recordingStartUptimeNs,
                                   long recordingStartWallMs,
                                   Long firstFrameTimestampNs) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("recording_start_elapsed_realtime_ns", recordingStartElapsedNs);
        json.put("recording_start_uptime_ns", recordingStartUptimeNs);
        json.put("recording_start_wall_ms", recordingStartWallMs);

        json.put("camera_timestamp_source", cameraTimestampSource == null
                ? JSONObject.NULL
                : (cameraUsesRealtime() ? "REALTIME" : "UNKNOWN"));
        json.put("sensor_clock", sensorClock.name());
        json.put("sensor_to_camera_offset_ns", sensorOffsetNs);
        json.put("boottime_minus_monotonic_ns", bootMinusMonotonicNs);
        json.put("camera_imu_same_clock", alignable());

        // The timestamp of the first frame the encoder actually wrote. For a camera feeding a
        // MediaRecorder surface this IS the presentation timestamp in the MP4, so it anchors
        // frames.jsonl to video.mp4 exactly, instead of by ordinal (audit D03).
        json.put("first_encoded_frame_t_ns",
                firstFrameTimestampNs == null ? JSONObject.NULL : firstFrameTimestampNs);
        return json;
    }
}
