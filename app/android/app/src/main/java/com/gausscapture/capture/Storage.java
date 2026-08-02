package com.gausscapture.capture;

import android.content.Context;
import android.os.Build;
import android.os.Environment;
import android.os.StatFs;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Where captures live, and whether a computer can see them.
 *
 * <p>This looks like a storage decision and is actually a workflow decision. The previous version
 * wrote to {@code getExternalFilesDir()}, which is private to the app and needs no permission —
 * and which One UI has hidden from both the on-device file manager and from MTP since Android 11.
 * The operator's entire routine is "record often, plug the phone into the Mac, take the data", and
 * under that path it cannot happen: the folder simply does not appear in Finder.
 *
 * <p>So the preferred location is {@code /sdcard/Documents/GaussCapture}, which is shared storage
 * and does appear. Reaching it with ordinary file I/O needs {@code MANAGE_EXTERNAL_STORAGE} on
 * Android 11+, which is a heavy permission — justified here because this is a sideloaded research
 * instrument for one operator, the alternative is copying every 180 MB video a second time through
 * MediaStore at the end of each take, and a capture that cannot be retrieved is a capture that was
 * never made.
 *
 * <p>When the permission is absent the app still works, falling back to the private directory. The
 * choice is recorded in every manifest, so a pack always says where it came from and no consumer
 * has to guess.
 */
final class Storage {

    /** Folder name under Documents, and the one an operator will look for in Finder. */
    static final String PUBLIC_FOLDER = "GaussCapture";

    /** Written by the desktop side once a take has been copied off. */
    static final String OFFLOAD_SENTINEL = "offloaded.json";

    enum Kind {
        /** /sdcard/Documents/GaussCapture — visible over MTP. */
        SHARED,
        /** App-private external files — needs adb. */
        PRIVATE
    }

    private final File root;
    private final Kind kind;

    private Storage(File root, Kind kind) {
        this.root = root;
        this.kind = kind;
    }

    /** Pick the best location this device and its permissions allow. */
    static Storage open(Context context) {
        if (canUseSharedStorage()) {
            File documents = Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_DOCUMENTS);
            File candidate = new File(documents, PUBLIC_FOLDER);
            if (candidate.isDirectory() || candidate.mkdirs()) {
                return new Storage(candidate, Kind.SHARED);
            }
        }
        File fallback = context.getExternalFilesDir(null);
        if (fallback == null) {
            // External storage is not mounted. Internal storage always exists, and a take that
            // has to be pulled over adb beats refusing to record.
            fallback = context.getFilesDir();
        }
        fallback.mkdirs();
        return new Storage(fallback, Kind.PRIVATE);
    }

    /**
     * Whether shared storage is both reachable and writable right now.
     *
     * <p>Checked rather than inferred from the permission alone: the grant can be revoked while the
     * app is alive, and on some builds the directory is unwritable even when it is granted.
     */
    static boolean canUseSharedStorage() {
        if (!Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
            return false;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            return false;
        }
        return true;
    }

    File root() {
        return root;
    }

    Kind kind() {
        return kind;
    }

    boolean isVisibleToDesktop() {
        return kind == Kind.SHARED;
    }

    /** The path to show the operator, and the one the desktop tool is told to read. */
    String describePath() {
        return root.getAbsolutePath();
    }

    /**
     * A directory for a new take, guaranteed not to collide with an existing one.
     *
     * <p>The old naming used one-second granularity and treated an existing directory as success,
     * so stopping and restarting inside the same second silently overwrote the previous take
     * (audit D08). Minutes of walking are not worth a second of clock resolution, so a suffix is
     * added until the name is genuinely free.
     */
    File newSessionDir(String presetId) {
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String base = presetId + "_" + stamp;
        File candidate = new File(root, base);
        for (int suffix = 2; candidate.exists(); suffix++) {
            candidate = new File(root, base + "_" + suffix);
        }
        return candidate.mkdirs() ? candidate : null;
    }

    /** Every take on disk, newest first — the order that matters when recording all day. */
    List<File> sessions() {
        File[] entries = root.listFiles(new java.io.FileFilter() {
            @Override
            public boolean accept(File candidate) {
                return candidate.isDirectory();
            }
        });
        if (entries == null) {
            return new ArrayList<>();
        }
        List<File> out = new ArrayList<>(Arrays.asList(entries));
        // Newest first. Names begin with a sortable timestamp, and the take just recorded is the
        // one the operator is looking for -- the old ascending sort buried it at the bottom.
        Collections.sort(out, new Comparator<File>() {
            @Override
            public int compare(File a, File b) {
                return b.getName().compareTo(a.getName());
            }
        });
        return out;
    }

    /** Free space in bytes, or -1 when it cannot be determined. */
    long freeBytes() {
        try {
            StatFs stat = new StatFs(root.getAbsolutePath());
            return stat.getAvailableBlocksLong() * stat.getBlockSizeLong();
        } catch (IllegalArgumentException error) {
            return -1;
        }
    }

    /**
     * Roughly how many more minutes fit, at the bitrate the encoder is configured for.
     *
     * <p>Shown before a take rather than discovered when the card fills mid-walk.
     */
    int minutesRemaining(int videoBitrate) {
        long free = freeBytes();
        if (free < 0 || videoBitrate <= 0) {
            return -1;
        }
        // Video dominates, but the IMU is not free: five sensors at the fastest rate produce
        // roughly 20 MB per minute of JSONL, which is worth counting when the card is nearly full.
        long bytesPerMinute = (long) videoBitrate * 60L / 8L + 20L * 1024L * 1024L;
        return (int) (free / bytesPerMinute);
    }

    /** True once the desktop side has recorded that it copied this take. */
    static boolean isOffloaded(File sessionDir) {
        return new File(sessionDir, OFFLOAD_SENTINEL).exists();
    }

    /**
     * A take that was interrupted: it has a directory but never got its manifest.
     *
     * <p>Process death, a flat battery or a camera error all leave one of these. They used to be
     * invisible to the UI and therefore undeletable, quietly holding a few hundred megabytes each
     * (audit D28).
     */
    static boolean isIncomplete(File sessionDir) {
        return !new File(sessionDir, "manifest.json").exists();
    }

    static long directorySize(File dir) {
        File[] entries = dir.listFiles();
        if (entries == null) {
            return 0;
        }
        long total = 0;
        for (File entry : entries) {
            total += entry.isDirectory() ? directorySize(entry) : entry.length();
        }
        return total;
    }

    static String formatBytes(long bytes) {
        if (bytes >= 1L << 30) {
            return String.format(Locale.US, "%.1f GB", bytes / (double) (1L << 30));
        }
        if (bytes >= 1L << 20) {
            return String.format(Locale.US, "%d MB", bytes >> 20);
        }
        return String.format(Locale.US, "%d KB", Math.max(1, bytes >> 10));
    }
}
