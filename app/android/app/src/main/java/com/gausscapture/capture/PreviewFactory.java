package com.gausscapture.capture;

import android.app.Activity;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.view.TextureView;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.Map;

import io.flutter.plugin.common.StandardMessageCodec;
import io.flutter.plugin.platform.PlatformView;
import io.flutter.plugin.platform.PlatformViewFactory;

/**
 * The camera preview, as a widget Flutter can place anywhere.
 *
 * <p>This is a platform view rather than a Flutter texture on purpose. The transform that makes the
 * preview upright and correctly proportioned is written against a real {@code TextureView}, and it
 * is the most error-prone code in the app: {@code TextureView} always stretches its buffer to the
 * view bounds and applies any transform on top, so the correction has to cancel the stretch using
 * the buffer's <em>true</em> dimensions first, then rotate, then fit — and using post-rotation
 * dimensions in the first step squeezes a 16:9 image into a 9:16 box in a way no later step can
 * undo. That code and its explanation survived the rewrite untouched. Re-deriving it against
 * Flutter's texture pipeline would have meant getting it wrong again.
 */
final class PreviewFactory extends PlatformViewFactory implements CameraController.Listener {

    private final Activity activity;
    private final CaptureEngine engine;
    private PreviewView current;

    PreviewFactory(Activity activity, CaptureEngine engine) {
        super(StandardMessageCodec.INSTANCE);
        this.activity = activity;
        this.engine = engine;
    }

    @NonNull
    @Override
    public PlatformView create(Context context, int viewId, @Nullable Object args) {
        current = new PreviewView(context);
        engine.attachPreview(current.textureView, this);
        return current;
    }

    /** Re-open the camera once the surface and the permission are both available. */
    void resume(int displayRotation) {
        if (current != null && current.ready) {
            engine.openCamera(displayRotation);
        }
    }

    @Override
    public void onReady(String summary) {
        // The interface reads this through the status channel rather than being pushed at.
    }

    @Override
    public void onError(String message) {
        // Surfaced through the engine's problem channel, which the Dart side already listens to.
    }

    @Override
    public void onFrameMetadata(long exposureNanos, int iso) {
        // Polled by the telemetry stream instead: a callback at 30 Hz across the platform boundary
        // would cost more than it tells anyone.
    }

    private final class PreviewView implements PlatformView {

        private final TextureView textureView;
        private volatile boolean ready;

        PreviewView(Context context) {
            textureView = new TextureView(context);
            textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture texture, int w, int h) {
                    ready = true;
                    engine.openCamera(rotation());
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int w, int h) {
                    CameraController camera = engine.camera();
                    if (camera != null) {
                        camera.applyTransform(rotation());
                    }
                }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
                    ready = false;
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture texture) {
                }
            });
        }

        private int rotation() {
            return activity.getWindowManager().getDefaultDisplay().getRotation();
        }

        @Override
        public View getView() {
            return textureView;
        }

        @Override
        public void dispose() {
            engine.closeCamera();
        }
    }
}
