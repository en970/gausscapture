#!/usr/bin/env bash
# Build the capture APK without Gradle.
#
# The Android Gradle Plugin pulls roughly a gigabyte of tooling and its own
# dependency cache. For a single-activity app with no libraries, the SDK's own
# tools are enough and every step stays visible:
#
#   aapt2 compile  ->  compiled resources
#   aapt2 link     ->  resource APK + R.java
#   javac          ->  .class files
#   d8             ->  classes.dex
#   zip            ->  add dex to the APK
#   zipalign       ->  align for mmap
#   apksigner      ->  sign with a local debug key
#
# Usage:  ./build.sh [install]
set -euo pipefail
cd "$(dirname "$0")"

SDK="${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}"
[ -d "$SDK" ] || { echo "Android SDK not found at $SDK"; exit 1; }

# Highest installed build-tools and platform, so this keeps working as the SDK
# is updated rather than pinning a version that will disappear.
BT="$(ls -1 "$SDK/build-tools" | sort -V | tail -1)"
PLATFORM="$(ls -1 "$SDK/platforms" | sort -V | tail -1)"
TOOLS="$SDK/build-tools/$BT"
ANDROID_JAR="$SDK/platforms/$PLATFORM/android.jar"

# macOS ships a /usr/bin/javac stub that exists but fails when run, so probe by
# executing it rather than by looking it up on PATH.
if ! javac -version >/dev/null 2>&1; then
  for candidate in /opt/homebrew/opt/openjdk@17 /opt/homebrew/opt/openjdk \
                   /usr/local/opt/openjdk@17 /usr/local/opt/openjdk; do
    if [ -x "$candidate/bin/javac" ]; then
      export PATH="$candidate/bin:$PATH"
      export JAVA_HOME="$candidate"
      break
    fi
  done
fi
javac -version >/dev/null 2>&1 || { echo "No working JDK. Try: brew install openjdk@17"; exit 1; }

echo "build-tools $BT   platform $PLATFORM   $(javac -version 2>&1)"

BUILD=build
rm -rf "$BUILD"
mkdir -p "$BUILD/res" "$BUILD/gen" "$BUILD/classes"

echo "==> compiling resources"
"$TOOLS/aapt2" compile --dir res -o "$BUILD/res/resources.zip"

echo "==> linking resources"
"$TOOLS/aapt2" link \
  -I "$ANDROID_JAR" \
  --manifest AndroidManifest.xml \
  --java "$BUILD/gen" \
  --min-sdk-version 28 \
  --target-sdk-version 34 \
  -o "$BUILD/base.apk" \
  "$BUILD/res/resources.zip"

echo "==> compiling java"
find src "$BUILD/gen" -name '*.java' > "$BUILD/sources.txt"
# -source/-target 8 with an explicit bootclasspath compiles strictly against
# Android's API surface. Compiling against the JDK's own java.* instead would
# let a call that does not exist on Android through javac, only to fail at
# runtime on the device. (JDK 17 permits -bootclasspath solely at source 8.)
javac -source 8 -target 8 -nowarn -Xlint:-options \
  -bootclasspath "$ANDROID_JAR" \
  -classpath "$ANDROID_JAR" \
  -d "$BUILD/classes" \
  @"$BUILD/sources.txt"

echo "==> dexing"
"$TOOLS/d8" --min-api 28 --output "$BUILD" $(find "$BUILD/classes" -name '*.class')

echo "==> packaging"
cp "$BUILD/base.apk" "$BUILD/unsigned.apk"
(cd "$BUILD" && zip -q unsigned.apk classes.dex)

echo "==> aligning"
"$TOOLS/zipalign" -f 4 "$BUILD/unsigned.apk" "$BUILD/aligned.apk"

# Deliberately outside $BUILD: that directory is wiped on every build, and a
# regenerated key changes the signature, which makes Android refuse the upgrade
# and forces an uninstall (losing recorded sessions) after every clean build.
KEYSTORE="debug.keystore"
if [ ! -f "$KEYSTORE" ]; then
  echo "==> generating a debug key"
  keytool -genkeypair -v -keystore "$KEYSTORE" -storepass android -keypass android \
    -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=GaussCapture Debug,O=GaussCapture,C=TR" >/dev/null 2>&1
fi

echo "==> signing"
"$TOOLS/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
  --out gausscapture.apk "$BUILD/aligned.apk"
"$TOOLS/apksigner" verify gausscapture.apk && echo "signature ok"

SIZE=$(( $(stat -f%z gausscapture.apk 2>/dev/null || stat -c%s gausscapture.apk) / 1024 ))
echo "built gausscapture.apk (${SIZE} KB)"

if [ "${1:-}" = "install" ]; then
  echo "==> installing"
  "$SDK/platform-tools/adb" install -r gausscapture.apk
fi
