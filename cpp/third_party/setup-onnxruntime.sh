#!/usr/bin/env bash
#
# setup-onnxruntime.sh — extract the vendored ONNX Runtime SDK tarball into
# ./onnxruntime/ (the layout the CMake build expects).
#
# The release archive (onnxruntime-<platform>-<ver>.tgz) extracts to a
# versioned top folder (e.g. onnxruntime-linux-x64-1.20.0/); this script
# renames it to onnxruntime/. Idempotent: re-runs re-extract cleanly.
#
# `cmake -S cpp -B cpp/build` runs the same logic automatically on first
# configure, so this script is optional — use it to pre-extract or to debug
# an extraction. The extracted tree is gitignored; the tarball is the
# committed source of truth.
#
# drvfs/NTFS note: the .tgz stores libonnxruntime.so and libonnxruntime.so.1
# as *symlinks* to the versioned libonnxruntime.so.<ver>. WSL-on-Windows-drive
# mounts (/mnt/...) and some NTFS mounts refuse to create those symlinks on
# extraction, so tar exits non-zero and leaves the .so missing — which would
# break both the link step (needs libonnxruntime.so) and the runtime RPATH
# load (needs libonnxruntime.so.1, the embedded SONAME). We tolerate tar's
# non-zero exit and materialize the missing .so* as real copies of the
# versioned lib, so the build works on any filesystem.
#
# Usage:
#   bash cpp/third_party/setup-onnxruntime.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TGZ="$(ls "$HERE"/onnxruntime-*.tgz 2>/dev/null | head -n1 || true)"
if [ -z "$TGZ" ]; then
    echo "setup-onnxruntime: no onnxruntime-*.tgz found in $HERE" >&2
    echo "  Download one from https://github.com/microsoft/onnxruntime/releases" >&2
    exit 1
fi

DEST="$HERE/onnxruntime"
echo "setup-onnxruntime: extracting $TGZ -> $DEST/"

# Stage into a temp dir, then move the versioned top folder into place.
TMP="$(mktemp -d "$HERE/.ort-extract.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Tolerate non-zero: drvfs can refuse to create the .so symlinks (see header).
# Real files (headers, the versioned .so, cmake, pkgconfig) still extract.
tar xzf "$TGZ" -C "$TMP" || true

SRC="$(find "$TMP" -maxdepth 1 -mindepth 1 -type d -name 'onnxruntime-*' | head -n1)"
if [ -z "$SRC" ]; then
    echo "setup-onnxruntime: extraction produced no onnxruntime-* folder in $TMP" >&2
    exit 1
fi

# Materialize the .so symlink chain as real copies where the symlinks are
# missing (drvfs/NTFS). The linker links against libonnxruntime.so; the binary's
# RPATH resolves libonnxruntime.so.1 (the SONAME) at runtime — both must exist.
LIB="$SRC/lib"
REAL="$(ls "$LIB"/libonnxruntime.so.*.* 2>/dev/null | sort -V | tail -n1 || true)"
if [ -n "$REAL" ]; then
    [ -e "$LIB/libonnxruntime.so.1" ] || cp -f "$REAL" "$LIB/libonnxruntime.so.1"
    [ -e "$LIB/libonnxruntime.so"   ] || cp -f "$REAL" "$LIB/libonnxruntime.so"
fi

rm -rf "$DEST"
mv "$SRC" "$DEST"

echo "setup-onnxruntime: done. $DEST  (ORT $(cat "$DEST/VERSION_NUMBER" 2>/dev/null || echo '?'))"
