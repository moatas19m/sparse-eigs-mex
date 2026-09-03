#!/usr/bin/env bash
# Vendored STATIC SuiteSparse build with 64-bit indices, for linking into a MEX file.
#
# Why vendored rather than Homebrew:
#   * MATLAB R2017a on macOS is x86_64 only; Homebrew on Apple Silicon ships arm64
#     libraries, which cannot link into a .mexmaci64.
#   * Static linking avoids dylib/rpath breakage when the MEX moves to another machine.
#   * We must match MATLAB's -largeArrayDims (64-bit mwIndex) => cholmod_l_* API.
#
# Usage:  ./build_suitesparse.sh [ARCH]      ARCH = arm64 | x86_64   (default: host)
set -euo pipefail
ARCH="${1:-$(uname -m)}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$ROOT/third_party/suitesparse-$ARCH"
SRC="$ROOT/third_party/SuiteSparse"
VER="v7.14.0"

[ -d "$SRC" ] || git clone --depth 1 --branch "$VER" \
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git "$SRC"

cmake -S "$SRC" -B "$ROOT/build/ss-$ARCH" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
  -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;colamd;ccolamd;cholmod;umfpack" \
  -DSUITESPARSE_USE_64BIT_BLAS=OFF \
  -DCHOLMOD_SUPERNODAL=ON \
  -DSUITESPARSE_USE_OPENMP=ON

cmake --build "$ROOT/build/ss-$ARCH" -j"$(getconf _NPROCESSORS_ONLN)"
cmake --install "$ROOT/build/ss-$ARCH"
echo "Static SuiteSparse ($ARCH) installed to $PREFIX"
