# Android arm64 build and test

Cross-compile perf-libs-linalg for Android arm64 using the Android Native
Development Kit (NDK) and Flang. Before configuring the build, set
`ANDROID_NDK_ROOT` to the NDK installation directory.

## Build the library

The library defaults to static output (`BUILD_SHARED_LIBS=OFF`); this example
explicitly selects shared output. Run these commands from the repository root.

```sh
cmake -S . -B build-android \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchains/android-arm64-flang.cmake" \
  -DANDROID_NDK_ROOT="$ANDROID_NDK_ROOT" \
  -DCMAKE_Fortran_COMPILER="/path/to/flang" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLINALG_THREADING_MODEL=serial
cmake --build build-android --parallel
cmake --install build-android --prefix /path/to/android-package --strip
```

Set `-DBUILD_SHARED_LIBS=OFF` to build and install `libpl_linalg.a` instead of
`libpl_linalg.so`. Static library output is supported for both serial and OpenMP
builds. When linking the archive into an Android application, also link the NDK
C++ runtime and, for OpenMP builds, `libomp.so`.

The C++ runtime choice is independent of `BUILD_SHARED_LIBS`. The toolchain
defaults to `ANDROID_STL=c++_shared`; set `-DANDROID_STL=c++_static` to use the
static C++ runtime.

The optional `--strip` flag removes debug information from the installed shared
library, reducing its size. The unstripped build-tree copy remains available for
debugging. Omit the flag to retain debug information in the installed
library. This option does not strip object files within static archives.

Set `-DLINALG_THREADING_MODEL=openmp` to use OpenMP. CMake locates the NDK's
`libomp.so` automatically for linking. Package this runtime with the
application; the library install step does not copy it.

When using `c++_shared`, package one copy of the NDK's `libc++_shared.so` with
the application. For `c++_static`, the final link includes the C++ runtime
statically instead. Choose a runtime consistent with the application's other
native libraries; avoid linking separate copies of the static C++ runtime into
multiple shared libraries. See the [NDK C++ runtime
guidance](https://developer.android.com/ndk/guides/cpp-support).

## Build the tests

Configure each build with `-DBUILD_TESTING=ON` and
`-DLINALG_ANDROID_FLANG_RUNTIME_LIBRARIES=/path/to/libflang_rt.runtime.a`. Use a
Flang runtime built for Android arm64, not the host runtime. The archive is
linked into the test executables and does not need to be copied to the
device. If you select a shared Flang runtime instead, copy the supplied shared
libraries to the device alongside the test executables.

The install step does not copy test executables, test inputs, or runtime
libraries. To run the BLAS tests through ADB, copy the `xblat*` executables
from the build's `netlib/tests/blas` directory and the `.in` files from
`BLAS/TESTING` in the fetched Reference-LAPACK source tree. Also copy
`libpl_linalg.so` for shared library builds, `libc++_shared.so` when using
`ANDROID_STL=c++_shared`, and `libomp.so` for OpenMP builds. Set
`LD_LIBRARY_PATH` to the device directory containing the shared libraries.
Level 1 tests do not require input files;
redirect the corresponding `.in` file to standard input for Level 2 and 3 tests.
