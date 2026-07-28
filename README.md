# perf-libs-linalg

A library of dense linear algebra functions for AArch64 systems.

It provides a full BLAS interface and a subset of LAPACK routines, using strategies and optimized kernels selected at runtime.

## Requirements

- CMake 3.21 or newer
- A toolchain with the required C++ and AArch64 ISA support
  - supported build compilers: GCC 14 and LLVM/Clang 22
  - assembler support is required for `+sve`, `+fp16`, `+bf16`, and `+sme2`
  - other C++26-capable compilers may work
- A Fortran compiler for the Netlib reference sources
- Python 3.10 or newer with Mako and NumPy
- `make` and `find`
- OpenMP development support when building with `LINALG_THREADING_MODEL=openmp`

The CMake project currently requests C++23 because older CMake versions do not provide portable C++26 standard detection.

## Limitations

- Linux and macOS on AArch64 are supported build/test platforms.
- Windows AArch64 support is experimental and assumes LLVM/Clang in a [wenv](https://gitlab.com/Linaro/windowsonarm/wenv)/MSYS2-style environment.
- The library provides a full BLAS interface and selected LAPACK routines, not a complete LAPACK implementation.
- CBLAS, LAPACKE, and Fortran modules are not currently provided.

## Quick Start

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install mako numpy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This builds the `pl_linalg` target.

## Build Options

- `BUILD_SHARED_LIBS=ON|OFF`: builds shared or static libraries where supported. The default is `OFF`.
- `LINALG_INTEGER_MODEL=lp64|ilp64`: selects the public integer ABI. The default is `lp64`.
- `LINALG_THREADING_MODEL=serial|openmp`: selects the threading backend. The default is `serial`.

Example:

```sh
cmake -S . -B build-openmp \
  -DCMAKE_BUILD_TYPE=Release \
  -DLINALG_THREADING_MODEL=openmp \
  -DLINALG_INTEGER_MODEL=ilp64
cmake --build build-openmp --parallel
```

## Testing

```sh
ctest --test-dir build --output-on-failure
```

This currently runs the Netlib BLAS test suite.

## Install

```sh
cmake --install build --prefix /path/to/install
```

This installs the `pl_linalg` library and public headers from `include/`.

## Source Layout

- `include/`: public C headers and ABI types
- `src/core/`: problem contexts, interfaces, strategies, and shared linear algebra machinery
- `src/kernels/`: optimized AArch64 kernels and kernel generators
- `src/runtime/`: CPU and system detection
- `src/dispatch/`: runtime dispatch and generated selection sources
- `netlib/`: imported Netlib reference sources and BLAS test suite
- `cmake/`: top-level CMake build options

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidance.

## License

This repository is licensed under either MIT or Apache-2.0 with LLVM exception. See `LICENSE` and `LICENSES/`.

Third-party source notices are listed in `THIRD_PARTY_NOTICES.md`.
