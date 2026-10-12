# Changelog

All notable changes to `perf-libs-linalg` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Add double-precision interleaved-batch BLAS and factorization routines.
- Add optional AddressSanitizer builds.

### Performance

- Reduce build times and library size.
- Improve instruction scheduling in the SVE complex GEMM kernels.
- Improve SVE transposed GEMV performance with strided output.
- Improve IMATADD, OMATADD, IMATCOPY and OMATCOPY performance.
- Improve Neon performance for CROT and ZROT.
- Enable SVE kernels for DOT, CROT, ZROT and WAXPBY.

### Fixed

- Fix a data race in parallel QR factorization.
- Fix excessive memory use in parallel GEMM.
- Fix incorrect results in parallel WAXPBY.
- Fix integer overflow in ILP64 ROTM.
- Prevent spurious parameter errors in TRSV.
- Respect thread throttling in parallel POTRF.

## [26.07]

This is the initial release of `perf-libs-linalg`.

### Added

- A full BLAS interface and selected LAPACK routines for AArch64 systems.
- Build support for Linux, macOS, and Windows on AArch64 systems.
- Support for LP64 and ILP64 integer models, and serial and OpenMP threading
  models.
