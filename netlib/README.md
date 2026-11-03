# Netlib BLAS/CBLAS/LAPACK

See [third-party notices](../THIRD_PARTY_NOTICES.md) for the upstream version,
repository, and license.

## Layout

- `extensions/`: local Netlib-derived, BLAS-like, or replacement routines.
- `tests/`: BLAS/CBLAS/LAPACK test integration.

## Dependency

Reference-LAPACK is pinned in `cmake/dependencies/lapack.cmake`.
For offline builds, set `LINALG_LAPACK_URL` to a local archive, or
`FETCHCONTENT_SOURCE_DIR_REFERENCE_LAPACK` to a source tree with the
patches in `netlib/patches/` applied.

## Local Changes

- `lsame_` is supplied by a C replacement in `extensions/` to avoid a Fortran
  runtime dependency from character comparison.
- Changes to fetched sources are kept in [patches/](patches/).
