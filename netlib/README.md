# Netlib BLAS/CBLAS/LAPACK

See [third-party notices](../THIRD_PARTY_NOTICES.md) for the upstream version,
repository, and license.

## Layout

- `upstream/`: imported upstream sources.
- `reference/`: upstream-derived files adapted for this repository's reference implementation.
- `extensions/`: local Netlib-derived, BLAS-like, or replacement routines.
- `tests/`: BLAS/CBLAS test integration.

## Dependency

Reference-LAPACK is pinned in `cmake/dependencies/lapack.cmake`.
For offline builds, set `LINALG_LAPACK_URL` to a local archive, or
`FETCHCONTENT_SOURCE_DIR_REFERENCE_LAPACK` to a source tree with the
patches in `netlib/patches/` applied.

## Local Changes

- `_reference` symbol naming for reference implementations.
- `lsame_` is supplied by a C replacement in `extensions/` to avoid a Fortran
  runtime dependency from character comparison.
- NaN propagation in selected `[sdcz]trsv` and `[sdcz]trsm` sources, where zero-value checks are removed to avoid masking NaNs (see [Reference-LAPACK issue #636](https://github.com/Reference-LAPACK/lapack/issues/636)).
- Changes to fetched sources are kept in [patches/](patches/).
