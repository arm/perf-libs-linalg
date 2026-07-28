# Netlib BLAS/LAPACK

This directory contains Netlib LAPACK/BLAS source used by this project. See
[third-party notices](../THIRD_PARTY_NOTICES.md) for the upstream version,
repository, and license.

## Layout

- `upstream/`: upstream files kept in their imported form.
- `reference/`: upstream-derived files adapted for this repository's reference implementation.
- `extensions/`: local Netlib-derived, BLAS-like, or replacement routines.
- `tests/`: upstream BLAS test sources and inputs used for validation.

## Local Changes

Current intentional differences from the upstream sources include:

- `_reference` symbol naming for reference implementations.
- `lsame_` is supplied by a C replacement in `extensions/` to avoid a Fortran
  runtime dependency from character comparison.
- NaN propagation in selected `[sdcz]trsv` and `[sdcz]trsm` sources, where zero-value checks are removed to avoid masking NaNs (see [Reference-LAPACK issue #636](https://github.com/Reference-LAPACK/lapack/issues/636)).
- Complex NRM2 test handling for cases where expected and computed norms are both infinite (see [Reference-LAPACK issue #1047](https://github.com/Reference-LAPACK/lapack/issues/1047)).
