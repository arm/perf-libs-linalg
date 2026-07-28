# Contributing

## Licensing

By contributing, you confirm that you have the right to submit the change under the repository license terms. Keep SPDX headers intact and add them to new source files where appropriate.

## Building

See [README.md](./README.md) for build requirements and options.

Before submitting a change, configure and build at least one relevant configuration:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Before submitting a change, also run the test suite and confirm that all tests pass:

```sh
ctest --test-dir build --output-on-failure
```

## Changelog

For notable user-facing changes, add a concise entry under `[Unreleased]` in
[CHANGELOG.md](./CHANGELOG.md). Routine internal changes do not need an entry.

## Pull Requests

Submit changes through a GitHub pull request. Describe the purpose of the
change, the testing performed, and any relevant performance considerations.

## Code Structure

- `include/`: public C headers and ABI types.
- `src/core/`: problem contexts, routine interfaces, strategies, and shared implementation code.
- `src/kernels/`: optimized kernels, assembly templates, and kernel generators.
- `src/runtime/`: CPU and system detection.
- `src/dispatch/`: runtime dispatch code and checked-in dispatch data.
- `netlib/`: imported Netlib reference sources and BLAS test suite.

The public BLAS and LAPACK routine interfaces translate arguments into typed problem contexts. Those contexts are resolved through strategy selection and system-specific dispatch to shared strategies and optimized kernels.

Most routine behavior is shared below the interface layer. A change in `src/core`, `src/kernels`, `src/runtime`, or `src/dispatch` may therefore affect multiple public routines.

System tuning and strategy-selection data live under `src/dispatch/specs/json_specs/`. The JSON files under `src/dispatch/architectures/json/` and `src/dispatch/chooser/json/` are inputs used to generate architecture and chooser sources during the build.
