#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set -eu

fallback_jobs=$1
shift
make_command=$1
shift

# CMake drives the dispatch Makefile through this wrapper so we can decide
# parallelism at build time. Recursive make builds should inherit MAKEFLAGS and
# join the parent jobserver; Ninja has no jobserver to pass through, so it uses
# the fallback job count instead of running this nested make serially.
case " ${MAKEFLAGS:-} " in
    *" --jobserver-auth="*|*" --jobserver-fds="*)
        exec "$make_command" "$@"
        ;;
esac

exec "$make_command" "-j${CMAKE_BUILD_PARALLEL_LEVEL:-$fallback_jobs}" "$@"
