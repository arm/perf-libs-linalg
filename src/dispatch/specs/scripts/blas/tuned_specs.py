# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

from .kernel_spec import KernelSpecGenerator


def _kspec_gen(routine):
    file_name = routine["routine"] + "_" + routine["datatype"]
    return KernelSpecGenerator(
        file_name=file_name, routine=routine, target=routine["target"]
    )


def spec(routine):
    if "kernel_spec" in routine:
        return _kspec_gen(routine).generate_kernel_spec(routine["kernel_spec"], True)
    return ""
