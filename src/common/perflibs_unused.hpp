/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_UNUSED_HPP
#define PERFLIBS_UNUSED_HPP

/**
 * PERFLIBS_UNUSED is used to mark a variable as unused when it
 * may not be trivial to dispense of a value - and a compiler
 * warning/error is generated, for example
 *
 * auto [ used, unused ] = my_function();
 * //unused variable warning without PERFLIBS_UNUSED
 * PERFLIBS_UNUSED(unused);
 */

#define PERFLIBS_UNUSED(x) (void)(x)

#endif // PERFLIBS_UNUSED_HPP
