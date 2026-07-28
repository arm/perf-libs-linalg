/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_OS_HPP
#define PERFLIBS_OS_HPP

namespace perflibs {

namespace linalg {

namespace os {

#ifdef __linux__
inline constexpr bool linux = true;
#else
inline constexpr bool linux = false;
#endif

#ifdef __APPLE__
inline constexpr bool mac = true;
#else
inline constexpr bool mac = false;
#endif

#if defined(_M_ARM64EC) || defined(__arm64ec__)
inline constexpr bool windows = true;
inline constexpr bool windows_arm64ec = true;
#elif defined(_WIN32)
inline constexpr bool windows = true;
inline constexpr bool windows_arm64ec = false;
#else
inline constexpr bool windows = false;
inline constexpr bool windows_arm64ec = false;
#endif

} // namespace os

} // namespace linalg

} // namespace perflibs

#endif
