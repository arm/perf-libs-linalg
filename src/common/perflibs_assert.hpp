/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_ASSERT_HPP
#define PERFLIBS_ASSERT_HPP

#include <cassert>
#include <cstdio>

#include "perflibs_unused.hpp"

#define PERFLIBS_ASSERT_MSG(pred, msg) \
	if(!(pred)){ std::fprintf(stderr, "assert failed at: %s:%d -- %s\n", __FILE__, __LINE__, msg); assert(pred); }

#define PERFLIBS_ASSERT_NO_MSG(pred) \
	if(!(pred)){ std::fprintf(stderr, "assert failed at: %s:%d -- predicate `%s`\n", __FILE__, __LINE__ , #pred); assert(pred); }

#define PERFLIBS_ASSERT_3RD_ARG(arg1, arg2, arg3, ...) arg3

#define PERFLIBS_ASSERT_CHOOSER(...) \
	PERFLIBS_ASSERT_3RD_ARG(__VA_ARGS__, PERFLIBS_ASSERT_MSG, PERFLIBS_ASSERT_NO_MSG, )

/*
 * This is defined by the build system:
 * #define PERFLIBS_ASSERT_ON 1
 * It is enabled when the library is built with asserts=1.
 */
#ifdef PERFLIBS_ASSERT_ON
	#define PERFLIBS_ASSERT(...) PERFLIBS_ASSERT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

	#define PERFLIBS_WARNING(pred, msg) \
		if(!(pred)){ std::fprintf(stderr, "WARNING failed at: %s:%d -- %s\n", __FILE__, __LINE__, msg); }
#else //PERFLIBS_ASSERT_ON
	#define PERFLIBS_ASSERT(pred, ...) if(!(pred)) { __builtin_unreachable(); }
	#define PERFLIBS_WARNING(pred, msg) { PERFLIBS_UNUSED(pred); }
#endif //PERFLIBS_ASSERT_ON

/**
 * A version of PERFLIBS_ASSERT that will be checked regardless of PERFLIBS_ASSERT_ON
 * this is useful when you have dead branches such as:
 *
 * if(cond1) ...; return x;
 * else if(cond2) ...; return y;
 * else assert(false); // unreachable branch (nothing to return)
 */
#define PERFLIBS_ALWAYS_ASSERT(...) PERFLIBS_ASSERT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#endif //PERFLIBS_ASSERT_HPP
