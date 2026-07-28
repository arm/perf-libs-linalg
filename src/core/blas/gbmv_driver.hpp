/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_GBMV_DRIVER_HPP
#define PERFLIBS_LINALG_GBMV_DRIVER_HPP

#include "blas/banded_mv_common.hpp"

#include "operators/parallelize.hpp"
#include "operators/copy_matrix.hpp"

#include "framework/buffer_pool.hpp"

#include "perflibs_assert.hpp"

namespace perflibs::linalg {
namespace {

/*
	This driver is responsible for calling to the right operator that will process sbmv, hbmv, gbmv or tbmv operations.
	It is also responsible by parallelizing the work given to each thread.
	The work is parallelized in the strd dimension.

	Example
		Input:
			A[                              y
				[###         ]             [#
				[####        ]              #
				[ ####       ]              #
				[  ####      ]              #
				[   ####     ]              #
				[    ####    ]              #]
			]
			x[############]

		If we are to complete any of the above operations in this matrix using two threads, then the way the work is split is:

		A_t1[                              y_t1           x[############]
				[###         ]             [#
				[####        ]              #
				[ ####       ]              #]
			]
			----------------------------------------------------------------------
		A_t2[                              y_t2            x[############]
				[  ####      ]             [#
				[   ####     ]              #
				[    ####    ]              #]
			]
*/
struct gbmv_driver {
	template <typename AType, typename BType, typename CType, typename ScalarType, typename Spec, typename... Args>
	void operator()(Spec specs, const AType& a, const BType& b, CType& c, compute_position pos,
	                ScalarType alpha, ScalarType beta, Args&&... args) {
		// Sanity checks
		PERFLIBS_ASSERT(c.cntg() == a.strd(), "c cntg doesn't match a strd");
		PERFLIBS_ASSERT(b.cntg() == a.cntg(), "b ctng doesn't match a cntg");
		PERFLIBS_ASSERT(b.strd() == c.strd(), "b strd doesn't match c strd");

		// Fix if y is nan
		if (beta == ScalarType{0}) {
			set(zero<ScalarType>, c);
		}

		// Bypass if alpha is 0
		if (alpha == zero<ScalarType>) {
			scale(beta, c);
			return;
		}

		const auto kernel_axpby = specs.kernel_axpby;
		const auto kernel_dot = specs.kernel_dot;

		if constexpr(is_hermitian_matrix_v<AType> || is_symmetric_matrix_v<AType>){
			auto driver =
				parallelize             { general_parallel_strat, a_strd, specs.max_threads,
				mv_banded_with_symmetry { kernel_axpby, kernel_dot } };

			driver(a, b, c, pos, alpha, beta);
			return;
		}

		if (a.is_trans()) {
			using b_value_type = std::remove_cv_t<typename BType::value_type>;
			const auto buf_size = b.cntg();

			b_value_type *buffer = is_cntg_contig(b)
					? nullptr
					: get_memory<b_value_type, memory_bank::level2>(buf_size * specs.max_threads);

			buffer_pool buffer_pool { buffer, specs.max_threads, buf_size };

			auto driver =
				parallelize      { general_parallel_strat, a_strd, specs.max_threads,
				copy_matrix      { b_matrix, buffer_pool, general_cntg_contig_generator{},
				mv_banded        { kernel_axpby, kernel_dot } } };

			driver(a, b, c, pos, alpha, beta);
			if (buffer != nullptr) {
				return_memory<b_value_type, memory_bank::level2>(buffer);
			}

		}
		else {
			using c_value_type = std::remove_cv_t<typename CType::value_type>;
			const auto buf_size = c.cntg();

			c_value_type *buffer = is_cntg_contig(c)
					? nullptr
					: get_memory<c_value_type, memory_bank::level2>(buf_size * specs.max_threads);

			buffer_pool buffer_pool { buffer, specs.max_threads, buf_size };

			auto driver =
				parallelize      { general_parallel_strat, a_strd, specs.max_threads,
				copy_matrix      { c_matrix, buffer_pool, general_cntg_contig_generator{},
				mv_banded        { kernel_axpby, kernel_dot } } };

			driver(a, b, c, pos, alpha, beta);
			if (buffer != nullptr) {
				return_memory<c_value_type, memory_bank::level2>(buffer);
			}
		}
    }
};

} // namespace anon
} // namespace perflibs::linalg

#endif
