/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_OPERATORS_PACK_HPP
#define PERFLIBS_LINALG_OPERATORS_PACK_HPP

#include "framework/which.hpp"
#include "framework/buffer_pool.hpp"
#include "framework/compute_position.hpp"

#include <utility>

namespace perflibs::linalg {

/**
 * packs one of the three input matrices from its unpacked form into
 * a form specified by the Conversion object.
 *
 * The matrix to pack is specified by the WhichMatrix template parameter
 *
 * In effect, this object is an adaptor to fit the conversion object into the stack system.
 */
template<which_matrix WhichMatrix, typename BufferPool, typename Convert, typename Next>
class pack {
	BufferPool buffer_;
	Convert convert_;
	Next next_;

public:
	PERFLIBS_LINALG_INLINE
	pack(BufferPool buffer , Convert convert, Next next)
	:	buffer_  { std::move(buffer)  }
	,	convert_ { std::move(convert) }
	,	next_    { std::move(next)    }
	{	}

	PERFLIBS_LINALG_INLINE
	pack(which_matrix_constant<WhichMatrix>, BufferPool buffer, Convert convert, Next next)
	: pack(std::move(buffer), std::move(convert), std::move(next)) { }

	template<typename AMatrixType, typename BMatrixType, typename CMatrixType, typename... Args>
	PERFLIBS_LINALG_INLINE
	void operator()(const AMatrixType& a, const BMatrixType& b, const CMatrixType& c, const compute_position& pos, Args... args) {
		auto buffer = buffer_.get_buffer(pos.thread_num);

		if constexpr(WhichMatrix == which_matrix::a) {
			auto packed_a = convert_(a, buffer);
			next_(packed_a, b, c, pos, std::forward<Args>(args)...);
		}
		else if constexpr(WhichMatrix == which_matrix::b) {
			auto packed_b = convert_(b, buffer);
			next_(a, packed_b, c, pos, std::forward<Args>(args)...);
		}
	}
}; //class pack

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_OPERATORS_PACK_HPP
