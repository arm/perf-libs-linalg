/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATRIX_INTERLEAVED_TILE_MATRIX_BASE_HPP
#define PERFLIBS_LINALG_MATRIX_INTERLEAVED_TILE_MATRIX_BASE_HPP

#include "framework/linalg_util.hpp"

namespace perflibs::linalg {

template<typename T>
class interleaved_tile_matrix_base {
public:
	using data_type        =T;
	using value_type       =T;
	using const_value_type =const T;
	using reference        =T&;
	using const_reference  =const T&;

private:
	T *data_;

	kernel_inttype cntg_interleave_;
	kernel_inttype strd_interleave_;

	kernel_inttype cntg_;
	kernel_inttype strd_;

	kernel_inttype cntg_interleave_step_;
	kernel_inttype strd_interleave_step_;

	kernel_inttype cntg_tile_step_;
	kernel_inttype strd_tile_step_;

public:
	interleaved_tile_matrix_base(T *data,
		kernel_inttype cntg_interleave,      kernel_inttype strd_interleave,
		kernel_inttype cntg,                 kernel_inttype strd,
		kernel_inttype cntg_interleave_step, kernel_inttype strd_interleave_step,
		kernel_inttype cntg_tile_step,       kernel_inttype strd_tile_step)

	:	data_                 { data }
	,	cntg_interleave_      { cntg_interleave }
	,	strd_interleave_      { strd_interleave }
	,	cntg_                 { cntg            }
	,	strd_                 { strd            }
	,	cntg_interleave_step_ { cntg_interleave_step }
	,	strd_interleave_step_ { strd_interleave_step }
	,	cntg_tile_step_       { cntg_tile_step }
	,	strd_tile_step_       { strd_tile_step }
	{	}

	kernel_inttype get_elem_index(kernel_inttype cntg, kernel_inttype strd) const {
		const auto tile_cntg_idx           = cntg / cntg_interleave_;
		const auto tile_strd_idx           = strd / strd_interleave_;

		const auto tile_cntg_offset        = tile_cntg_idx * cntg_tile_step_;
		const auto tile_strd_offset        = tile_strd_idx * strd_tile_step_;

		const auto interleaved_cntg        = cntg % cntg_interleave_;
		const auto interleaved_strd        = strd % strd_interleave_;

		const auto interleaved_cntg_offset = interleaved_cntg * cntg_interleave_step_;
		const auto interleaved_strd_offset = interleaved_strd * strd_interleave_step_;

		return tile_cntg_offset + interleaved_cntg_offset
		     + tile_strd_offset + interleaved_strd_offset;
	}

	reference operator()(kernel_inttype cntg, kernel_inttype strd) const {
		return data_ [ get_elem_index(cntg, strd) ];
	}

	T *data_at(kernel_inttype cntg, kernel_inttype strd) const {
		return data_ + get_elem_index(cntg, strd);
	}

	PERFLIBS_LINALG_INLINE value_type    *data()                  const { return data_; }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg()                  const { return cntg_;       }
	PERFLIBS_LINALG_INLINE kernel_inttype strd()                  const { return strd_;       }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_interleave()       const { return cntg_interleave_;  }
	PERFLIBS_LINALG_INLINE kernel_inttype strd_interleave()       const { return strd_interleave_;  }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_interleave_step()  const { return cntg_interleave_step_;  }
	PERFLIBS_LINALG_INLINE kernel_inttype strd_interleave_step()  const { return strd_interleave_step_;  }
	PERFLIBS_LINALG_INLINE kernel_inttype cntg_tile_step()        const { return cntg_tile_step_;  }
	PERFLIBS_LINALG_INLINE kernel_inttype strd_tile_step()        const { return strd_tile_step_;  }

	auto sub_matrix(kernel_inttype cntg_idx, kernel_inttype cntg_size,
                    kernel_inttype strd_idx, kernel_inttype strd_size) const {

		return interleaved_tile_matrix_base {
			data_at(cntg_idx, strd_idx),
			cntg_interleave_,          strd_interleave_,
			cntg_size,                 strd_size,
			cntg_interleave_step_,     strd_interleave_step_,
			cntg_tile_step_,           strd_tile_step_
		};
	}
};

} //namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATRIX_INTERLEAVED_TILE_MATRIX_BASE_HPP
