/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_MATMUL_FWD_HPP
#define PERFLIBS_LINALG_MATMUL_FWD_HPP

#include "packages/matmul/problem_context_bases.hpp"
#include "spec/problem_context.hpp"
#include "framework/compute.hpp"

//DO NOT include strategies.hpp or anything from the strategies dir

namespace perflibs::linalg {

namespace matmul {

class set_or_scale;
class symmetric_matrix_vector;
class compressed_symmetric_matrix_vector;
class compressed_rank_one_update;
class matadd2_buffer_naive;
class matadd3_naive;
class matadd3_zero_scalar;
class rank_k_update_large;
class rank_k_update_basic;
class rank_one_update;

class matmul3_atomic;
class matmul3_inner_product;
class matmul3_vector_scalar;
class matmul3_matrix_vector;
class matmul3_outer_product;
class matmul3_outer_product_beta_one;
class matmul3_unpacked;
class matmul3_interleaved_basic;
class matmul3_interleaved_sequential;


class matmul3_gemm_reference;
class matmul3_symm_hemm_l_reference;
class matmul3_symm_hemm_r_reference;

template<bool SpawnThreads>
class matmul3_interleaved_large;
template<bool UseAccelerator>
class matmul3_interleaved_large_no_sync;

class matmul4_vector_scalar_strategy;

class backstop;
class compressed_rank_two_update;
class rank_update_generic;
class rank_two_update;
class rank_update_2k_interleaved;
class inplace_matmul_vector;
class inplace_matmul_large;
class compressed_triangular_matrix_vector;
class compressed_general_matrix_vector;
class herm_rank1_strategy;



class matmul3_interleaved_tri_gen_gen;
class matmul3_interleaved_gen_tri_gen;

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<matmul::matmul2<T...>, ArchitectureSpec>& pctx);

template<typename AType, typename BType, typename CType, typename ScalarType, typename ArchitectureSpec>
void compute(const spec::problem_context<
	matmul::matmul3<
		general_matrix<matrix_base<const AType>>,
		general_matrix<matrix_base<const BType>>,
		general_matrix<matrix_base<      CType>>,
		ScalarType
	>,
	ArchitectureSpec>& pctx);


template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<matmul::matmul3<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<matmul::matmul4<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<matmul::rank_update_2k<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<matmul::matadd2<T...>, ArchitectureSpec>& pctx);

template<typename... T, typename ArchitectureSpec>
void compute(const spec::problem_context<matmul::matadd3<T...>, ArchitectureSpec>& pctx);

struct matmul2_axpby_kernel_map {
	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return 1_ki;
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.b.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return typename ProblemContext::scalar_type { 1.1 };
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::b_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
};

} // namespace matmul


template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::inplace_matmul_vector>>
: matmul::matmul2_axpby_kernel_map {
	axpby_kernel_tag(const spec::strategy_tag<matmul::inplace_matmul_vector>&) { }
};


template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_triangular_matrix_vector>>
: matmul::matmul2_axpby_kernel_map {
	axpby_kernel_tag(const spec::strategy_tag<matmul::compressed_triangular_matrix_vector>&) { }
};


template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::matmul3_outer_product>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::matmul3_outer_product>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return 1_ki;
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return typename ProblemContext::scalar_type { 1.1 };
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return pctx.beta;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_physical() && is_strd_contig(pctx.a) && pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::matmul3_outer_product>

template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::matmul3_outer_product_beta_one>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::matmul3_outer_product_beta_one>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return 1_ki;
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return typename ProblemContext::scalar_type { 1.1 };
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_physical() && is_strd_contig(pctx.a) && pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::matmul3_outer_product_beta_one>

template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::rank_two_update>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::rank_two_update>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return pctx.alpha;
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return pctx.beta;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return false;
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::rank_two_update>

template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_general_matrix_vector>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::compressed_general_matrix_vector>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return pctx.alpha;
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_general_matrix_vector>

template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_rank_one_update>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::compressed_rank_one_update>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return pctx.alpha;
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_rank_one_update>

template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_symmetric_matrix_vector>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::compressed_symmetric_matrix_vector>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return pctx.alpha;
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::compressed_symmetric_matrix_vector>

template<>
struct axpby_kernel_tag<spec::strategy_tag<matmul::symmetric_matrix_vector>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::symmetric_matrix_vector>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return pctx.a.strd_step();
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return typename ProblemContext::scalar_type { 1.1 };
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::scalar_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return pctx.a.is_conj();
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      std::remove_const_t<typename ProblemContext::a_matrix_type::value_type>>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::scalar_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul::symmetric_matrix_vector>

template<bool Spawn>
struct axpby_kernel_tag<spec::strategy_tag<matmul::matmul3_interleaved_large<Spawn>>> {
	axpby_kernel_tag(const spec::strategy_tag<matmul::matmul3_interleaved_large<Spawn>>&) { }

	template<typename ProblemContext>
	auto incx(const ProblemContext& pctx) const {
		return 1_ki;
	}

	template<typename ProblemContext>
	auto incy(const ProblemContext& pctx) const {
		return pctx.c.cntg_step();
	}

	template<typename ProblemContext>
	auto alpha(const ProblemContext& pctx) const {
		return one<typename ProblemContext::c_matrix_type::value_type>;
	}

	template<typename ProblemContext>
	auto beta(const ProblemContext& pctx) const {
		return one<typename ProblemContext::c_matrix_type::value_type>;
	}

	template<typename ProblemContext>
	auto is_conj(const ProblemContext& pctx) const {
		return false;
	}

	template<typename XType, typename YType, typename ScalarType, typename ProblemContext>
	constexpr static bool match() {
		return std::is_same_v<XType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<YType,      typename ProblemContext::c_matrix_type::value_type>
		    && std::is_same_v<ScalarType, typename ProblemContext::c_matrix_type::value_type>;
	}
}; // struct axpby_kernel_tag<spec::strategy_tag<matmul3_interleaved_large<Spawn>>

} // namespace perflibs::linalg

#endif //PERFLIBS_LINALG_MATMUL_FWD_HPP
