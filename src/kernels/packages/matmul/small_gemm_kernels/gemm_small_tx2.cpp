/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "gemm_small_framework.hpp"
#include "gemm_small_tx2.hpp"
#include "sgemm_unrolled.hpp"
#include "perflibs_blas_types.hpp"
#include "perflibs_util.hpp"

extern "C" {

// The kernels to be used for this target.

// SGEMM
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_NN_n4m12k8_bet1;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_NN_n4m12k8_bet0;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_NN_n4m12k8;

perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_NT_n4m12k8_bet1;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_NT_n4m12k8_bet0;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_NT_n4m12k8;

perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_TN_n3m5k8_bet1;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_TN_n3m5k8_bet0;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_TN_n3m5k8;

perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_TT_m4n12k8_bet1;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_TT_m4n12k8_bet0;
perflibs::gemm::kfunc_t<float> sgemm_alt_kernel_TT_m4n12k8;
}


namespace perflibs::gemm {

// Parallel tiling and cache blocking functions {{{

// SGEMM {{{

// Autotuning overrides
kernel_inttype sgemm_m_tile_tx2 = 0;
kernel_inttype sgemm_n_tile_tx2 = 0;
kernel_inttype sgemm_k_tile_tx2 = 0;

//functions for 2d parallel decomposition used in small kernel case
void sgemm_set_m_tile_tx2(kernel_inttype val) { sgemm_m_tile_tx2 = val; }
void sgemm_set_n_tile_tx2(kernel_inttype val) { sgemm_n_tile_tx2 = val; }
void sgemm_set_k_tile_tx2(kernel_inttype val) { sgemm_k_tile_tx2 = val; }

// NN {{{
kernel_inttype sgemm_get_m_tile_nn_tx2(kernel_inttype mfact, kernel_inttype m) {
	if (sgemm_m_tile_tx2) {
		return sgemm_m_tile_tx2*mfact;
	}

	kernel_inttype value = 12;// 24 best for n = 1000, but doesn't scale down well to fewer threads - autotune!

	return value*mfact;
}
kernel_inttype sgemm_get_n_tile_nn_tx2(kernel_inttype nfact, kernel_inttype n) {
	if (sgemm_n_tile_tx2) {
		return sgemm_n_tile_tx2*nfact;
	}

	kernel_inttype value = 16;// 32 best for n = 1000, but doesn't scale down well to fewer threads - autotune!

	return value*nfact;
}
kernel_inttype sgemm_get_k_tile_nn_tx2(kernel_inttype kfact, kernel_inttype k) {
	if (sgemm_k_tile_tx2) {
		return sgemm_k_tile_tx2*kfact;
	}

	kernel_inttype value = 192; // kfact * cache block in this dim

	return value*kfact;
}
//}}}

// NT {{{
kernel_inttype sgemm_get_m_tile_nt_tx2(kernel_inttype mfact, kernel_inttype m) {
	if (sgemm_m_tile_tx2) {
		return sgemm_m_tile_tx2*mfact;
	}

	kernel_inttype value = 12;

	return value*mfact;
}
kernel_inttype sgemm_get_n_tile_nt_tx2(kernel_inttype nfact, kernel_inttype n) {
	if (sgemm_n_tile_tx2) {
		return sgemm_n_tile_tx2*nfact;
	}

	kernel_inttype value = 16;

	return value*nfact;
}
kernel_inttype sgemm_get_k_tile_nt_tx2(kernel_inttype kfact, kernel_inttype k) {
	if (sgemm_k_tile_tx2) {
		return sgemm_k_tile_tx2*kfact;
	}

	kernel_inttype value = 128;

	return value*kfact;
}
// }}}

// TN {{{
kernel_inttype sgemm_get_m_tile_tn_tx2(kernel_inttype mfact, kernel_inttype m) {
	if (sgemm_m_tile_tx2) {
		return sgemm_m_tile_tx2*mfact;
	}

	kernel_inttype value = 40;

	return value*mfact;
}
kernel_inttype sgemm_get_n_tile_tn_tx2(kernel_inttype nfact, kernel_inttype n) {
	if (sgemm_n_tile_tx2) {
		return sgemm_n_tile_tx2*nfact;
	}

	kernel_inttype value = 33;

	return value*nfact;
}
kernel_inttype sgemm_get_k_tile_tn_tx2(kernel_inttype kfact, kernel_inttype k) {
	if (sgemm_k_tile_tx2) {
		return sgemm_k_tile_tx2*kfact;
	}

	kernel_inttype value = 256;

	return value*kfact;
}
// }}}

// TT {{{
kernel_inttype sgemm_get_m_tile_tt_tx2(kernel_inttype mfact, kernel_inttype m) {
	if (sgemm_m_tile_tx2) {
		return sgemm_m_tile_tx2*mfact;
	}

	kernel_inttype value = 16;

	return value*mfact;
}
kernel_inttype sgemm_get_n_tile_tt_tx2(kernel_inttype nfact, kernel_inttype n) {
	if (sgemm_n_tile_tx2) {
		return sgemm_n_tile_tx2*nfact;
	}

	kernel_inttype value = 12;

	return value*nfact;
}
kernel_inttype sgemm_get_k_tile_tt_tx2(kernel_inttype kfact, kernel_inttype k) {
	if (sgemm_k_tile_tx2) {
		return sgemm_k_tile_tx2*kfact;
	}

	kernel_inttype value = 192;

	return value*kfact;
}
// }}}

// Autotuning overrides
kernel_inttype sgemm_c2_block_tx2 = 0;
kernel_inttype sgemm_c3_block_tx2 = 0;

//cache blocking sizes within 2d parallel decomposition
void sgemm_set_c2_block_tx2(kernel_inttype val) { sgemm_c2_block_tx2  = val; }
void sgemm_set_c3_block_tx2(kernel_inttype val) { sgemm_c3_block_tx2  = val; }

// NN {{{
kernel_inttype sgemm_get_c2_block_nn_tx2() {
	if (sgemm_c2_block_tx2) {
		return sgemm_c2_block_tx2;
	}

	return 60;
}
kernel_inttype sgemm_get_c3_block_nn_tx2() {
	if (sgemm_c3_block_tx2) {
		return sgemm_c3_block_tx2;
	}

	return 240;
}
// }}}

// NT {{{
kernel_inttype sgemm_get_c2_block_nt_tx2() {
	if (sgemm_c2_block_tx2) {
		return sgemm_c2_block_tx2;
	}

	return 12;
}
kernel_inttype sgemm_get_c3_block_nt_tx2() {
	if (sgemm_c3_block_tx2) {
		return sgemm_c3_block_tx2;
	}

	return 120;
}
// }}}

// TN {{{
kernel_inttype sgemm_get_c2_block_tn_tx2() {
	if (sgemm_c2_block_tx2) {
		return sgemm_c2_block_tx2;
	}

	return 24;
}
kernel_inttype sgemm_get_c3_block_tn_tx2() {
	if (sgemm_c3_block_tx2) {
		return sgemm_c3_block_tx2;
	}

	return 240;
}
// }}}

// TT {{{
kernel_inttype sgemm_get_c2_block_tt_tx2() {
	if (sgemm_c2_block_tx2) {
		return sgemm_c2_block_tx2;
	}

	return 84;
}
kernel_inttype sgemm_get_c3_block_tt_tx2() {
	if (sgemm_c3_block_tx2) {
		return sgemm_c3_block_tx2;
	}

	return 240;
}
// }}}
// }}}

// }}}

// Note: NOT using unordered_map and a static initializer in get_kernel() below: both require linking to libstdc++.
// Single precision kernels
static const struct simple_map<float> kernel_table_tx2_s = {

	// In the setup get_kernel calls below, where alpha or beta = 2.0 this is used in place of "Any value" as
	// described in the comments for the get_kernel_key function

	// NN kernels
	get_kernel_key(NN,nmk,12,4,8,1.0,1.0), &sgemm_alt_kernel_NN_n4m12k8_bet1,
	get_kernel_key(NN,nmk,12,4,8,2.0,1.0), &sgemm_alt_kernel_NN_n4m12k8_bet1,
	get_kernel_key(NN,nmk,12,4,8,1.0,0.0), &sgemm_alt_kernel_NN_n4m12k8_bet0,
	get_kernel_key(NN,nmk,12,4,8,2.0,0.0), &sgemm_alt_kernel_NN_n4m12k8_bet0,
	get_kernel_key(NN,nmk,12,4,8,1.0,2.0), &sgemm_alt_kernel_NN_n4m12k8,
	get_kernel_key(NN,nmk,12,4,8,2.0,2.0), &sgemm_alt_kernel_NN_n4m12k8,

	// NT kernels
	get_kernel_key(NT,nmk,12,4,8,1.0,1.0), &sgemm_alt_kernel_NT_n4m12k8_bet1,
	get_kernel_key(NT,nmk,12,4,8,2.0,1.0), &sgemm_alt_kernel_NT_n4m12k8_bet1,
	get_kernel_key(NT,nmk,12,4,8,1.0,0.0), &sgemm_alt_kernel_NT_n4m12k8_bet0,
	get_kernel_key(NT,nmk,12,4,8,2.0,0.0), &sgemm_alt_kernel_NT_n4m12k8_bet0,
	get_kernel_key(NT,nmk,12,4,8,1.0,2.0), &sgemm_alt_kernel_NT_n4m12k8,
	get_kernel_key(NT,nmk,12,4,8,2.0,2.0), &sgemm_alt_kernel_NT_n4m12k8,

	// TN kernels
	get_kernel_key(TN,nmk,5,3,8,1.0,1.0), &sgemm_alt_kernel_TN_n3m5k8_bet1,
	get_kernel_key(TN,nmk,5,3,8,2.0,1.0), &sgemm_alt_kernel_TN_n3m5k8_bet1,
	get_kernel_key(TN,nmk,5,3,8,1.0,0.0), &sgemm_alt_kernel_TN_n3m5k8_bet0,
	get_kernel_key(TN,nmk,5,3,8,2.0,0.0), &sgemm_alt_kernel_TN_n3m5k8_bet0,
	get_kernel_key(TN,nmk,5,3,8,1.0,2.0), &sgemm_alt_kernel_TN_n3m5k8,
	get_kernel_key(TN,nmk,5,3,8,2.0,2.0), &sgemm_alt_kernel_TN_n3m5k8,

	// TT kernels
	get_kernel_key(TT,mnk,4,12,8,1.0,1.0), &sgemm_alt_kernel_TT_m4n12k8_bet1,
	get_kernel_key(TT,mnk,4,12,8,2.0,1.0), &sgemm_alt_kernel_TT_m4n12k8_bet1,
	get_kernel_key(TT,mnk,4,12,8,1.0,0.0), &sgemm_alt_kernel_TT_m4n12k8_bet0,
	get_kernel_key(TT,mnk,4,12,8,2.0,0.0), &sgemm_alt_kernel_TT_m4n12k8_bet0,
	get_kernel_key(TT,mnk,4,12,8,1.0,2.0), &sgemm_alt_kernel_TT_m4n12k8,
	get_kernel_key(TT,mnk,4,12,8,2.0,2.0), &sgemm_alt_kernel_TT_m4n12k8,
};

template <>
kfunc_t<float> *get_kernel_tx2<float>(trans_t trans_opt, l_order_t loop_o, kernel_inttype rblock_m, kernel_inttype rblock_n, kernel_inttype rblock_k, float alpha, float beta) {

	return kernel_table_tx2_s[get_kernel_key(trans_opt, loop_o, rblock_m, rblock_n, rblock_k, alpha, beta)];

}


void sgemm_small_tx2(
	const kernel_inttype max_threads,
	const perflibs_trans transa, const perflibs_trans transb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
	float alpha, const float *a, kernel_inttype lda,
	const float *b, kernel_inttype ldb, float beta,
	float *c, kernel_inttype ldc) {

	if (!is_trans(transa) && !is_trans(transb)) {
		sgemm_small_nn_tx2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		return;
	}

	struct gemm_small_options<float> options;

	options.gemm_small_recursive = &gemm_small_driver<float>;

	options.max_threads = max_threads;

	// Functions that will return elements of the arrays, to be set according to transpose options. {{{
	if (!is_trans(transa)) {
		options.get_Aij = [] (l_order_t order, kernel_inttype lda, kernel_inttype dim1, kernel_inttype dim2, kernel_inttype dim3) {
			switch(order) {
				case nmk :
					return lda*dim3 + dim2;
				case mnk :
					return lda*dim3 + dim1;
				case nkm :
					return lda*dim2 + dim3;
				case knm :
					return lda*dim1 + dim3;
				case mkn :
					return lda*dim2 + dim1;
				case kmn :
					return lda*dim1 + dim2;
			}
			return kernel_inttype(0);
		};
	}
	else {
		options.get_Aij = [] (l_order_t order, kernel_inttype lda, kernel_inttype dim1, kernel_inttype dim2, kernel_inttype dim3) {
			switch (order) {
				case nmk :
					return lda*dim2 + dim3;
				case mnk :
					return lda*dim1 + dim3;
				case nkm :
					return lda*dim3 + dim2;
				case knm :
					return lda*dim3 + dim1;
				case mkn :
					return lda*dim1 + dim2;
				case kmn :
					return lda*dim2 + dim1;
			}
			return kernel_inttype(0);
		};
	}

	if (!is_trans(transb)) {
		options.get_Bij = [] (l_order_t order, kernel_inttype ldb, kernel_inttype dim1, kernel_inttype dim2, kernel_inttype dim3) {
			switch (order) {
				case nmk :
					return ldb*dim1 + dim3;
				case mnk :
					return ldb*dim2 + dim3;
				case nkm :
					return ldb*dim1 + dim2;
				case knm :
					return ldb*dim2 + dim1;
				case mkn :
					return ldb*dim3 + dim2;
				case kmn :
					return ldb*dim3 + dim1;
			}
			return kernel_inttype(0);
		};
	}
	else  {
		options.get_Bij = [] (l_order_t order, kernel_inttype ldb, kernel_inttype dim1, kernel_inttype dim2, kernel_inttype dim3) {
			switch (order) {
				case nmk :
					return ldb*dim3 + dim1;
				case mnk :
					return ldb*dim3 + dim2;
				case nkm :
					return ldb*dim2 + dim1;
				case knm :
					return ldb*dim1 + dim2;
				case mkn :
					return ldb*dim2 + dim3;
				case kmn :
					return ldb*dim1 + dim3;
			}
			return kernel_inttype(0);
		};
	}
	// }}}

	// OPTION: NN {{{
	if (!is_trans(transa) && !is_trans(transb)) {

		options.trans_opt = NN;
		options.loop_o = nmk;
		options.gemm_unrolled = &sgemm_unrolled_NN;

		// reg blocking factors
		options.mfact = 12;
		options.nfact = 4;
		options.kfact = 8;

		options.get_tile_m = &sgemm_get_m_tile_nn_tx2;
		options.get_tile_n = &sgemm_get_n_tile_nn_tx2;
		options.get_tile_k = &sgemm_get_k_tile_nn_tx2;

		options.get_cblock_2 = &sgemm_get_c2_block_nn_tx2;
		options.get_cblock_3 = &sgemm_get_c3_block_nn_tx2;

	}
	// }}} // end of option NN

	// OPTION: NT {{{
	else if (!is_trans(transa) && is_trans(transb)) {

		options.trans_opt = NT;
		options.loop_o = nmk;
		options.gemm_unrolled = &sgemm_unrolled_NT;

		// reg blocking factors
		options.mfact = 12;
		options.nfact = 4;
		options.kfact = 8;

		options.get_tile_m = &sgemm_get_m_tile_nt_tx2;
		options.get_tile_n = &sgemm_get_n_tile_nt_tx2;
		options.get_tile_k = &sgemm_get_k_tile_nt_tx2;

		options.get_cblock_2 = &sgemm_get_c2_block_nt_tx2;
		options.get_cblock_3 = &sgemm_get_c3_block_nt_tx2;

	}
	// }}} // end of option NT

	// OPTION: TN {{{
	else if (is_trans(transa) && !is_trans(transb)) {

		options.trans_opt = TN;
		options.loop_o = nmk;
		options.gemm_unrolled = &sgemm_unrolled_TN;

		// reg blocking factors
		options.mfact = 5;
		options.nfact = 3;
		options.kfact = 8;

		options.get_tile_m = &sgemm_get_m_tile_tn_tx2;
		options.get_tile_n = &sgemm_get_n_tile_tn_tx2;
		options.get_tile_k = &sgemm_get_k_tile_tn_tx2;

		options.get_cblock_2 = &sgemm_get_c2_block_tn_tx2;
		options.get_cblock_3 = &sgemm_get_c3_block_tn_tx2;

	}
	// }}} // end of option TN

	// OPTION: TT {{{
	else if (is_trans(transa) && is_trans(transb)) {

		options.trans_opt = TT;
		options.loop_o = mnk;
		options.gemm_unrolled = &sgemm_unrolled_TT;

		// reg blocking factors
		options.mfact = 4;
		options.nfact = 12;
		options.kfact = 8;

		options.get_tile_m = &sgemm_get_m_tile_tt_tx2;
		options.get_tile_n = &sgemm_get_n_tile_tt_tx2;
		options.get_tile_k = &sgemm_get_k_tile_tt_tx2;

		options.get_cblock_2 = &sgemm_get_c2_block_tt_tx2;
		options.get_cblock_3 = &sgemm_get_c3_block_tt_tx2;

	}
	// }}} // end of option TT

	gemm_small_driver<float>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, options);

}


} //namespace perflibs gemm

extern "C" void perflibs_sgemm_set_m_tile_tx2(kernel_inttype val) { perflibs::gemm::sgemm_set_m_tile_tx2(val); }
extern "C" void perflibs_sgemm_set_n_tile_tx2(kernel_inttype val) { perflibs::gemm::sgemm_set_n_tile_tx2(val); }
extern "C" void perflibs_sgemm_set_k_tile_tx2(kernel_inttype val) { perflibs::gemm::sgemm_set_k_tile_tx2(val); }

extern "C" void perflibs_sgemm_set_c2_block_tx2(kernel_inttype val) { perflibs::gemm::sgemm_set_c2_block_tx2(val); }
extern "C" void perflibs_sgemm_set_c3_block_tx2(kernel_inttype val) { perflibs::gemm::sgemm_set_c3_block_tx2(val); }
