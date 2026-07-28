/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "gemm_small_framework.hpp"
#include "perflibs_assert.hpp"
#include "perflibs_util.hpp"

namespace perflibs::gemm {

/**
 * This is the hashing function used to allow kernels of different types (as given
 * by the function parameters) to be placed in a unique position in a lookup table.
 */
template<typename FloatType>
int get_kernel_key(trans_t trans_opt, l_order_t loop_o, uint64_t rblock_m, uint64_t rblock_n, uint64_t rblock_k, FloatType alpha, FloatType beta) {
	// Each rblock has an upper bound of 32
	/*
	 * Trans   rblock_m    rblock_n    rblock_k    inner_dim	Key
	 * NN      1           1           1           k          1
	 * NN      1           1           2           k          2
	 * ...
	 * NN      1           1           32          k          32
	 * NN      1           2           1           k          33
	 * NN      1           2           2           k          34
	 * ...
	 * NN      1           2           32          k          64
	 * ...
	 * NN      1           32          1           k          (rblock_n-1)*32 + rblock_k
	 * ...
	 * NN      1           32          32          k          1024
	 * NN      2           1           1           k          1025
	 * ...
	 * NN      32          32          32          k          32768 (rblock_m-1)*1024 + (rblock_n-1)*32 + rblock_k
	 * Repeat for NN with m as inner dim (32769 - 65536)
	 * Repeat for NN with n as inner dim (65537 - 98304)
	 * Any Alpha && Any Beta (0 - 3*32768 - 1)
	 * Alpha = 1 && Beta = 0 (3*32768 - 6*32768 - 1)
	 * Alpha = 1 && Beta = 1 (6*32768 - 9*32768 - 1)
	 * Alpha = 1 && Any Beta (9*32768 - 12*32768 -1)
	 * Any Alpha && Beta = 0 (12*32768 - 15*32768 - 1)
	 * Any Alpha && Beta = 1 (15*32768 - 18*32768 - 1)
	 *
	 * Note that Alpha = 0 is not handled separately here. This case is handled in the driver before the call to
	 * the small function.
	 *
	 * Then repeat for NT, TN, TT, giving 4*18*32768 combinations.
	 *
	 */

	PERFLIBS_ASSERT(rblock_m<=32 && rblock_n<=32 && rblock_k<=32);

	int alba = 0;
	if (alpha==FloatType(1.0) && beta==FloatType(0.0))
		alba = 3*32768;
	else if (alpha==FloatType(1.0) && beta==FloatType(1.0))
		alba = 6*32768;
	else if (alpha==FloatType(1.0))
		alba = 9*32768;
	else if (beta==FloatType(0.0))
		alba = 12*32768;
	else if (beta==FloatType(1.0))
		alba = 15*32768;

	int key = trans_opt + alba + int(loop_o >> 1) + int((rblock_m-1)*1024) + int((rblock_n-1)*32) + int(rblock_k);
#if 0
	fprintf(stderr,"\n%s: input:\n", __func__);
	fprintf(stderr,"%s: trans_opt = %d:\n", __func__, trans_opt);
	fprintf(stderr,"%s: loop_o = %d:\n", __func__, loop_o);
	fprintf(stderr,"%s: rblock_m = %zu:\n", __func__, rblock_m);
	fprintf(stderr,"%s: rblock_n = %zu:\n", __func__, rblock_n);
	fprintf(stderr,"%s: rblock_k = %zu:\n", __func__, rblock_k);
	fprintf(stderr,"%s: alpha = %f:\n", __func__, alpha);
	fprintf(stderr,"%s: beta = %f:\n", __func__, beta);
	fprintf(stderr,"%s: output:\n", __func__);
	fprintf(stderr,"%s: trans_opt = %d\n", __func__, trans_opt);
	fprintf(stderr,"%s: loop_o = %d\n", __func__, (loop_o >> 1));
	fprintf(stderr,"%s: alba = %d\n", __func__, alba);
	fprintf(stderr,"%s: rblock_m offset = %d\n", __func__, int((rblock_m-1)*1024));
	fprintf(stderr,"%s: rblock_n offset = %d\n", __func__, int((rblock_n-1)*32));
	fprintf(stderr,"%s: rblock_k offset = %d\n", __func__, int(rblock_k));
	fprintf(stderr,"%s: return: %d\n", __func__, key);
#endif
	return key;
}

template int get_kernel_key(trans_t trans_opt, l_order_t loop_o, uint64_t rblock_m, uint64_t rblock_n, uint64_t rblock_k, double alpha, double beta);
template int get_kernel_key(trans_t trans_opt, l_order_t loop_o, uint64_t rblock_m, uint64_t rblock_n, uint64_t rblock_k, float alpha, float beta);



/**
 * Accessor function for matrix C.
 */
kernel_inttype get_Cij(l_order_t order, kernel_inttype ldc, kernel_inttype dim1, kernel_inttype dim2, kernel_inttype dim3) {
	switch(order) {
		case nmk :
			return ldc*dim1 + dim2;
		case mnk :
			return ldc*dim2 + dim1;
		case nkm :
			return ldc*dim1 + dim3;
		case knm :
			return ldc*dim2 + dim3;
		case mkn :
			return ldc*dim3 + dim1;
		case kmn :
			return ldc*dim3 + dim2;
	}
	return kernel_inttype(0);
}



/**
 * This function implements the triple-nested loop structure of a gemm. The loop
 * ordering, transpose options, cache blocking are provided as parameters to the function.
 *
 * Loop over the 3 dimensions in the order given by LO such that if e.g.
 * LO = nmk then dim1 = n, dim2 = m, dim3 = k,
 * LO = nkm then dim1 = n, dim2 = k, dim3 = m.
 */
template <typename FloatType>
void gemm_small(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, FloatType alpha, const FloatType *A, kernel_inttype lda, const FloatType *B, kernel_inttype ldb, FloatType *C, kernel_inttype ldc, FloatType beta, const struct gemm_small_options<FloatType> &options) {

	kernel_inttype cacheb2 = options.get_cblock_2();
	kernel_inttype cacheb3 = options.get_cblock_3();
	const trans_t t_opt = options.trans_opt;
	const l_order_t l_order  = options.loop_o;
	const kernel_inttype rb_m = options.mfact;
	const kernel_inttype rb_n = options.nfact;
	const kernel_inttype rb_k = options.kfact;

	// Which level is the k-dimension?
	int kdim;
	// Dimensions that need to be set after processing the loop ordering
	kernel_inttype dim1, dim2, dim3, rb_dim1, rb_dim2, rb_dim3;

	if (l_order == nmk) {
		dim1 = n;
		dim2 = m;
		dim3 = k;
		rb_dim2 = rb_m;
		rb_dim1 = rb_n;
		rb_dim3 = rb_k;
		kdim = 3;
	}
	else if (l_order == mnk) {
		dim1 = m;
		dim2 = n;
		dim3 = k;
		rb_dim1 = rb_m;
		rb_dim2 = rb_n;
		rb_dim3 = rb_k;
		kdim = 3;
	}
	else if (l_order == nkm) {
		dim1 = n;
		dim2 = k;
		dim3 = m;
		rb_dim3 = rb_m;
		rb_dim1 = rb_n;
		rb_dim2 = rb_k;
		kdim = 2;
	}
	else if (l_order == knm) {
		dim1 = k;
		dim2 = n;
		dim3 = m;
		rb_dim3 = rb_m;
		rb_dim2 = rb_n;
		rb_dim1 = rb_k;
		kdim = 1;
	}
	else if (l_order == mkn) {
		dim1 = m;
		dim2 = k;
		dim3 = n;
		rb_dim1 = rb_m;
		rb_dim3 = rb_n;
		rb_dim2 = rb_k;
		kdim = 2;
	}
	else if (l_order == kmn) {
		dim1 = k;
		dim2 = m;
		dim3 = n;
		rb_dim2 = rb_m;
		rb_dim3 = rb_n;
		rb_dim1 = rb_k;
		kdim = 1;
	}
	else {
		PERFLIBS_ASSERT(0);
		return;
	}

	PERFLIBS_ASSERT(dim1%rb_dim1==0);
	PERFLIBS_ASSERT(dim2%rb_dim2==0);
	PERFLIBS_ASSERT(dim3%rb_dim3==0);

	// Make sure the cache blocking sizes are multiples of the register block sizes
	cacheb2 = std::max(cacheb2 - cacheb2%rb_dim2, rb_dim2);
	cacheb3 = std::max(cacheb3 - cacheb3%rb_dim3, rb_dim3);

	FloatType beta2, beta3, beta4;
	for (kernel_inttype dim3_i=0; dim3_i<dim3; dim3_i+=cacheb3) {

		// If dim3 is k loop then we need to check whether or not beta should be set to 1.0
		beta2 = (kdim==3 && dim3_i>0) ? 1.0 : beta;

		for (kernel_inttype dim2_i=0; dim2_i<dim2; dim2_i+=cacheb2) {

			auto dim2_cb = std::min(cacheb2, dim2 - dim2_i);
			auto dim3_cb = std::min(cacheb3, dim3 - dim3_i);

			// Pointers into the current cache block
			const FloatType *AA = &A[options.get_Aij(l_order, lda, 0, dim2_i, dim3_i)];
			const FloatType *BB = &B[options.get_Bij(l_order, ldb, 0, dim2_i, dim3_i)];
			FloatType *CC = &C[get_Cij(l_order, ldc, 0, dim2_i, dim3_i)];

			// Work within a cache block - kernel only iterates over dim3 (cache block)
			for (kernel_inttype dim1_ii=0; dim1_ii<dim1; dim1_ii+=rb_dim1) {

				// If dim1 is k loop then we need to check whether or not beta should be set to 1.0
				beta3 = (kdim==1 && dim1_ii>0) ? 1.0 : beta2;

				for (kernel_inttype dim2_ii=0; dim2_ii<dim2_cb; dim2_ii+=rb_dim2) {

					// If dim2 is k loop then we need to check whether or not beta should be set to 1.0
					beta4 = (kdim==2 && (dim2_i>0 || dim2_ii>0)) ? 1.0 : beta3;

					// Look up required kernel based on problem parameters
					auto kernel = get_kernel<FloatType>(t_opt, l_order, rb_m, rb_n, rb_k, alpha, beta4);
					PERFLIBS_ASSERT(kernel!=NULL);

					// Execute the kernel, which is the inner loop
					kernel(dim3_cb, &AA[options.get_Aij(l_order, lda, dim1_ii, dim2_ii, 0)], lda,
						&BB[options.get_Bij(l_order, ldb, dim1_ii, dim2_ii, 0)], ldb,
						&CC[get_Cij(l_order, ldc, dim1_ii, dim2_ii, 0)], ldc, alpha, beta4);

				}
			}

		}
	}
}


/**
 * Carve up, decide on parallel decomposition, and dispatch to the triple-nested loop function gemm_small.
 */
template<typename FloatType>
void gemm_small_driver(int_type nota, int_type notb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
	FloatType alpha, const FloatType *a, kernel_inttype lda,
	const FloatType *b, kernel_inttype ldb, FloatType beta,
	FloatType *c, kernel_inttype ldc, const struct gemm_small_options<FloatType> &options) {

	const kernel_inttype mfact = options.mfact;
	const kernel_inttype nfact = options.nfact;
	const kernel_inttype kfact = options.kfact;

	const kernel_inttype mblock = std::max(m - m%mfact, m%mfact);
	const kernel_inttype nblock = std::max(n - n%nfact, n%nfact);
	const kernel_inttype kblock = std::max(k - k%kfact, k%kfact);

	int nt = options.max_threads;

#ifdef _OPENMP
	// {{{ When multithreading divide the work into tiles.
	kernel_inttype tile_m, tile_n, tile_k;
	if (nt > 1) {

		// The starting tile sizes, to be adjusted down appropriately: tunable parameters.
		tile_m = options.get_tile_m(mfact, m);
		tile_n = options.get_tile_n(nfact, n);
		tile_k = options.get_tile_k(kfact, k);

		// How many threads can we use to divide up m?
		int nt_m = m/tile_m;
		nt_m = m%tile_m ? nt_m + 1 : nt_m;

		// How many threads can we use to divide up n?
		int nt_n = n/tile_n;
		nt_n = n%tile_n ? nt_n + 1 : nt_n;

		int nt_nm = nt_m*nt_n;

		// Adjust number of threads in n, don't give any thread less than 50 elements in this dim
		while (nt_nm < nt && tile_n > 50) {

			// Decrease tile size in n first
			tile_n -= nfact;

			// How many threads can we use to divide up n?
			nt_n = n/tile_n;
			nt_n = n%tile_n ? nt_n + 1 : nt_n;

			// How many threads does this give us?
			nt_nm = nt_n*nt_m;
		}

		// Adjust number of threads in m, don't give any thread less than 50 elements in this dim
		while (nt_nm < nt && tile_m > 50) {

			// Decrease tile size in m
			tile_m -= mfact;

			// How many threads can we use to divide up m?
			nt_m = m/tile_m;
			nt_m = m%tile_m ? nt_m + 1 : nt_m;

			// How many threads does this give us?
			nt_nm = nt_n*nt_m;
		}

		// Reduce the number of threads to use if necessary
		if (nt > nt_nm)
			nt = nt_nm;
	} //}}}
#endif

//{{{ Debug print
#if 0
	printf("mblock = %zu\n", mblock);
	printf("nblock = %zu\n", nblock);
	printf("kblock = %zu\n", kblock);
	if (nt > 1)
		printf("nt = %d   tile_m = %zu   tile_n = %zu   tile_k = %zu\n", nt, tile_m, tile_n, tile_k);
	printf("------------------------------\n");
#endif
//}}}

	// Serial case {{{
	if (nt<2) {
		// Iterate over the largest problem that fits in the register blocks
		for (kernel_inttype ni=0; ni<n; ni+=nblock) {
			for (kernel_inttype mi=0; mi<m; mi+=mblock) {
				for (kernel_inttype ki=0; ki<k; ki+=kblock) {

					const kernel_inttype mm = std::min(mblock, m - mi);
					const kernel_inttype nn = std::min(nblock, n - ni);
					const kernel_inttype kk = std::min(kblock, k - ki);

					FloatType beta2 = ki==0 ? beta : 1.0;

					// Pointers into the block
					const FloatType *a1 = &a[options.get_Aij(mnk, lda, mi, ni, ki)];
					const FloatType *b1 = &b[options.get_Bij(mnk, ldb, mi, ni, ki)];
					FloatType *c1 = &c[get_Cij(mnk, ldc, mi, ni, ki)];

					if (mm%mfact==0 && nn%nfact==0 && kk%kfact==0) {
						gemm_small(
							mm, nn, kk,
							alpha, a1, lda, b1, ldb, c1, ldc, beta2,
							options);
					}
					else {
						options.gemm_unrolled(mm, nn, kk, alpha, a1, lda, b1, ldb, beta2, c1, ldc);
					}
				}
			}
		}
		return;
	}
	// }}}

	// Parallel case {{{
#ifdef _OPENMP

	#pragma omp parallel default(none) \
		shared(nota, notb, n, m, k, a, lda, b, ldb, c, ldc, alpha, beta, nt, tile_m, tile_n, tile_k) \
		shared(options) \
		firstprivate(mfact, nfact, kfact) \
		num_threads(nt)
	{

		// Parallelise by tiles

		/*
		 * Use static scheduling here to avoid
		 * assertion failures with LLVM OpenMP versions <= 15
		 */
		//#pragma omp for schedule(dynamic,1) collapse(2)
		#pragma omp for schedule(static,1) collapse(2)
		for (kernel_inttype ii=0; ii<n; ii+=tile_n) {
			for (kernel_inttype ji=0; ji<m; ji+=tile_m) {
				for (kernel_inttype ki=0; ki<k; ki+=tile_k) {

					const kernel_inttype mm = std::min(tile_m, int(m) - ji);
					const kernel_inttype nn = std::min(tile_n, int(n) - ii);
					const kernel_inttype kk = std::min(tile_k, int(k) - ki);

					FloatType beta2 = ki==0 ? beta : 1.0;

					// Pointers into the tile
					const FloatType *a1 = &a[options.get_Aij(nmk, lda, ii, ji, ki)];
					const FloatType *b1 = &b[options.get_Bij(nmk, ldb, ii, ji, ki)];
					FloatType *c1 = &c[get_Cij(nmk, ldc, ii, ji, ki)];

					if (mm%mfact==0 && nn%nfact==0 && kk%kfact==0) {
						gemm_small(
							mm, nn, kk,
							alpha, a1, lda, b1, ldb, c1, ldc,
							beta2, options);
					}
					else {
						// Recursive call here, where since it will be nested it will go into the serial section
						// and the problem will be blocked properly to minimize use of the unrolled kernels.
						options.gemm_small_recursive(nota, notb, mm, nn, kk, alpha, a1, lda, b1, ldb,
							beta2, c1, ldc, options);
					}

				}
			}
		}

	}
#endif
	// }}}

}


template void gemm_small_driver(int_type nota, int_type notb, kernel_inttype m, kernel_inttype n, kernel_inttype k,
	float alpha, const float *a, kernel_inttype lda,
	const float *b, kernel_inttype ldb, float beta,
	float *c, kernel_inttype ldc, const struct gemm_small_options<float> &options);

} // end namespaces
