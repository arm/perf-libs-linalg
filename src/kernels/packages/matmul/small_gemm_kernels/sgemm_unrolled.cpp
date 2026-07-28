/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#include "perflibs_blas_types.hpp"
#include "sgemm_unrolled_NN_nmk.hpp"
#include "sgemm_unrolled_NN_nkm.hpp"
#include "sgemm_unrolled_NT_nmk.hpp"
#include "sgemm_unrolled_NT_nkm.hpp"
#include "sgemm_unrolled_TN_nmk.hpp"
#include "sgemm_unrolled_TN_nkm.hpp"
#include "sgemm_unrolled_TT_mnk.hpp"
#include "sgemm_unrolled_TT_mkn.hpp"
#include <cstddef>
#include <array>

namespace perflibs { namespace gemm {

using fptr = decltype(&unrolled_kernel_NN_nmk<1,1,1>);

//{{{ NN
constexpr int max_unrollNN = 4;

template<typename ArrayType, int fid, kernel_inttype I_max, kernel_inttype I=0>
struct populateNN {
	void operator() (ArrayType& array) const {
		constexpr int unrolln = I/(max_unrollNN*max_unrollNN) + 1;
		constexpr int unrollm = ((I/max_unrollNN)%max_unrollNN) + 1;
		constexpr int unrollk = I%max_unrollNN + 1;
		array[I]= fid ? &unrolled_kernel_NN_nkm<unrolln, unrollm, unrollk> : &unrolled_kernel_NN_nmk<unrolln, unrollm, unrollk>;
		populateNN<ArrayType, fid, I_max, I+1>()(array);
	}
};
template<typename ArrayType, int fid, kernel_inttype I_max>
struct populateNN<ArrayType, fid, I_max, I_max> {
	void operator() (ArrayType& array) const {	}
};

void sgemm_unrolled_NN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc) {

	int k_notset = 1;
	int m_notset = 1;
	int n_notset = 1;

	int iunrollk = 1;
	int iunrollm = 1;
	int iunrolln = 1;

	int mquarter = m>>2;

	for (int i = max_unrollNN; i > 0; i--) {
		if (k_notset && k%i==0) {
			iunrollk = i;
			k_notset = 0;
		}

		if (n_notset && n%i==0) {
			iunrolln = i;
			n_notset = 0;
		}

		if (m_notset && (mquarter)%i==0) {
			iunrollm = i;
			m_notset = 0;
		}
	}

	if (mquarter==0)
		iunrollm = 1;

	iunrolln--;
	iunrollm--;
	iunrollk--;

	const int max_unrollNN_cubed = max_unrollNN*max_unrollNN*max_unrollNN;

	std::array<fptr, max_unrollNN_cubed> look_up_nmk;
	std::array<fptr, max_unrollNN_cubed> look_up_nkm;

	int lt_index = max_unrollNN*max_unrollNN*iunrolln + max_unrollNN*iunrollm + iunrollk;

	populateNN<decltype(look_up_nmk), 0, max_unrollNN_cubed>()(look_up_nmk);
	populateNN<decltype(look_up_nkm), 1, max_unrollNN_cubed>()(look_up_nkm);

	if (m > 7 && k < 4) {
		//printf("nkm iunrolln = %d\n",iunrolln+1);
		//printf("nkm iunrollm = %d\n",iunrollm+1);
		//printf("nkm iunrollk = %d\n",iunrollk+1);
		look_up_nkm[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	else {
		//printf("nmk iunrolln = %d\n",iunrolln+1);
		//printf("nmk iunrollm = %d\n",iunrollm+1);
		//printf("nmk iunrollk = %d\n",iunrollk+1);
		look_up_nmk[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

}
//}}}

//{{{ NT
constexpr int max_unrollNT = 4;

template<typename ArrayType, int fid, kernel_inttype I_max, kernel_inttype I=0>
struct populateNT {
	void operator() (ArrayType& array) const {
		constexpr int unrolln = I/(max_unrollNT*max_unrollNT) + 1;
		constexpr int unrollm = ((I/max_unrollNT)%max_unrollNT) + 1;
		constexpr int unrollk = I%max_unrollNT + 1;
		array[I]= fid ? &unrolled_kernel_NT_nkm<unrolln, unrollm, unrollk> : &unrolled_kernel_NT_nmk<unrolln, unrollm, unrollk>;
		populateNT<ArrayType, fid, I_max, I+1>()(array);
	}
};
template<typename ArrayType, int fid, kernel_inttype I_max>
struct populateNT<ArrayType, fid, I_max, I_max> {
	void operator() (ArrayType& array) const {	}
};

void sgemm_unrolled_NT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc) {

	int k_notset = 1;
	int m_notset = 1;
	int n_notset = 1;

	int iunrollk = 1;
	int iunrollm = 1;
	int iunrolln = 1;

	int mquarter = m>>2;

	for (int i = max_unrollNT; i > 0; i--) {
		if (k_notset && k%i==0) {
			iunrollk = i;
			k_notset = 0;
		}

		if (n_notset && n%i==0) {
			iunrolln = i;
			n_notset = 0;
		}

		if (m_notset && (mquarter)%i==0) {
			iunrollm = i;
			m_notset = 0;
		}
	}

	if (mquarter==0)
		iunrollm = 1;

	iunrolln--;
	iunrollm--;
	iunrollk--;

	const int max_unrollNT_cubed = max_unrollNT*max_unrollNT*max_unrollNT;

	std::array<fptr, max_unrollNT_cubed> look_up_nmk;
	std::array<fptr, max_unrollNT_cubed> look_up_nkm;

	int lt_index = max_unrollNT*max_unrollNT*iunrolln + max_unrollNT*iunrollm + iunrollk;

	populateNT<decltype(look_up_nmk), 0, max_unrollNT_cubed>()(look_up_nmk);
	populateNT<decltype(look_up_nkm), 1, max_unrollNT_cubed>()(look_up_nkm);

	if (m > 7 && k < 4) {
		//printf("nkm iunrolln = %d\n",iunrolln+1);
		//printf("nkm iunrollm = %d\n",iunrollm+1);
		//printf("nkm iunrollk = %d\n",iunrollk+1);
		look_up_nkm[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	else {
		//printf("nmk iunrolln = %d\n",iunrolln+1);
		//printf("nmk iunrollm = %d\n",iunrollm+1);
		//printf("nmk iunrollk = %d\n",iunrollk+1);
		look_up_nmk[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

}
//}}}

// {{{ TN
constexpr int max_unrollTN = 4;

template<typename ArrayType, int fid, kernel_inttype I_max, kernel_inttype I=0>
struct populateTN {
	void operator() (ArrayType& array) const {
		constexpr int unrolln = I/(max_unrollTN*max_unrollTN) + 1;
		constexpr int unrollm = ((I/max_unrollTN)%max_unrollTN) + 1;
		constexpr int unrollk = I%max_unrollTN + 1;
		array[I]= fid ? &unrolled_kernel_TN_nkm<unrolln, unrollm, unrollk> : &unrolled_kernel_TN_nmk<unrolln, unrollm, unrollk>;
		populateTN<ArrayType, fid, I_max, I+1>()(array);
	}
};
template<typename ArrayType, int fid, kernel_inttype I_max>
struct populateTN<ArrayType, fid, I_max, I_max> {
	void operator() (ArrayType& array) const {	}
};

void sgemm_unrolled_TN(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc) {

	int k_notset = 1;
	int m_notset = 1;
	int n_notset = 1;

	int iunrollk = 1;
	int iunrollm = 1;
	int iunrolln = 1;

	int mquarter = m>>2;

	for (int i = max_unrollTN; i > 0; i--) {
		if (k_notset && k%i==0) {
			iunrollk = i;
			k_notset = 0;
		}

		if (n_notset && n%i==0) {
			iunrolln = i;
			n_notset = 0;
		}

		if (m_notset && (mquarter)%i==0) {
			iunrollm = i;
			m_notset = 0;
		}
	}

	if (mquarter==0)
		iunrollm = 1;

	iunrolln--;
	iunrollm--;
	iunrollk--;

	const int max_unrollTN_cubed = max_unrollTN*max_unrollTN*max_unrollTN;

	std::array<fptr, max_unrollTN_cubed> look_up_nmk;
	std::array<fptr, max_unrollTN_cubed> look_up_nkm;

	int lt_index = max_unrollTN*max_unrollTN*iunrolln + max_unrollTN*iunrollm + iunrollk;

	populateTN<decltype(look_up_nmk), 0, max_unrollTN_cubed>()(look_up_nmk);
	populateTN<decltype(look_up_nkm), 1, max_unrollTN_cubed>()(look_up_nkm);

	if (m > 7 && k < 4) {
		//printf("nkm iunrolln = %d\n",iunrolln+1);
		//printf("nkm iunrollm = %d\n",iunrollm+1);
		//printf("nkm iunrollk = %d\n",iunrollk+1);
		look_up_nkm[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	else {
		//printf("nmk iunrolln = %d\n",iunrolln+1);
		//printf("nmk iunrollm = %d\n",iunrollm+1);
		//printf("nmk iunrollk = %d\n",iunrollk+1);
		look_up_nmk[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

}
//}}}

// {{{ TT
constexpr int max_unrollTT = 3; // gcc 7.1.0 produces incorrect results when this is 4

template<typename ArrayType, int fid, kernel_inttype I_max, kernel_inttype I=0>
struct populateTT {
	void operator() (ArrayType& array) const {
		constexpr int unrolln = I/(max_unrollTT*max_unrollTT) + 1;
		constexpr int unrollm = ((I/max_unrollTT)%max_unrollTT) + 1;
		constexpr int unrollk = I%max_unrollTT + 1;
		array[I]= fid ? &unrolled_kernel_TT_mkn<unrolln, unrollm, unrollk> : &unrolled_kernel_TT_mnk<unrolln, unrollm, unrollk>;
		populateTT<ArrayType, fid, I_max, I+1>()(array);
	}
};
template<typename ArrayType, int fid, kernel_inttype I_max>
struct populateTT<ArrayType, fid, I_max, I_max> {
	void operator() (ArrayType& array) const {	}
};

void sgemm_unrolled_TT(const kernel_inttype m, const kernel_inttype n, const kernel_inttype k, const float alpha,
							const float * restrict A, const kernel_inttype lda,
							const float * restrict B, const kernel_inttype ldb, const float beta,
							float * restrict C, const kernel_inttype ldc) {

	int k_notset = 1;
	int m_notset = 1;
	int n_notset = 1;

	int iunrollk = 1;
	int iunrollm = 1;
	int iunrolln = 1;

	int nquarter = n>>2;

	for (int i = max_unrollTT; i > 0; i--) {
		if (k_notset && k%i==0) {
			iunrollk = i;
			k_notset = 0;
		}

		if (n_notset && nquarter%i==0) {
			iunrolln = i;
			n_notset = 0;
		}

		if (m_notset && m%i==0) {
			iunrollm = i;
			m_notset = 0;
		}
	}

	if (nquarter==0)
		iunrolln = 1;

	iunrolln--;
	iunrollm--;
	iunrollk--;

	const int max_unrollTT_cubed = max_unrollTT*max_unrollTT*max_unrollTT;

	std::array<fptr, max_unrollTT_cubed> look_up_nmk;
	std::array<fptr, max_unrollTT_cubed> look_up_nkm;

	int lt_index = max_unrollTT*max_unrollTT*iunrolln + max_unrollTT*iunrollm + iunrollk;

	populateTT<decltype(look_up_nmk), 0, max_unrollTT_cubed>()(look_up_nmk);
	populateTT<decltype(look_up_nkm), 1, max_unrollTT_cubed>()(look_up_nkm);

	if (m > 7 && k < 4) {
		//printf("mkn iunrolln = %d\n",iunrolln+1);
		//printf("mkn iunrollm = %d\n",iunrollm+1);
		//printf("mkn iunrollk = %d\n",iunrollk+1);
		look_up_nkm[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	else {
		//printf("mnk iunrolln = %d\n",iunrolln+1);
		//printf("mnk iunrollm = %d\n",iunrollm+1);
		//printf("mnk iunrollk = %d\n",iunrollk+1);
		look_up_nmk[lt_index](m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

}
//}}}

}}
