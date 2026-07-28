/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_LINALG_BANDED_MV_COMMON_HPP
#define PERFLIBS_LINALG_BANDED_MV_COMMON_HPP

#include "framework/linalg_util.hpp"
#include "spec/problem_context_helpers.hpp"
#include "perflibs_assert.hpp"

namespace perflibs::linalg {
namespace {

/**
 * Given: y = alpha Ax + beta y where A is a generic matrix
 * or y = alpha Ax where A is triangular matrix
 *
 * For a certain matrix, we are only interested in the bands marked with "#"
 *  [###         ]
 *  [####        ]
 *  [ ####       ]
 *  [  ####      ]
 *  [   ####     ]
 *  [    ####    ]
 *
 * Triviallly yi = alpha.dot_product(x, A[i]) + beta.yi
 * Instead, we make the most of the sparsity of banded matrices, and we compute the above
 * equation using axpby calls for each column of the matrix, by calculating the contribution of xj and the relevant band of A to
 * the final result y.
 * Note: 0<=j<x_cntg; 0<=i<y_strd; axpby => y = alpha.x + beta.y
 *
 * We need to scale y before executing the axpby due to some corner cases
 * where the band is too small and some elements of y would not be
 * modified at all without the scale.
 *
 * When multithreading a whole strded panel of A (lets say all rows between p and q including)
 * will be given to the thread along with the elements of y that will need updating from
 * those rows (elements p to q in y). The vector x is given fully to each thread. Even though a
 * whole strded panel is given to the thread, a lot of these elements will be 0
 * either in the beginning columns, or the last columns, or both. For that reason
 * we may also want to find the subregion of A which will be actually relevant to
 * computation ("Active Region") to reduce use of function calls that are not achieving anything useful.
 *
 * The way we find the active region given a random panel of A goes by using the diagonal
 * as a reference point (If we are interpreting row i then the diagonal will be the ith element of that row).
 * From that we get that the first non zero element of that row will actually start kl elements before the
 * diagonal (or 0, if the result of that subtraction indicates that the first element should be outside that row.
 *
 * Using a similar kind of thinking, we can find the last non zero element of the row, which will be ku elements
 * after the diagonal element in the band (or max_cntg in case the value of that addition indicate that the
 * last element should be somehow outside the row).
 *
 * Once we get this active region, we start the computation:
 *
 * For each band of the matrix "j" (where j belongs to the active region) we do the following steps:
 *
 * 1) Pre-compute alpha * x(j, 0) for use in the axpby.
 *
 * 2) Obtain the actual band of the current column. Due to variations of banded matrix multiplication according
 * to the type of the matrices involved (generic, triangular, symmetric or hermitian), we get the positions and the size of the
 * band since obtaining these positions gives us more power of manipulation over the band we are getting and we can adjust them
 * more easily according to the type of matrix we have. When computing these positions, multithreading is also taken into account.
 *
 * 3) Obtain the relevant submatrix of y (They are based on the positions computed in the step above)
 *
 * 4) Call the axpby (y = xj * band_j + y).
**/

template <typename KernelAxpby, typename KernelDot>
class mv_banded {
	KernelAxpby kernel_axpby_;
	KernelDot kernel_dot_;
public:
	mv_banded(KernelAxpby kernel_axpby, KernelDot kernel_dot)
	:	kernel_axpby_ { std::move(kernel_axpby) }
	,	kernel_dot_   { std::move(kernel_dot) }
	{	}
	template <typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	void operator()(const AType& a, const BType& x, CType& y, compute_position pos,
	                ScalarType alpha, ScalarType beta, Args&&... args) {
		// Sanity checks
		PERFLIBS_ASSERT(y.cntg() == a.strd(), "y cntg doesn't match a strd");
		PERFLIBS_ASSERT(x.cntg() == a.cntg(), "x ctng doesn't match a cntg");
		PERFLIBS_ASSERT(x.strd() == y.strd(), "x strd doesn't match y strd");

		scale(beta, y);

		if (!a.is_trans()) {
			beta = one<ScalarType>;
			const kernel_inttype first_band_cntg  = max(a.absolute_strd() - a.kl(), 0);
			const kernel_inttype last_band_cntg   = min(a.absolute_strd() + a.strd() + a.ku() -1, a.cntg()-1);

			for (kernel_inttype j = first_band_cntg; j <= last_band_cntg; j++) {
				auto [band_start_strd_pos, band_end_strd_pos] = a.get_strd_band_pos(j);

				if constexpr(is_triangular_matrix_v<AType>){
				/* If A is unit then we shouldn't compute xi*A(i,i). Instead what should
				 * be added is xi*1. This is already done by setting y to x and beta to 1
				 * by the caller of this function. So instead it is in our interest to just ensure
				 * that xi*A(i,i) does not appear in yi. The way we do it is by cropping the
				 * band we are giving to axpby (conveniently, the diagonals for triangular matrix
				 * are located in the tips:
				 *      * if a is lower, then the diagonal is at the first element of the band. If we want to
				 *        ignore this first element all we need to do is increasing the first position.
				 *      * if a is upper then the diagonal is the last element of the band. If we want to ignore
				 *        the last element all we need to do is decreasing the last element of the band.
				 * To add complexity to the problem we also have to take into consideration the multithreaded case
				 * where the band fetched is actually a partial band of the parent matrix which means they might not even
				 * contain a diagonal in the first place. The way to go around this is only doing this band cropping when
				 * the band contains a diagonal:
				 *      * if a is lower, then the diagonal will be part of the band once j is bigger than a.absolute_strd(),
				 *        which makes sense because that's when the first element of that band becomes a part of submatrix a,
				 *        insread of only belonging to the parent.
				 *      * if a is upper, then the diagonal will be part of the band once j is smaller than a.absolute_strd()+a.strd(),
				 *        which makes sense because that's when j starts going outside of the range of the submatrix a.
				 *
				 */
					if(a.is_unit()){
						if(a.is_lower()){
							if(j >= a.absolute_strd() ){
								++band_start_strd_pos;
							}
						}
						else{
							if(j < a.absolute_strd()+a.strd()){
								--band_end_strd_pos;
							}
						}
					}
				}

				const kernel_inttype band_size = band_end_strd_pos - band_start_strd_pos + 1;

				auto band_j = a.sub_matrix(j, 1, band_start_strd_pos, band_size);
				auto band_j_ptr = band_j.data();
				auto alpha_x = alpha*x(j, 0);
				auto sub_y = y.sub_matrix(band_start_strd_pos, band_size, 0, 1);
				auto sub_y_ptr = sub_y.data();
				if ( band_size > 0 ) {
					kernel_axpby_(band_size, alpha_x, band_j_ptr, beta, sub_y_ptr, band_j.cntg_step(), sub_y.cntg_step());
				}
			}
		}
		else {
			kernel_inttype start_cntg_shift{0};
			kernel_inttype end_cntg_shift{0};
			if constexpr(is_triangular_matrix_v<AType>) {
				start_cntg_shift = a.is_upper() && a.is_unit() ? 1 : 0;
				end_cntg_shift = a.is_lower() && a.is_unit() ? -1 : 0;
			}

			for (kernel_inttype j = 0; j < a.strd(); ++j) {
				// Assuming that matrix A is decomposed only in the strd dimension for parallelisation
				// (j + absolute_strd) gives us the diagonal position. Then, we take into account the kl and ku, and
				// and crop the start and end indices to 0 and cntg-1. Add a shift of +1 and -1 for unit triangular matrices

				const kernel_inttype band_start_cntg_pos = max(j + a.absolute_strd() - a.kl(), 0) + start_cntg_shift;
				const kernel_inttype band_end_cntg_pos = min(j + a.absolute_strd() + a.ku(), a.cntg()-1) + end_cntg_shift;
				const kernel_inttype band_size = band_end_cntg_pos - band_start_cntg_pos + 1;
				auto band_j = a.sub_matrix(band_start_cntg_pos,band_size,j,1);
				auto sub_x = x.sub_matrix(band_start_cntg_pos,band_size,0,1);
				if ( band_size > 0 ) {
					y(j,0, write) += alpha * kernel_dot_(band_size, band_j.data(), sub_x.data(), band_j.strd_step(), sub_x.cntg_step());
				}
			}
		} // IsTrans

	} // operator()

}; // mv_banded

/**
 * Given: y = alpha Ax + beta y, where A is either a symmeric or an hermitian matrix
 *
 * For a certain matrix, we are only interested in the bands marked with "#"
 *  [###         ]
 *  [####        ]
 *  [ ####       ]
 *  [  ####      ]
 *  [   ####     ]
 *  [    ####    ]
 *
 * Triviallly yi = alpha.dot_product(x, A[i]) + beta.yi.
 * Instead, we make the most of the sparsity of banded matrices, and we compute the above
 * equation using axpy calls for each column of the matrix, by calculating the contribution of xj and the relevant band of A to
 * the final result y.
 * Note: 0<=j<x_cntg; 0<=i<y_strd; axby => y = alpha.x + y; dot = sum(x[i]*y[i])
 *
 * First thing, this algorithm implies scaling y by beta. Since multiplying a bvector by scalar is a higher priority operation than sum, and since
 * sum is commutative, then this scaling operation can be actually the first one we make (which should be in the responsibility of this operator caller)
 * This way this operator only has to compute y = alpha Ax + y.
 *
 * Second we need to actually find the active region in A which is the area of A we will be processing within a certain thread
 * When multithreading a whole strded panel of A (lets say all rows between p and q including)
 * will be given to the thread along with the elements of y that will need updating from
 * those rows (elements p to q in y). The vector x is given fully to each thread. Even though a
 * whole strded panel is given to the thread, a lot of these elements will be 0
 * either in the beginning columns, or the last columns, or both. For that reason
 * we may also want to find the subregion of A which will be actually relevant to
 * computation ("Active Region") to reduce use of function calls that are not achieving anything useful.
 * The way we find the active region given a random panel of A goes by using the diagonal
 * as a reference point (If we are interpreting row i then the diagonal will be the ith element of that row).
 * From that we get that the first non zero element of that row will actually start kl elements before the
 * diagonal (or 0, if the result of that subtraction indicates that the first element should be outside that row.
 * Using a similar kind of thinking, we can find the last non zero element of the row, which will be ku elements
 * after the diagonal element in the band (or max_cntg in case the value of that addition indicate that the
 * last element should be somehow outside the row).

 * Once we get this active region, we start the computation.
 * For each band of the matrix "j" (where j belongs to the active region) we have to consider its physical contribution and its virtual
 * contribution.
 *
 * THE PHYSICAL CONTRIBUTION :
 * 1) Pre-compute alpha * x(j, 0) for use in the axpby.
 * 2) Obtain the actual band of the current column. Due to variations of banded matrix multiplication according
 * to the type of the matrices involved (symmetric or hermitian), we get the positions and the size of the
 * band since obtaining these positions gives us more power of manipulation over the band we are getting and we can adjust them
 * more easily according to the type of matrix we have. When computing these positions, multithreading is also taken into account.
 * 3) Obtain the relevant submatrix of y (They are based on the positions computed in the step above)
 * 4) Call the axpby (y = xj * band_j + y).
 *
 * THE VIRTUAL CONTRIBUTION :
 * 1) Adjust the band we are using. We want the same band as above but we do not want to take into account the diagonal (while
 * always having multithreading in our minds).
 *    1.1 Get the size of the virtual band
 *         We have to compute the size of the virtual band since we need to know how many elements we need to compute
 *         from the band. On sequential execution this would just mean decrementing the size of the original band by 1.
 *         However, taking multithreading into account, we cannot do this because the band_size variable does not hold the
 *         size of the full band in the original matrix, just the size of the sub band that results from the
 *         intersection between the band and the submatrix that this thread is processing.
 *
 * 2) Get the subvector of x with the elements that will participate in this computation
 * 3) Update yj to alpha*dot_product(sub_x, adjusted_band_j)
 *          When multithreading, we don't have access to full y, just a subpart of it, which from the thread point
 *          of view represent the whole of y. Sequentially, if we are dealing with band j then we would update
 *          y(j,1). When multithreading we have to shift the j that accesses y by how far from the top of
 *          the matrix this y is starting (accessing the jth band in the contiguous dimension means
 *          updating the j - a.absolute_strd() element in y where a.absolute_strd() returns the absolute strd start
 *          position of a in its parent matrix (which will be the same offset of this thread y in comparison to its parent y) )
 * If the matrix is hermitian, similarly, we still want to compute the virtual side using the same steps. However this time we
 * have to remember that the physical band does not contain the diagonal. We also cannot forget to compute the contribution of
 * the diagonal separately
 *
 * VISUAL AID :
 * Consider the following matrix A which could be hermitian or symmetric
 * (lower matrix, # represents diagonals, ? represents virtual values, " " represent 0):
 * 0 [#??  ]
 * 1 [a#?? ]
 * 2 [bc#??]
 * 3 [ de#?]
 * 4 [  fg#]
 * Let's say multithreading is occurring, so thread i is supposed to process rows 2 and 3
 * (it is ensured that multithreading for banded matrix operations occur in the strd dimension only):
 *    01234
 * 2 [bc#??]
 * 3 [ de#?]
 * So the active region (the columns which we will be processing in this submatrix are columns 0,1,2 and 3 (the fourth
 * column only contains virtual values)).
 * So in iteration 0 we process band [b]. We only do its physical processing since it does not contain a diagonal.
 * Similarly, in iteration 1 we would be processing band 1 [c,d]. It does not contain a diagonal so we only proceed with computing its physical
 * contribution.
 * Once again iteration 2 we are processing band [#e] and we compute its physical contribution. However this band actually contains a diagonal.
 * So it is eligible for virtual computation as well. This is where the ? values will be taken into consideration, even though they are not
 * physically stored. The way we get the ? values is by applying a simple operation o (which could stand for the conj operation or the
 * identity operation) performed on the respective element on the physical side of the matrix. So technically the matrix we are computing in this
 * thread actually looks more like:
 * 2 [b c # o(e) o(f)]
 * 3 [  d e #    o(g)]
 * In the physical computation we take into account the contribution of [#e] into the final result in y. In the virtual computation we take into
 * account the contribution of [o(e) o(f)]. Access to this band can be granted reusing the pointer to the physical band [# e], by incrementing
 * the pointer by one storage unit so we are not including the diagonal anymore, and then by increasing the size of the virtual band we are considering so we
 * actually inclyde f in it as well (even though f does not belong to the input submatrix given to this thread, it is okay to access it, since are
 * reading only). After doing this we will end with a pointer and a size which will define [e, f] which we call the virtual band.
 * How do we convert this to [o(e), o(f)]. If o is the identity operation, we do not need to, since that just falls back to [e, f].
 * If the matrix is hermitian, then the dot product call that we make to compute this contribution will have conj set to true, so that
 * the conj operation will be performed to the input inside the dot product call.
 * We still need to fetch the other operands in order to successfully compute the virtual computation: We need the right submatrix of x ([x[3], x[4]]) and
 * we need the right y (y[2]) for this case.
 * The whole of x is known to the thread, so the difficulty in this step is in trying to get a pointer to the beginning of the x submatrix we
 * want to consider. That is not too hard since the index of the element in x to which we want to point at will be the same as the index of the
 * column of the first virtual element in the virtual band (which is just the index of the physical band column +1). For y things complicate more
 * because of multithreading, where for this case only [y[2], y[3]] is known to the thread. In the original submatrix, the y to update would be the one
 * that has j as its index (In this case y[2] would be updated because we were in iteration 2 processing physical band 2) With multithreading, we are actually
 * updating sub_y[0]. So how do we go from updating original y at index 2 to updating sub y at index 0. Well that is just a translation of index 2 by -2 units.
 * If we subject all y indexes to this transformation (index_y = j - 2) we will get the index in sub_y that we want to update at iteration j.
 * Obviously the translation is by an offset of -2 for this case, but as the sub_y matrix given to a thread has its starting point further away from the
 * starting point in original y, the offset decreases. So how do we know how much to shift j by whenever we are working with a sub y from the perspective of a certain thread?
 * By the same offset the submatrix A is delocalized along the strded dimension.
 * So if the strd of sub_A in the original matrix is 2 (like in this case), that is exactly how much we will have to shift y known original index (j) when we
 * are updating it with dot calls during the virtual computation
 *
 * NOTE that for hermitian/symmetric matrices if the provided matrix is lower "L" then it means that when we access its attributes
 * in this framework (ie a.is_upper) we should get that this matrix is actually seen as an upper matrix by the system.
 * Reason for this has to do with how symmetric matrices are stored according to this framework
 * which causes a flip of the uplo of the matrix.
 * The same phenomenon happens with matrices specified to be upper ("U") by the user which actually make a.is_lower() return true
 * and a.is_upper return false
 *
**/

template <typename KernelAxpby, typename KernelDot>
class mv_banded_with_symmetry {
	KernelAxpby kernel_axpby_;
	KernelDot kernel_dot_;
public:
	mv_banded_with_symmetry(KernelAxpby kernel_axpby, KernelDot kernel_dot)
	:	kernel_axpby_ { std::move(kernel_axpby) }
	,	kernel_dot_   { std::move(kernel_dot)   }
	{	}

	template <typename AType, typename BType, typename CType, typename ScalarType, typename... Args>
	void operator()(const AType& a, const BType& x, CType& y, compute_position pos,
	                ScalarType alpha, ScalarType beta, Args&&... args) {
		// Sanity checks
		PERFLIBS_ASSERT(y.cntg() == a.strd(), "y cntg doesn't match a strd");
		PERFLIBS_ASSERT(x.cntg() == a.cntg(), "x ctng doesn't match a cntg");
		PERFLIBS_ASSERT(x.strd() == y.strd(), "x strd doesn't match y strd");

		scale(beta, y);
		beta = one<ScalarType>;

		auto is_lower = a.is_upper();
		const kernel_inttype first_band_cntg  = max(a.absolute_strd() - a.kl(), 0);
		const kernel_inttype last_band_cntg   =  min(a.absolute_strd() + a.strd() + a.ku() -1, a.cntg()-1);

		for (kernel_inttype j = first_band_cntg; j <= last_band_cntg; j++) {
			auto [band_start_strd_pos, band_end_strd_pos] = a.get_strd_band_pos(j);

			if constexpr(is_hermitian_matrix_v<AType>){
				/* If A is hermitian then we should compute xi*A(i,i) separately since the diagonal might
				 * contain a complex value (of which we are only interested in the real part).
				 * This is done later in the code, so for now we have to make sure
				 * that xi*A(i,i) does not appear in yi. The way we do it is by cropping the
				 * band we are giving to axpby (conveniently, the diagonals for triangular matrix
				 * are located in the tips:
				 *      * if a is lower, then the diagonal is at the first element of the band. If we want to
				 *        ignore this first element all we need to do is increasing the first position.
				 *      * if a is upper then the diagonal is the last element of the band. If we want to ignore
				 *        the last element all we need to do is decreasing the last element of the band.
				 * To add complexity to the problem we also have to take into consideration the multithreaded case
				 * where the band fetched is actually a partial band of the parent matrix which means they might not even
				 * contain a diagonal in the first place. The way to go around this is only doing this band cropping when
				 * the band contains a diagonal:
				 *      * if a is lower, then the diagonal will be part of the band once j is bigger than a.absolute_strd(),
				 *        which makes sense because that's when the first element of that band becomes a part of submatrix a,
				 *        insread of only belonging to the parent.
				 *      * if a is upper, then the diagonal will be part of the band once j is smaller than a.absolute_strd()+a.strd(),
				 *        which makes sense because that's when j starts going outside of the range of the submatrix a.
				 */
					if(is_lower){ //when a is lower
						if(j >= a.absolute_strd()){
							++band_start_strd_pos;
						}
					}
					else{
						if(j < a.absolute_strd()+a.strd()){
							--band_end_strd_pos;
						}
					}
				}

			const kernel_inttype band_size = band_end_strd_pos - band_start_strd_pos + 1;
			auto band_j = a.sub_matrix(j, 1, band_start_strd_pos, band_size);
			auto band_j_ptr = band_j.data();
			auto alpha_x = alpha*x(j, 0);
			auto sub_y = y.sub_matrix(band_start_strd_pos, band_size, 0, 1);
			auto sub_y_ptr = sub_y.data();

			if constexpr(is_hermitian_matrix_v<AType> ){
				if( max(a.kl(), a.ku()) == 1){
					if(band_size>0){
						sub_y_ptr[0] += alpha_x * band_j_ptr[0];
					}
				}
				else{
					kernel_axpby_(band_size, alpha_x, band_j_ptr, one<ScalarType>, sub_y_ptr, band_j.cntg_step(), sub_y.cntg_step());
				}
			}
			else{
				kernel_axpby_(band_size, alpha_x, band_j_ptr, one<ScalarType>, sub_y_ptr, band_j.cntg_step(), sub_y.cntg_step());
			}

			if(is_lower){ // The lower case as specified by the user
				/*
				* We compute the contribution of the virtual side when we are processing a band that would have contained a diagonal
				* which only occurs when we are dealing with j bigger or equal than the absolute strd position of
				* this submatrix in its parent. Otherwise we assume that the virtual side has already been computed by
				* previous threads.
				*/
				if(j < a.absolute_strd()) continue;
				auto y_pos = j - a.absolute_strd();
				// We are interested in a pointer to the start of the virtual band that does not contain a diagonal
				// For hermitian matrices that has already been taken into account. For symmetric ones the adjustmment
				// is still necessary.
				auto band_j_ptr_v = band_j.data();
				if constexpr(is_hermitian_matrix_v<AType> ){
					//This is where we finally compute the diagonal contribution for hermitian matrices.
					y(y_pos, 0, write) += alpha_x*a(j,y_pos);
				}
				else{ // is_symmetric matrix
					//This represents the adjustment necessary during sbmv so that the virtual band we are considering does not include
					// the diagonal.
					band_j_ptr_v += band_j.cntg_step();
				}
				/*
				* The actual size of the virtual band in the original matrix is usually kl. However, sometimes the distance
				* between the start of the virtual band j and the right side of the matrix (along the strd dimension) is
				* sometimes not big enough, so we have to crop it accordingly (explains the use of the min operator).
				*/
				const kernel_inttype virt_band_size = min(a.cntg() - j - 1, a.kl()) ;
				auto x_cntg = x.sub_matrix(j + 1, virt_band_size, 0, 1);
				//auto x_mut_ptr = const_cast<ScalarType *>(x_cntg.data());
				auto x_mut_ptr = x_cntg.data();
				if(virt_band_size>0){
					if constexpr (is_hermitian_matrix_v<AType> ){
						if( max(a.kl(), a.ku()) ==1){
							y(y_pos, 0, write) += alpha * x_mut_ptr[0] * conj(band_j_ptr_v[0]);
						}
						else{
							y(y_pos, 0, write) += alpha* kernel_dot_(virt_band_size, band_j_ptr_v,  x_mut_ptr , band_j.cntg_step(), x.cntg_step() );
						}
					}
					else{
						y(y_pos, 0, write) += alpha* kernel_dot_(virt_band_size, band_j_ptr_v,  x_mut_ptr , band_j.cntg_step(), x.cntg_step() );
					}
				}
			}
			else{  // The upper case as specified by the user
				/*
				* Again, we compute the contribution of the virtual side wen we are processing a band that contains would have contained a diagonal
				* which only occurs when we are dealing with j < a.absolute_strd()+a.strd()
				* Otherwise we save the computation of the virtual side for the future threads (when the respective diagonal has been
				* encountered).
				*/
				if(j >= a.absolute_strd() + a.strd() )  continue;
				auto y_pos = j - a.absolute_strd();
				/*
				* The actual size of the virtual band in the original matrix is usually ku. However, sometimes the distance
				* between the end of the virtual band j and the left side of the matrix (along the strd dimension) is
				* sometimes not big enough, so we have to crop it accordingly (explains the use of the min operator).
				*/
				const kernel_inttype virt_band_size = min(j, a.ku()) ;
				auto offset = virt_band_size - band_size ;
				if constexpr ( is_hermitian_matrix_v<AType> ){
					//This is where we finally compute the diagonal contribution for hermitian matrices.
					y(y_pos, 0, write) += alpha_x*a(j,y_pos);
				}
				else{
					//For sbmv, the offset we want to use will be slightly bigger since we want to skip the diagonal as well in the
					//virtual band
					++offset;
				}
				/*
				* With the assumption that the current band would have contained a diagonal then we are interested in this very band without
				* its diagonal. This is not so simple since the sub band we are currently looking at (band_j) could have any starting
				* point within its original band according to where it is split along the strd dimension before feeding it as
				* input to the thread. Our problem becomes trying adjust the pointer we currently have to band_j so it points to
				* the start of band j in the original matrix. We do this by trying to find out the distance (offset) between the first
				* element of the sub band j and the first element of the original band j. Once we have that offset then we use it to
				* set our pointer to the accurate start position of the original band.
				*/
				auto band_j_ptr_v = band_j.data() - band_j.cntg_step() * offset;
				auto x_cntg = x.sub_matrix(j-virt_band_size, virt_band_size, 0, 1);
				auto x_mut_ptr = x_cntg.data();
				if(virt_band_size>0){
					if constexpr (is_hermitian_matrix_v<AType> ){
						if( max(a.kl(), a.ku()) ==1){
							y(y_pos, 0, write) += alpha * x_mut_ptr[0] * conj(band_j_ptr_v[0]);
						}
						else{
							y(y_pos, 0, write) += alpha* kernel_dot_(virt_band_size, band_j_ptr_v,  x_mut_ptr , band_j.cntg_step(), x.cntg_step() );
						}
					}
					else{
						y(y_pos, 0, write) += alpha* kernel_dot_(virt_band_size, band_j_ptr_v,  x_mut_ptr , band_j.cntg_step(), x.cntg_step() );
					}
				}
			}
		}
	}
}; // mv_banded_with_symmetry

}
} //namespace perflibs::linalg
#endif // PERFLIBS_LINALG_BANDED_MV_COMMON_HPP
