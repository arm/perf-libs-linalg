/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PL_LINALG_STATUS_H
#define PL_LINALG_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ENUMs */

typedef enum pl_linalg_status {
	PL_LINALG_STATUS_SUCCESS=0,
	PL_LINALG_STATUS_INPUT_PARAMETER_ERROR=1,
	PL_LINALG_STATUS_EXECUTION_FAILURE=2,
} pl_linalg_status_t;

#ifdef __cplusplus
} //extern "C"
#endif

#endif //PL_LINALG_STATUS_H
