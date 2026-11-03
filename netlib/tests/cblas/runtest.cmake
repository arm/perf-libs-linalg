# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(args)
if(DEFINED INPUT)
  list(APPEND args INPUT_FILE "${INPUT}")
endif()

execute_process(COMMAND "${TEST}" ${args}
  RESULT_VARIABLE result OUTPUT_VARIABLE log ERROR_VARIABLE log)
message("${log}")
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "CBLAS test ${TEST} returned ${result}")
endif()

# The drivers can report failures and exit successfully.
if(log MATCHES "FAIL|SUSPECT|TESTS ABANDONED|FATAL ERROR|NOT RECOGNIZED|NOT DETECTED|XERBLA WAS CALLED|illegal value")
  message(FATAL_ERROR "CBLAS test ${TEST} reported a failure")
endif()

if(DEFINED INPUT)
  if(NOT log MATCHES "END OF TESTS")
    message(FATAL_ERROR "CBLAS test ${TEST} did not complete")
  endif()
else()
  # Each level-1 driver contains ten cases and has no end-of-tests marker.
  string(REGEX MATCHALL "----- PASS -----" passes "${log}")
  list(LENGTH passes pass_count)
  if(NOT pass_count EQUAL 10)
    message(FATAL_ERROR "CBLAS test ${TEST}: expected 10 passes, found ${pass_count}")
  endif()
endif()
