set(ARGS)
if(DEFINED INPUT)
  list(APPEND ARGS INPUT_FILE "${INPUT}")

  # BLAS2/3 drivers write pass/fail details to the configured summary file.
  file(STRINGS "${INPUT}" SUMMARY LIMIT_COUNT 1)
  string(REGEX REPLACE "^[ \t]*'?([^' \t]+)'?.*$" "\\1" SUMMARY "${SUMMARY}")
  file(REMOVE "${SUMMARY}")
endif()

message("Running: ${TEST}")
execute_process(
  COMMAND "${TEST}"
  ${ARGS}
  RESULT_VARIABLE RET
  OUTPUT_VARIABLE LOG
  ERROR_VARIABLE LOG
)

if(DEFINED SUMMARY AND EXISTS "${SUMMARY}")
  file(READ "${SUMMARY}" SUMMARY_OUTPUT)
  string(APPEND LOG "\n${SUMMARY_OUTPUT}")
endif()

message("${LOG}")

# if the test does not return 0, then fail it
if(NOT ${RET} EQUAL 0)
  message(FATAL_ERROR "Test ${TEST} returned ${RET}")
endif()

# Netlib BLAS drivers use plain STOP, so failures can still exit 0.
set(FAILURE_PATTERN "FAIL|FAILED|TESTS ABANDONED|FATAL ERROR|NOT RECOGNIZED")
if(LOG MATCHES "${FAILURE_PATTERN}")
  if(DEFINED SUMMARY_OUTPUT)
    message("BLAS summary output (${SUMMARY}):\n${SUMMARY_OUTPUT}")
  endif()
  message(FATAL_ERROR "Test ${TEST} reported a BLAS failure")
endif()

if(DEFINED INPUT)
  if(NOT EXISTS "${SUMMARY}")
    message(FATAL_ERROR "Test ${TEST} did not create summary file ${SUMMARY}")
  endif()
  if(NOT SUMMARY_OUTPUT MATCHES "END OF TESTS")
    message("BLAS summary output (${SUMMARY}):\n${SUMMARY_OUTPUT}")
    message(FATAL_ERROR "Test ${TEST} did not complete summary output")
  endif()
endif()

message("Test ${TEST} returned ${RET}")
