# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

include_guard(GLOBAL)
include(FetchContent)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(JSON_BuildTests OFF)
set(JSON_Install OFF)
set(JSON_MultipleHeaders OFF)
set(JSON_SystemInclude ON)

FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH
    SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  TLS_VERIFY TRUE
)
FetchContent_MakeAvailable(nlohmann_json)
