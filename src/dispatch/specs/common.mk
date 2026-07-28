# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

ifndef PKG_JSON_SPECS_DIR
$(error PKG_JSON_SPECS_DIR is not set)
endif

SHELL=/bin/bash -o pipefail

CLANG_FORMAT = clang-format
PYTHON ?= python3

.DEFAULT_GOAL := all

.PHONY: clean
clean:
	rm -rf $(PKG_HPP_SPECS_DIR)
