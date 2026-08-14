# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

KERNEL_SPECS_SOURCE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
KERNEL_SPECS_JSON_DIR := $(KERNEL_SPECS_SOURCE_DIR)/json
KERNEL_SPECS_SCRIPTS_DIR := $(KERNEL_SPECS_SOURCE_DIR)/scripts
KERNEL_SPECS_BUILD_DIR := $(GENERATE_DIR)/kernel_specs

KERNEL_SPECS_JSON_NAMES := $(notdir $(basename $(wildcard $(KERNEL_SPECS_JSON_DIR)/*.json)))
KERNEL_SPECS_SCRIPT_NAMES := $(notdir $(basename $(wildcard $(KERNEL_SPECS_SCRIPTS_DIR)/*.py)))
KERNEL_SPECS_NAMES := $(filter $(KERNEL_SPECS_SCRIPT_NAMES),$(KERNEL_SPECS_JSON_NAMES))
KERNEL_SPECS_HPP_PATHS := $(addprefix $(KERNEL_SPECS_BUILD_DIR)/,$(addsuffix .hpp,$(KERNEL_SPECS_NAMES)))

.PRECIOUS: $(KERNEL_SPECS_HPP_PATHS)
$(KERNEL_SPECS_BUILD_DIR)/%.hpp: $(KERNEL_SPECS_JSON_DIR)/%.json $(KERNEL_SPECS_SCRIPTS_DIR)/%.py
	mkdir -p $(@D)
	$(PYTHON) $(KERNEL_SPECS_SCRIPTS_DIR)/$*.py $< $@.tmp
	$(FORMAT) $@.tmp || true
	mv $@.tmp $@
