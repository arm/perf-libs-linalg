# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

KERNEL_SPECS_SOURCE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
KERNEL_SPECS_JSON_DIR := $(KERNEL_SPECS_SOURCE_DIR)/json
KERNEL_SPECS_SCRIPTS_DIR := $(KERNEL_SPECS_SOURCE_DIR)/scripts
KERNEL_SPECS_BUILD_DIR := $(GENERATE_DIR)/kernel_specs

KERNEL_SPECS_JSON_NAMES := $(notdir $(basename $(wildcard $(KERNEL_SPECS_JSON_DIR)/*.json)))
KERNEL_SPECS_SCRIPT_NAMES := $(notdir $(basename $(wildcard $(KERNEL_SPECS_SCRIPTS_DIR)/*.py)))
KERNEL_SPECS_NAMES := $(filter $(KERNEL_SPECS_SCRIPT_NAMES),$(KERNEL_SPECS_JSON_NAMES))

AXPBY_KERNEL_JSON_PATHS := $(wildcard $(KERNEL_SPECS_JSON_DIR)/axpby_*.json)
AXPBY_KERNEL_GENERATOR := $(KERNEL_SPECS_SCRIPTS_DIR)/axpby_kernels.py
AXPBY_KERNEL_HPP_PATH := $(KERNEL_SPECS_BUILD_DIR)/axpby_kernels.hpp

INTERLEAVE_KERNEL_JSON_PATH := $(KERNEL_SPECS_JSON_DIR)/interleave_kernels.json
INTERLEAVE_KERNEL_GENERATOR := $(KERNEL_SPECS_SCRIPTS_DIR)/interleave_kernels.py
INTERLEAVE_FALLBACK_DATA_TYPES := $(shell $(PYTHON) $(INTERLEAVE_KERNEL_GENERATOR) --print-fallback-data-types $(INTERLEAVE_KERNEL_JSON_PATH))

INTERLEAVE_FALLBACK_CPP_PATHS = $(foreach architecture,$(ARCHITECTURES),$(foreach data_type,$(INTERLEAVE_FALLBACK_DATA_TYPES),$(GENERATE_DIR)/architectures/$(architecture)/linalg/kernel_specs/interleave_fallback_instantiations_$(data_type).cpp))

KERNEL_SPECS_HPP_PATHS := $(addprefix $(KERNEL_SPECS_BUILD_DIR)/,$(addsuffix .hpp,$(KERNEL_SPECS_NAMES))) $(AXPBY_KERNEL_HPP_PATH)
KERNEL_SPECS_CPP_PATHS = $(foreach architecture,$(ARCHITECTURES),$(GENERATE_DIR)/architectures/$(architecture)/linalg/kernel_specs/axpby_fallback_instantiations.cpp)
KERNEL_SPECS_CPP_PATHS += $(INTERLEAVE_FALLBACK_CPP_PATHS)
KERNEL_SPECS_OBJ_PATHS = $(patsubst $(GENERATE_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(KERNEL_SPECS_CPP_PATHS))

.PRECIOUS: $(KERNEL_SPECS_HPP_PATHS)
$(KERNEL_SPECS_BUILD_DIR)/%.hpp: $(KERNEL_SPECS_JSON_DIR)/%.json $(KERNEL_SPECS_SCRIPTS_DIR)/%.py
	mkdir -p $(@D)
	$(PYTHON) $(KERNEL_SPECS_SCRIPTS_DIR)/$*.py $< $@.tmp
	$(FORMAT) $@.tmp || true
	test -s $@.tmp && mv $@.tmp $@

.PRECIOUS: $(GENERATE_DIR)/architectures/%/linalg/kernel_specs/axpby_fallback_instantiations.cpp
$(GENERATE_DIR)/architectures/%/linalg/kernel_specs/axpby_fallback_instantiations.cpp: $(AXPBY_KERNEL_JSON_PATHS) $(AXPBY_KERNEL_GENERATOR)
	mkdir -p $(@D)
	$(PYTHON) $(AXPBY_KERNEL_GENERATOR) --fallback-instantiations $* $(AXPBY_KERNEL_JSON_PATHS) > $@.tmp
	$(FORMAT) $@.tmp || true
	test -s $@.tmp && mv $@.tmp $@

define make_interleave_fallback_instantiations
.PRECIOUS: $(GENERATE_DIR)/architectures/%/linalg/kernel_specs/interleave_fallback_instantiations_$1.cpp
$(GENERATE_DIR)/architectures/%/linalg/kernel_specs/interleave_fallback_instantiations_$1.cpp: $(INTERLEAVE_KERNEL_JSON_PATH) $(INTERLEAVE_KERNEL_GENERATOR)
	mkdir -p $$(@D)
	$(PYTHON) $(INTERLEAVE_KERNEL_GENERATOR) --fallback-instantiations $$* $1 $$< $$@.tmp
	$(FORMAT) $$@.tmp || true
	test -s $$@.tmp && mv $$@.tmp $$@
endef

$(foreach data_type,$(INTERLEAVE_FALLBACK_DATA_TYPES),$(eval $(call make_interleave_fallback_instantiations,$(data_type))))

$(AXPBY_KERNEL_HPP_PATH): $(AXPBY_KERNEL_JSON_PATHS) $(AXPBY_KERNEL_GENERATOR)
	mkdir -p $(@D)
	$(PYTHON) $(AXPBY_KERNEL_GENERATOR) $(AXPBY_KERNEL_JSON_PATHS) > $@.tmp
	$(FORMAT) $@.tmp || true
	test -s $@.tmp && mv $@.tmp $@
