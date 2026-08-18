# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Rules to generate hpp specs from json spec files.

# ==============================================================
# USEFUL VARIABLES

# For now ROOT DIRECTORY is the json specs directory
export ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ALL_SPECS_DIR ?= $(ROOT_DIR)
# SCRIPTS_DIR points to directory with scripts necessary for header generation
SCRIPTS_DIR := $(ALL_SPECS_DIR)/scripts
# Directory with (preferably tuned) json spec files
JSON_SPECS_DIR         ?= $(ALL_SPECS_DIR)/json_specs
# For now rules only apply to a target at a time
# Options among directories in JSON_SPECS_DIR (blas, factorization, ...)
PACKAGE                ?= blas
# Optional list of target systems (space-separated) to restrict spec generation.
SYSTEMS                ?=
# Subdirectory for particular package with the json files
PKG_JSON_SPECS_DIR     := $(JSON_SPECS_DIR)/$(PACKAGE)/targets
PKG_JSON_DEFAULTS_DIR  := $(JSON_SPECS_DIR)/$(PACKAGE)/defaults
PKG_JSON_TEMPLATES_DIR := $(JSON_SPECS_DIR)/$(PACKAGE)/templates

# Build directories where headers will be stored:
HPP_SPECS_DIR        ?= $(ALL_SPECS_DIR)/hpp_specs
PKG_HPP_SPECS_DIR    := $(HPP_SPECS_DIR)/$(PACKAGE)

# List all json files in package directory.
JSON_SPECS_PATHS=$(shell find $(PKG_JSON_SPECS_DIR) -type f -name '*.json' -print)
# If SYSTEMS is provided, restrict to the listed target system directories.
ifneq ($(strip $(SYSTEMS)),)
JSON_SPECS_PATHS=$(foreach system,$(SYSTEMS),$(wildcard $(PKG_JSON_SPECS_DIR)/$(system)/*.json))
endif
JSON_DEFAULTS_PATHS=$(shell find $(PKG_JSON_DEFAULTS_DIR) -type f -name '*.json' -print)
JSON_TEMPLATES_PATHS=$(shell find $(PKG_JSON_TEMPLATES_DIR) -type f -name '*.json' -print)

include common.mk

# ==============================================================
# PHASE 1: Generate corresponding hpp header files for each json in
# JSON_SPECS_DIR/PACKAGE/targets/*/*.json
# Example: json_specs/blas/targets/a64fx/axpby_c32.json --> hpp_specs/blas/a64fx/axpby_c32.hpp

# Requires the following scripts:
GENERATE_TUNED_SPEC_PATH=$(SCRIPTS_DIR)/generate_tuned_spec.py
GENERATE_LIVE_TARGET_PATH=$(SCRIPTS_DIR)/generate_live_target_routine_specs.py
GENERATE_INCLUDES_PATH=$(SCRIPTS_DIR)/gen_includes.py

# For each json file found in PKG_JSON_SPECS_DIR, substitute file path relative
# to the json directory by a path relative to the
# directory where headers are stored. Keep file name.
HPP_SPEC_PATHS=$(patsubst $(PKG_JSON_SPECS_DIR)/%.json,$(PKG_HPP_SPECS_DIR)/%.hpp,$(JSON_SPECS_PATHS))
# Stores the directories in which these headers are going to be spread around
# (according to package/target)
HPP_SPEC_DIRS=$(sort $(dir $(HPP_SPEC_PATHS)))

# Add in live target specs (generated from defaults) if enabled
ifeq ($(live), 1)
ifeq ($(JSON_DEFAULTS_PATHS),)
$(error live=1 needs generated defaults in $(PKG_JSON_DEFAULTS_DIR); run live-target defaults generation first)
endif
HPP_SPEC_PATHS+=$(patsubst $(PKG_JSON_DEFAULTS_DIR)/%.json,$(PKG_HPP_SPECS_DIR)/live_targets/%.hpp,$(JSON_DEFAULTS_PATHS))
endif

# For each target directory generated, add an extra header file, tuned_specs.hpp
# which congregates all the routines for this target.
INCLUDE_PATHS=$(addsuffix tuned_specs.hpp, $(HPP_SPEC_DIRS))

.PHONY: hpp-specs
hpp-specs: $(HPP_SPEC_PATHS) $(INCLUDE_PATHS)

# Rule to create an hpp header file per json file
$(PKG_HPP_SPECS_DIR)/%.hpp: $(PKG_JSON_SPECS_DIR)/%.json $(GENERATE_TUNED_SPEC_PATH)
	mkdir -p $(@D)
	@echo Generating $@ from $<
	$(PYTHON) $(GENERATE_TUNED_SPEC_PATH) \
		-i $< --package $(PACKAGE) | \
		($(CLANG_FORMAT) || cat) > $@

# Rule to create live target hpp header file per json defaults file
$(PKG_HPP_SPECS_DIR)/live_targets/%.hpp: $(PKG_JSON_DEFAULTS_DIR)/%.json $(GENERATE_LIVE_TARGET_PATH)
	mkdir -p $(@D)
	@echo Generating $@ from $<
	$(PYTHON) $(GENERATE_LIVE_TARGET_PATH) \
		-i $< --package $(PACKAGE) -t hpp | \
		($(CLANG_FORMAT) || cat) > $@

# Rule to make tuned_specs.hpp file per target
%/tuned_specs.hpp: $(GENERATE_INCLUDES_PATH) $(HPP_SPEC_PATHS)
	$(GENERATE_INCLUDES_PATH) $(PACKAGE) $$(basename $(@D)) $(PKG_JSON_SPECS_DIR) $@

# ==============================================================
# PHASE 2: Generate an hpp file that connects headers for all the targets
# Similar idea to last step of PHASE1

GENERATE_ROUTINE_SPECS_FLAGS :=
ifeq ($(live), 1)
	GENERATE_ROUTINE_SPECS_FLAGS := --is-live
endif

# Requires the following scripts:
GENERATE_ROUTINE_SPECS_PATH=$(SCRIPTS_DIR)/generate_routine_specs.py
HPP_TUNED_ROUTINE_SPECS_PATH=$(PKG_HPP_SPECS_DIR)/tuned_routine_specs.hpp

.PHONY: routine-specs
routine-specs: $(HPP_TUNED_ROUTINE_SPECS_PATH)

# Rule to create tuned_routines_spec.hpp living right under HPP_SPECS_DIR/PACKAGE
$(HPP_TUNED_ROUTINE_SPECS_PATH): $(JSON_TEMPLATES_PATHS)  $(GENERATE_ROUTINE_SPECS_PATH)
	@echo Generating routine specs header file
	mkdir -p $(@D)
	$(PYTHON) $(GENERATE_ROUTINE_SPECS_PATH) \
		-p $(PACKAGE) \
		-i $(JSON_TEMPLATES_PATHS) \
		$(GENERATE_ROUTINE_SPECS_FLAGS) | \
		($(CLANG_FORMAT) || cat) > $@


# ========================================
.PHONY: all
all: hpp-specs routine-specs
