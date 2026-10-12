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
# List of target systems (space-separated) to restrict spec generation.
SYSTEMS                ?=
# Subdirectory for particular package with the json files
PKG_JSON_SPECS_DIR     := $(JSON_SPECS_DIR)/$(PACKAGE)/targets
PKG_JSON_DEFAULTS_DIR  := $(JSON_SPECS_DIR)/$(PACKAGE)/defaults
PKG_JSON_TEMPLATES_DIR := $(JSON_SPECS_DIR)/$(PACKAGE)/templates

# Build directories where headers will be stored:
HPP_SPECS_DIR        ?= $(ALL_SPECS_DIR)/hpp_specs
PKG_HPP_SPECS_DIR    := $(HPP_SPECS_DIR)/$(PACKAGE)

TARGETS := $(sort $(SYSTEMS))
ifeq ($(TARGETS),)
ifneq ($(MAKECMDGOALS),clean)
$(error SYSTEMS is required for spec generation, e.g. SYSTEMS="n1_c6g")
endif
endif
JSON_TEMPLATES_PATHS=$(shell find $(PKG_JSON_TEMPLATES_DIR) -type f -name '*.json' -print)

include common.mk

# ==============================================================
# PHASE 1: Generate hpp headers from templates, defaults and target overrides
# Example: json_specs/blas/targets/a64fx/axpby_c32.json --> hpp_specs/blas/a64fx/axpby_c32.hpp

# Requires the following scripts:
GENERATE_TUNED_SPEC_PATH=$(SCRIPTS_DIR)/generate_tuned_spec.py
GENERATE_LIVE_TARGET_PATH=$(SCRIPTS_DIR)/generate_live_target_routine_specs.py
GENERATE_INCLUDES_PATH=$(SCRIPTS_DIR)/gen_includes.py
SPECS_PATH := $(SCRIPTS_DIR)/specs.py
PYTHON_SOURCES := $(shell find $(SCRIPTS_DIR) -type f -name '*.py')

# Rebuild when templates or defaults are added or removed.
# Track the package directory too, since defaults/ may not exist.
INPUT_DIRS := $(wildcard \
	$(JSON_SPECS_DIR)/$(PACKAGE) \
	$(PKG_JSON_TEMPLATES_DIR) \
	$(PKG_JSON_DEFAULTS_DIR))

# specs.py lists routine_datatype=path entries. Split each into its two parts.
spec_key = $(word 1,$(subst =, ,$(1)))
spec_path = $(word 2,$(subst =, ,$(1)))

# Choose the template, default or machine file for each spec.
# Stop Make if specs.py fails instead of using an incomplete list.
define target_specs
SPECS_$(1) := $$(shell $(PYTHON) $(SPECS_PATH) list \
	--templates-dir $(PKG_JSON_TEMPLATES_DIR) \
	--defaults-dir $(PKG_JSON_DEFAULTS_DIR) \
	--target-dir $(PKG_JSON_SPECS_DIR)/$(1) || echo SPEC_DISCOVERY_FAILED)

ifneq ($$(filter SPEC_DISCOVERY_FAILED,$$(SPECS_$(1))),)
$$(error Failed to discover specs for $(PACKAGE)/$(1))
endif

# Removing a machine file must rebuild its header using the default.
INPUT_DIRS_$(1) := $(INPUT_DIRS) \
	$(wildcard $(PKG_JSON_SPECS_DIR) $(PKG_JSON_SPECS_DIR)/$(1))
endef

$(foreach target,$(TARGETS),$(eval $(call target_specs,$(target))))

HPP_SPEC_PATHS := $(foreach target,$(TARGETS),$(foreach spec,$(SPECS_$(target)),\
	$(PKG_HPP_SPECS_DIR)/$(target)/$(call spec_key,$(spec)).hpp))

# Live readers use expanded templates, overridden by optional authored defaults.
ifeq ($(live), 1)
LIVE_SPECS := $(shell $(PYTHON) $(SPECS_PATH) list \
	--templates-dir $(PKG_JSON_TEMPLATES_DIR) \
	--defaults-dir $(PKG_JSON_DEFAULTS_DIR) || echo SPEC_DISCOVERY_FAILED)

ifneq ($(filter SPEC_DISCOVERY_FAILED,$(LIVE_SPECS)),)
$(error Failed to discover live specs for $(PACKAGE))
endif

HPP_SPEC_PATHS += $(foreach spec,$(LIVE_SPECS),\
	$(PKG_HPP_SPECS_DIR)/live_targets/$(call spec_key,$(spec)).hpp)
endif

HPP_SPEC_DIRS := $(sort $(dir $(HPP_SPEC_PATHS)))

# Each target has a tuned_specs.hpp listing its routine headers.
INCLUDE_PATHS=$(addsuffix tuned_specs.hpp, $(HPP_SPEC_DIRS))

.PHONY: hpp-specs
hpp-specs: $(HPP_SPEC_PATHS) $(INCLUDE_PATHS)

# Generate each machine header from its selected template, default or override.
define machine_spec_rule
$(PKG_HPP_SPECS_DIR)/$(1)/$(call spec_key,$(2)).hpp: \
	$(call spec_path,$(2)) $(PYTHON_SOURCES) \
	$(SCRIPTS_DIR)/systems.json $(INPUT_DIRS_$(1))
	mkdir -p $$(@D)
	@echo Generating $$@ from $$<
	$(PYTHON) $(GENERATE_TUNED_SPEC_PATH) \
		-i $$< --spec-key $(call spec_key,$(2)) --target $(1) --package $(PACKAGE) | \
		($(CLANG_FORMAT) || cat) > $$@
endef

$(foreach target,$(TARGETS),$(foreach spec,$(SPECS_$(target)),\
	$(eval $(call machine_spec_rule,$(target),$(spec)))))

# One template can supply several datatypes. Give each header its own rule.
define live_spec_rule
$(PKG_HPP_SPECS_DIR)/live_targets/$(call spec_key,$(1)).hpp: \
	$(call spec_path,$(1)) $(PYTHON_SOURCES) $(INPUT_DIRS)
	mkdir -p $$(@D)
	@echo Generating $$@ from $$<
	$(PYTHON) $(GENERATE_LIVE_TARGET_PATH) \
		-i $$< --spec-key $(call spec_key,$(1)) --package $(PACKAGE) -t hpp | \
		($(CLANG_FORMAT) || cat) > $$@
endef

$(foreach spec,$(LIVE_SPECS),$(eval $(call live_spec_rule,$(spec))))

# List the routine headers for each target in tuned_specs.hpp.
# Track input directories so removing a spec also removes its include.
define includes_rule
$(PKG_HPP_SPECS_DIR)/$(1)/tuned_specs.hpp: \
	$(filter $(PKG_HPP_SPECS_DIR)/$(1)/%,$(HPP_SPEC_PATHS)) $(GENERATE_INCLUDES_PATH) \
	$(INPUT_DIRS) \
	$(if $(filter-out live_targets,$(1)),$(wildcard $(PKG_JSON_SPECS_DIR) $(PKG_JSON_SPECS_DIR)/$(1)))
	$(PYTHON) $(GENERATE_INCLUDES_PATH) $(PACKAGE) $(1) $$@ \
		--templates-dir "$(PKG_JSON_TEMPLATES_DIR)" \
		--defaults-dir "$(PKG_JSON_DEFAULTS_DIR)" \
		$(if $(filter-out live_targets,$(1)),--target-dir "$(PKG_JSON_SPECS_DIR)/$(1)")
endef

$(foreach target,$(notdir $(patsubst %/,%,$(HPP_SPEC_DIRS))),\
	$(eval $(call includes_rule,$(target))))

# ==============================================================
# PHASE 2: Declare routine spec types from the templates

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
