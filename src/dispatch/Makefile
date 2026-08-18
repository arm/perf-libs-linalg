# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

SHELL := /bin/bash -e -o pipefail

FORMAT=clang-format -i
PYTHON ?= python3

ifeq ($(GENERATE_DIR),)
$(error GENERATE_DIR not specified!)
endif

ifeq ($(BUILD_DIR),)
$(error BUILD_DIR not specified!)
endif

ifneq (,$(filter build-config,$(MAKECMDGOALS)))
ifeq ($(BUILD_LIB),)
$(error BUILD_LIB not specified!)
endif
endif

ifeq ($(CONFIG_JSON_PATH),)
$(error CONFIG_JSON_PATH not specified!)
endif

ifeq ($(FIND),)
$(error FIND not specified!)
endif

SOURCE_DIR          := $(abspath .)
SCRIPTS_DIR          = $(SOURCE_DIR)/scripts
SPECS_DIR            = $(SOURCE_DIR)/specs
SPECS_BUILD_DIR      = $(GENERATE_DIR)/hpp_specs

include $(SOURCE_DIR)/kernel_specs/kernel_specs.mk

CPP_FILES_TO_COPY    = $(patsubst $(SOURCE_DIR)%,$(GENERATE_DIR)%,$(shell $(FIND) $(SOURCE_DIR) -regex ".*\.cpp"))
HPP_FILES_TO_COPY    = $(patsubst $(SOURCE_DIR)%,$(GENERATE_DIR)%,$(shell $(FIND) $(SOURCE_DIR) -regex ".*\.hpp"))
OBJ_FROM_COPIED_CPP  = $(patsubst $(SOURCE_DIR)%.cpp,$(BUILD_DIR)%.o,$(shell $(FIND) $(SOURCE_DIR) -regex ".*\.cpp"))

.PHONY: build-specs
.PRECIOUS: build-specs
build-specs:
	$(MAKE) -C $(SPECS_DIR) HPP_SPECS_DIR=$(SPECS_BUILD_DIR) PACKAGE=blas SYSTEMS="$(SYSTEMS)"
	$(MAKE) -C $(SPECS_DIR) HPP_SPECS_DIR=$(SPECS_BUILD_DIR) PACKAGE=factorization SYSTEMS="$(SYSTEMS)"
	$(MAKE) -C $(SPECS_DIR) HPP_SPECS_DIR=$(SPECS_BUILD_DIR) PACKAGE=strategy_selection SYSTEMS="$(SYSTEMS)"

TUNED_SPECS_DIR = $(SPECS_BUILD_DIR)

.PRECIOUS: $(GENERATE_DIR)/%.cpp
$(GENERATE_DIR)/%.cpp: $(SOURCE_DIR)/%.cpp
	mkdir -p $(@D)
	cp $< $@

.PRECIOUS: $(GENERATE_DIR)/%.hpp
$(GENERATE_DIR)/%.hpp: $(SOURCE_DIR)/%.hpp
	mkdir -p $(@D)
	cp $< $@


CHOOSER_SOURCE_DIR          = $(SOURCE_DIR)/chooser
CHOOSER_JSON_DIR            = $(CHOOSER_SOURCE_DIR)/json
CHOOSER_GENERATE_DIR        = $(GENERATE_DIR)/chooser
CHOOSER_BUILD_DIR           = $(BUILD_DIR)/chooser
CHOOSER_JSON_PATHS          = $(shell $(FIND) $(CHOOSER_JSON_DIR)/ -regex ".*.json")
CHOOSER_CPP_PATHS           = ${patsubst ${CHOOSER_JSON_DIR}/%,${CHOOSER_GENERATE_DIR}/%,${CHOOSER_JSON_PATHS:.json=.cpp}}
CHOOSER_OBJ_PATHS           = ${patsubst ${CHOOSER_JSON_DIR}/%,${CHOOSER_BUILD_DIR}/%,${CHOOSER_JSON_PATHS:.json=.o}}
CHOOSER_GENERATED_HPP_PATHS = $(CHOOSER_GENERATE_DIR)/get_architecture.hpp

ARCHITECTURES               = $(shell $(PYTHON) $(SCRIPTS_DIR)/print_architectures.py $(CONFIG_JSON_PATH))
SYSTEMS                     = $(shell $(PYTHON) $(SCRIPTS_DIR)/print_systems.py $(CONFIG_JSON_PATH))
ARCHITECTURES_SOURCE_DIR    = $(SOURCE_DIR)/architectures
ARCHITECTURES_GENERATE_DIR  = $(GENERATE_DIR)/architectures
ARCHITECTURES_BUILD_DIR     = $(BUILD_DIR)/architectures
ARCHITECTURES_JSON_DIR      = $(ARCHITECTURES_SOURCE_DIR)/json
ARCHITECTURES_JSON_PATHS    = $(shell $(FIND) $(ARCHITECTURES_JSON_DIR) -regex ".*\.json")
ARCHITECTURES_OBJ_PATHS     = $(foreach arch,$(ARCHITECTURES),${ARCHITECTURES_JSON_PATHS:$(ARCHITECTURES_JSON_DIR)/%.json=$(ARCHITECTURES_BUILD_DIR)/$(arch)/%.o})
ARCHITECTURES_PRIORITY_OBJ_PATHS = $(foreach arch,$(ARCHITECTURES),$(filter $(ARCHITECTURES_BUILD_DIR)/$(arch)/linalg/blas/%,$(ARCHITECTURES_OBJ_PATHS)))
ARCHITECTURES_REMAINING_OBJ_PATHS = $(filter-out $(ARCHITECTURES_PRIORITY_OBJ_PATHS),$(ARCHITECTURES_OBJ_PATHS))
ARCHITECTURES_GET_TRS_PATHS = $(foreach arch,$(ARCHITECTURES),$(ARCHITECTURES_GENERATE_DIR)/$(arch)/linalg/get_spec.hpp)

ALL_HPP_PATHS = ${CHOOSER_GENERATED_HPP_PATHS} ${HPP_FILES_TO_COPY} ${ARCHITECTURES_GET_TRS_PATHS} ${ARCHITECTURES_GENERATE_DIR}/linalg/specs.hpp ${KERNEL_SPECS_HPP_PATHS}
ALL_CPP_PATHS = ${CPP_FILES_TO_COPY} ${CHOOSER_CPP_PATHS} ${KERNEL_SPECS_CPP_PATHS}
ALL_OBJ_PATHS = $(ARCHITECTURES_PRIORITY_OBJ_PATHS) $(OBJ_FROM_COPIED_CPP) $(CHOOSER_OBJ_PATHS) $(KERNEL_SPECS_OBJ_PATHS) $(ARCHITECTURES_REMAINING_OBJ_PATHS)

SOURCE_DEPENDENCIES  = ${OBJ_FROM_COPIED_CPP:.o=.d}
SOURCE_DEPENDENCIES += ${CHOOSER_OBJ_PATHS:.o=.d}
SOURCE_DEPENDENCIES += ${ARCHITECTURES_OBJ_PATHS:.o=.d}
SOURCE_DEPENDENCIES += ${KERNEL_SPECS_OBJ_PATHS:.o=.d}
-include $(SOURCE_DEPENDENCIES)


CONFIG_CXXFLAGS=$(NO_TARGET_COPTS) -std=c++23 -Wall -Wno-error=unknown-pragmas -MMD -fno-rtti -fno-exceptions -nostdlib

# -Wmaybe-uninitialized often reports false positives
ifeq ($(COMPILER_NAME), gfortran)
	CONFIG_CXXFLAGS += -Wno-maybe-uninitialized
endif

ifeq ($(COMPILER_NAME), gfortran)
	CONFIG_CXXFLAGS += -fcx-fortran-rules
else ifeq ($(COMPILER_NAME), nvfortran)
	CONFIG_CXXFLAGS += -fcx-fortran-rules
endif

ifneq ($(COMPILER_NAME),nvfortran)
	CONFIG_CXXFLAGS += -MP
endif

ifneq (,$(filter $(TARGET_OS),windows windows_arm64ec))
	# This suppress warning STL4037: The effect of instantiating the template std::complex
	# for any type other than float, double, or long double is unspecified.
	CONFIG_CXXFLAGS+=-D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
	# Suppress security warnings for old CRT functions
	CONFIG_CXXFLAGS+=-D_CRT_SECURE_NO_WARNINGS
	# Suppress deprecation warnings for POSIX function names
	CONFIG_CXXFLAGS+=-D_CRT_NONSTDC_NO_WARNINGS
	# This is needed for _Fcomplex and _Dcomplex when compiling c++ >= 17 code
	CONFIG_CXXFLAGS+=-D_CRT_USE_C_COMPLEX_H

	ifeq ($(TARGET_OS), windows_arm64ec)
		CONFIG_CXXFLAGS+=--target=arm64ec-pc-windows-msvc
	endif
else
	CONFIG_CXXFLAGS +=-fPIC
endif

ifeq ($(use_openmp), 1)
	CONFIG_CXXFLAGS+=-fopenmp
	ifneq (,$(filter $(TARGET_OS),windows windows_arm64ec))
		CONFIG_CXXFLAGS+=-D_OPENMP_LLVM_RUNTIME
	endif
endif

ifeq ($(int64), 1)
	CONFIG_CXXFLAGS+=-DINTEGER64
endif

ifeq ($(debug), 1)
	CONFIG_CXXFLAGS+=-g -DPERFLIBS_LINALG_DEBUG=1
else
	CONFIG_CXXFLAGS+=-O3 -funroll-loops
endif

ifeq ($(asserts), 1)
	CONFIG_CXXFLAGS+=-DPERFLIBS_ASSERT_ON=1
endif

ifeq ($(coverage), 1)
	CONFIG_CXXFLAGS+=-fprofile-arcs -ftest-coverage
endif

ifeq ($(memcheck), 1)
	CONFIG_CXXFLAGS+=-g -fsanitize=address
endif

ifeq ($(live), 1)
	CONFIG_CXXFLAGS+=-DPL_LINALG_LIVE_TARGET=1
endif

CONFIG_CXXOPTS+=$(CONFIG_CXXFLAGS) \
                $(if $(INCLUDE_DIR),-I$(INCLUDE_DIR)) \
                $(if $(CENTRAL_DIR),-I$(CENTRAL_DIR)/common) \
                -I$(GENERATE_DIR) \
                $(if $(CENTRAL_DIR),-I$(CENTRAL_DIR)/blas/linalg) \
                $(if $(CENTRAL_DIR),-I$(CENTRAL_DIR)/blas/SRC) \
                $(if $(EXTRA_DIR),-I$(EXTRA_DIR)) \
                $(EXTRA_CXXFLAGS) \
                -I$(TUNED_SPECS_DIR) \
                -I$(CHOOSER_GENERATE_DIR)

.PRECIOUS: $(CHOOSER_GENERATE_DIR)/get_architecture.hpp
$(CHOOSER_GENERATE_DIR)/get_architecture.hpp: $(CONFIG_JSON_PATH) $(SCRIPTS_DIR)/generate_get_architecture.py
	mkdir -p $$(dirname $@)
	$(PYTHON) $(SCRIPTS_DIR)/generate_get_architecture.py\
		--config-json-path $(CONFIG_JSON_PATH) \
		--output-file-path $@
	$(FORMAT) $@ || true

.PRECIOUS: $(CHOOSER_GENERATE_DIR)/%.cpp
$(CHOOSER_GENERATE_DIR)/%.cpp: $(CHOOSER_JSON_DIR)/%.json $(SCRIPTS_DIR)/generate_chooser.py
	mkdir -p $$(dirname $@)
	$(PYTHON) $(SCRIPTS_DIR)/generate_chooser.py\
		--routine-json-file-path $<\
		--architectures-json-file-path $(CONFIG_JSON_PATH) \
		--output-file-path $@
	$(FORMAT) $@ || true


.PRECIOUS: $(BUILD_DIR)/%.o
$(BUILD_DIR)/%.o: $(GENERATE_DIR)/%.cpp | $(ALL_HPP_PATHS) build-specs
	@mkdir -p $$(dirname $@)
	@echo "$(CXX) -c $< -o $@ $(CONFIG_CXXOPTS) -MMD -MP" >  $@.sh && chmod +x $@.sh
ifeq ($(dev),1)
	@echo -e "\033[0;35mbuilding\033[0m $@"
	@(time ($@.sh &> $@.txt ) &> $@.time ) \
		&& ( echo -e "\033[0;32mSucceeded to build\033[0m $@ \033[0;34mTime taken:\033[0m $$(sed -n "s/real\s//p" $@.time)" ) \
		|| ( echo -e "\033[0;31mFailed to build\033[0m $@ \nbuild command: $@.sh\nerror: $@.txt" && false )
else
	$(SHELL) -x $@.sh
endif


#generates specs.hpp which in turn includes the files for the architectures supported in the config
.PRECIOUS: $(ARCHITECTURES_GENERATE_DIR)/linalg/specs.hpp
$(ARCHITECTURES_GENERATE_DIR)/linalg/specs.hpp: $(SCRIPTS_DIR)/generate_architecture_specs.py  $(CONFIG_JSON_PATH)
	mkdir -p $(@D)
	$(PYTHON) $(SCRIPTS_DIR)/generate_architecture_specs.py \
		--config-json-path $(CONFIG_JSON_PATH) \
		--output-file-path $@
	$(FORMAT) $@ || true

.PRECIOUS: $(ARCHITECTURES_GENERATE_DIR)/%/linalg/get_spec.hpp
$(ARCHITECTURES_GENERATE_DIR)/%/linalg/get_spec.hpp: $(SCRIPTS_DIR)/generate_get_spec.py $(CONFIG_JSON_PATH)
	mkdir -p $(@D)
	$(PYTHON) $(SCRIPTS_DIR)/generate_get_spec.py \
		--output-file-path $@ \
		--config-json-path $(CONFIG_JSON_PATH) \
		--architecture $* \
		--tuned-routine-spec \
		--kernel waxpby,dot,trsv,trsm,rot,rotg,rotm,rotmg,find_index,l1_norm,l2_norm,swap\
		--kernel-spec gemv
	$(FORMAT) $@ || true

$(ARCHITECTURES_BUILD_DIR)/%.o: CONFIG_CXXOPTS+=-I$(ARCHITECTURES_GENERATE_DIR)

#because make won't let us write rule that have multiple pattern matches we have to stamp out a version of this for each arch
define make_cpp_from_json_paths
.PRECIOUS: $(ARCHITECTURES_GENERATE_DIR)/$1/%.cpp
$(ARCHITECTURES_GENERATE_DIR)/$1/%.cpp: $(ARCHITECTURES_JSON_DIR)/%.json $(SCRIPTS_DIR)/generate_instantiations.py
	mkdir -p $$(@D)
	$(PYTHON) $(SCRIPTS_DIR)/generate_instantiations.py \
		--instantiation-json-file-path $$< \
		--routine-json-dir $(CHOOSER_JSON_DIR)/$$(dir $$(patsubst $(ARCHITECTURES_JSON_DIR)/%,%,$$<)) \
		--architecture-name $1 \
		--output-file-path $$@
	$(FORMAT) $$@ || true

$(ARCHITECTURES_BUILD_DIR)/$1/%.o: CONFIG_CXXOPTS+=$(shell $(PYTHON) $(SCRIPTS_DIR)/print_build_flags.py $(CONFIG_JSON_PATH) $1)
endef

$(foreach arch, $(ARCHITECTURES), $(eval $(call make_cpp_from_json_paths,$(arch))))

.PHONY: print-objects
print-objects:
	@printf '%s\n' $(ALL_OBJ_PATHS)

.PHONY: build-objects
build-objects: build-specs $(ALL_OBJ_PATHS)

ifneq ($(BUILD_LIB),)
$(BUILD_LIB): $(ALL_OBJ_PATHS)
	$(SCRIPTS_DIR)/archive.sh $(AR) r $@ $^
endif

.PHONY: build-config
build-config: build-specs $(BUILD_LIB)

.PHONY: generate-config
generate-config: $(ALL_HPP_PATHS) $(ALL_CPP_PATHS) build-specs

.PHONY: clean-generate
clean-generate:
	rm -rf $(CHOOSER_GENERATE_DIR)
	rm -rf $(ARCHITECTURES_GENERATE_DIR)
	rm -rf $(GENERATE_DIR)

.PHONY: clean-build
clean-build:
	rm -rf $(ALL_OBJ_PATHS)
	rm -rf $(ARCHITECTURES_BUILD_DIR)
	rm -rf $(CHOOSER_BUILD_DIR)
	rm -rf $(BUILD_DIR)

.PHONY: clean-specs
clean-specs:
	$(MAKE) -C $(SPECS_DIR) clean HPP_SPECS_DIR=$(SPECS_BUILD_DIR) PACKAGE=blas
	$(MAKE) -C $(SPECS_DIR) clean HPP_SPECS_DIR=$(SPECS_BUILD_DIR) PACKAGE=factorization
	$(MAKE) -C $(SPECS_DIR) clean HPP_SPECS_DIR=$(SPECS_BUILD_DIR) PACKAGE=strategy_selection

.PHONY: clean
clean: clean-build clean-generate clean-specs
