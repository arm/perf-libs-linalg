#!/usr/bin/env python3
# vim: set et sw=4 sts=4 fileencoding=utf-8:
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


# Allows us to use print(..., file=sys.stderr) to print to standard error when
# using Python 2.
from __future__ import print_function

import argparse
import sys
from os import path as ospath
from pathlib import Path, PurePosixPath

# Load modules from the current directory
from atomic_gemm_sve import valid_atomic_kern_params_sve, generate_source_files_sve
from atomic_gemm_neon import valid_atomic_kern_params_neon, generate_source_files_neon
from atomic_gemm_common import *

def convert_to_posix_path(path, target_os):
    # If we are building a 'windows' or 'windows_arm64ec' target...
    if target_os in ('windows', 'windows_arm64ec'):
        windows_path = str(PurePosixPath(Path(path).resolve()))
        # prepend leading /, convert drive name to lower case, and remove :\
        # This only works with wenv style path, i.e. /c/home/work/directory
        posix_path = "/" + windows_path[0].lower() + windows_path[3:]
        return PurePosixPath(posix_path)
    else:
        return PurePosixPath(path)

# We convert paths to POSIX style only when generating makefile include.
# We still need Windows style path when opening files in Python on Windows.
def generate_makefile(python_dir, make_output_filename, source_filenames, sve, target_os):
    with open(make_output_filename, "w") as f:
        f.write("atomic_kernel_sources := ")
        for fn in source_filenames:
            f.write(f"\\\n  $(BUILD_DIR)/{ospath.basename(fn)}")
        f.write('\n.PRECIOUS: $(atomic_kernel_sources)\n\n')

        script_name = convert_to_posix_path(__file__, target_os)
        # Fix the python dir path if on windows to be unix style
        fix_python_dir= str(convert_to_posix_path(python_dir, target_os))
        dirname = script_name.parents[0]
        script_name = ospath.basename(str(script_name))
        # Get the subdirectory of PYTHON_DIR in which these python scripts live and construct path using $(PYTHON_DIR)
        scripts_dir = "$(PYTHON_DIR)/"+str(dirname).replace(fix_python_dir,'')
        neon_fn = "$(BUILD_DIR)/atomic_neon_%gemm_kernels.S"
        sve_fn = "$(BUILD_DIR)/atomic_sve_%gemm_kernels.S "

        if sve:
            f.write(f"{sve_fn} {neon_fn}: {scripts_dir}/{script_name} {scripts_dir}/atomic_gemm_sve.py {scripts_dir}/atomic_gemm_neon.py\n")
        else:
            f.write(f"{neon_fn}: {scripts_dir}/{script_name} {scripts_dir}/atomic_gemm_neon.py\n")

        f.write(f"\t{scripts_dir}/{script_name} $(BUILD_DIR) {'--sve ' if sve else ''}--dtype $* --target-os {target_os}\n\n")

        if sve:
            # Write rules for building dispatch file and SVE kernels with SVE flags
            sve_flags = "-march=armv8-a+sve"
            f.write("$(BUILD_DIR)/atomic_sve_%_kernels$(OBJECT_SUFFIX): $(BUILD_DIR)/atomic_sve_%_kernels.S\n")
            f.write("\t$(SVE_CC) -c $< -o $@ $(CENTRAL_SVE_ASFLAGS) {} -MMD -MP\n\n".format(sve_flags))
            f.write("$(BUILD_DIR)/atomic_kernels_%$(OBJECT_SUFFIX): $(BUILD_DIR)/atomic_kernels_%.cpp\n")
            f.write("\t$(CXX) -c $< -o $@ $(CENTRAL_CXXFLAGS) {} -MMD -MP\n".format(sve_flags))

def write_to_dispatch_file(f, params_list, sve):
    if not params_list:
        return

    for params in params_list:
        # We don't need a forward declaration of real valued kernels
        # with complex conjugation. These are dealt with and called through
        # to the real valued functions in the dispatch code
        if not params.dtype.is_complex and (params.trans.conja or params.trans.conjb):
            continue
        f.write("{}\n".format(params.function_signature))

    dtypes = {param.dtype for param in params_list}
    suffix = "sve" if sve else "neon"
    for dtype in dtypes:
        f.write(
            'template<>\n'
            'bool perflibs::dispatch_atomic_{1}<{0}>('
            'perflibs::linalg::perflibs_trans transa, perflibs::linalg::perflibs_trans transb, '
            'kernel_inttype m, kernel_inttype n, kernel_inttype k, '
            'const {0} *a, kernel_inttype lda, const {0} *b, '
            'kernel_inttype ldb, {0} *c, kernel_inttype ldc, '
            '{0} alpha, {0} beta) {{\n'.format(dtype.c_type, suffix))
        kernel_m = "m"
        if sve:
            # Get the number of vectors that can hold m
            kernel_m = "m_vecs"
            f.write("\tuint64_t els_per_vec;\n")
            f.write("\tasm(\"cnt{} %[out]\" : [out]\"=r\" (els_per_vec));\n".format(dtype.suffix))
            if dtype.is_complex:
                # There are half as many els_per_vec in complex cases
                f.write("\tels_per_vec /= 2;\n")
            f.write("\tauto m_vecs = (m + els_per_vec - 1) / els_per_vec;\n")

        params_for_dtype = [
            params for params in params_list if params.dtype == dtype
        ]
        # Create a set to ensure non duplicate entries, then
        # create a sorted list from the set so that the generated code is the same each time
        for trans in sorted({param.trans for param in params_for_dtype}):
            if dtype.is_complex:
                f.write("\tif (transa == {} && transb == {}) {{\n".format(
                    trans.perflibs_constants[0], trans.perflibs_constants[1]))
            else:
                # Skip conjugate transpose cases which will be handled together with
                # transpose cases for non-complex datatype
                if trans.conja or trans.conjb:
                    continue
                else:
                    if not trans.transa and not trans.transb:
                        f.write("\tif (transa == {} && transb == {}) {{\n".format(
                            trans.perflibs_constants[0], trans.perflibs_constants[1]))
                    elif trans.transa and not trans.transb:
                        f.write("\tif ((transa == {} || transa == {}) && transb == {}) {{\n".format(
                            "perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_CONJTRANS", trans.perflibs_constants[1]))
                    elif not trans.transa and trans.transb:
                        f.write("\tif (transa == {} && (transb == {} || transb == {})) {{\n".format(
                            trans.perflibs_constants[0], "perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_CONJTRANS"))
                    elif trans.transa and trans.transb:
                        f.write("\tif ((transa == {} || transa == {}) && (transb == {} || transb == {})) {{\n".format(
                            "perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_CONJTRANS", "perflibs::linalg::PERFLIBS_TRANS", "perflibs::linalg::PERFLIBS_CONJTRANS"))

            params_for_trans = [
                params for params in params_for_dtype
                if params.trans == trans
            ]
            m_params = {(param.m_vecs if sve else param.m) for param in params_for_trans}
            f.write("\t\tif ({} > {}) return false;\n".format(kernel_m,max(m_params)))
            for m in m_params:
                f.write("\t\tif ({} == {}) {{\n".format(kernel_m, m))
                params_for_m = [
                    params for params in params_for_trans if (params.m_vecs == m if sve else params.m == m)
                ]
                n_params = {param.n for param in params_for_m}
                f.write("\t\t\tif (n > {}) return false;\n".format(max(n_params)))
                for n in n_params:
                    f.write("\t\t\tif (n == {}) {{\n".format(n))
                    params_for_n = [
                        params for params in params_for_m if params.n == n
                    ]
                    k_params = {param.k for param in params_for_n}
                    f.write("\t\t\t\tif (k > {}) return false;\n".format(max(k_params)))
                    for param in params_for_n:
                        f.write("\t\t\t\tif (k == {}) {{\n".format(param.k))
                        argslist = "a, lda, b, ldb, c, ldc, alpha, beta" if not sve else "a, lda, b, ldb, c, ldc, m, alpha, beta"
                        if (not dtype.is_complex and (trans in [
                                Transpose.CN, Transpose.NC, Transpose.CC, Transpose.TC, Transpose.CT
                        ])):
                            param.trans = trans.equiv_tran
                        f.write((
                            '\t\t\t\t\t{}({});\n'
                        ).format(param.routine_name, argslist))
                        f.write('\t\t\t\t\treturn true;\n')
                        f.write("\t\t\t\t}\n")
                    f.write('\t\t\t}\n')
                f.write('\t\t}\n')
            f.write('\t}\n')
        f.write('\treturn false;\n}\n')

def generate_dispatch(dispatch_output_fn, params_list_neon, params_list_sve):
    with open(dispatch_output_fn, "w") as f:
        f.write('#include "atomic_kernels.hpp"\n')
        f.write('\n')
        write_to_dispatch_file(f, params_list_neon, False)
        write_to_dispatch_file(f, params_list_sve, True)

def generate_source_filenames(dtypes, source_file_dest, sve):
    source_filenames = {
        dtype: Path(
                source_file_dest,
                "atomic_{}{}gemm_kernels.S".format("sve_" if sve else "neon_", dtype.name)
        ).resolve() for dtype in dtypes
    }
    return source_filenames


def generate_source_files(atom_kern_params_list_neon, atom_kern_params_list_sve, source_filenames_neon, source_filenames_sve):
    """Write a source file for each type containing all kernels for that type
    """
    if atom_kern_params_list_sve:
        generate_source_files_sve(atom_kern_params_list_sve, source_filenames_sve)
    generate_source_files_neon(atom_kern_params_list_neon, source_filenames_neon)

def main(args, argv):

    trans = (list(Transpose.__members__.values()) if not args.trans
             else [Transpose[args.trans]])
    dtypes = (list(Type.__members__.values()) if not args.dtype
              else [Type[args.dtype.lower()]])

    dry_run = args.dry_run or args.makefile or args.dispatch

    valid_params_list_sve = None
    if args.sve:
        valid_params_list_sve, valid_dtypes_sve = valid_atomic_kern_params_sve(dtypes, args.target_os)
    valid_params_list_neon, valid_dtypes_neon = valid_atomic_kern_params_neon(trans, dtypes, args.target_os)

    if not valid_params_list_neon and not valid_params_list_sve:
        print ("No kernels generated, exiting.")
        exit(2)

    num_sve_kernels = len(valid_params_list_sve) if valid_params_list_sve else 0
    num_atomic_kernels = len(valid_params_list_neon) + num_sve_kernels
    print("Found valid parameters for {} {} GEMM atomic kernels ({} sve, {} neon)".format(num_atomic_kernels, [t.c_type for t in dtypes], num_sve_kernels, len(valid_params_list_neon)))
    source_filenames_neon = generate_source_filenames(valid_dtypes_neon, args.output_directory, False)
    source_filenames = list(source_filenames_neon.values())

    source_filenames_sve = None
    if args.sve:
        source_filenames_sve = generate_source_filenames(valid_dtypes_sve, args.output_directory, True)
        source_filenames += list(source_filenames_sve.values())

    if not dry_run:
        generate_source_files(valid_params_list_neon, valid_params_list_sve, source_filenames_neon, source_filenames_sve)

    if args.dry_run:
        for fn in source_filenames:
            print (fn)

    if args.makefile:
        if not args.python_script_dir:
            print("Args error: got --makefile but also needed --python_script_dir to be set!")
            exit(2)
        generate_makefile(args.python_script_dir, args.makefile, source_filenames, args.sve, args.target_os)

    if args.dispatch:
        generate_dispatch(args.dispatch, valid_params_list_neon, valid_params_list_sve)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', help="The directory into which to store output files")
    parser.add_argument('-d', '--dry-run', action='store_true',
        help="Do not generate any assembly files (if --makefile or --dispatch are specified," \
             " these output files are still generated), but print out the names of the assembly" \
             " files which would have been generated.")
    parser.add_argument('--dtype', type=str, choices=['s', 'd', 'c', 'z'],
        help="The data type for which to generate files. If no datatype is specified," \
             " output files are generated for all datatypes.")
    parser.add_argument('--trans', type=str, choices=[
        'NN', 'NT', 'TN', 'TT', 'CN', 'NC', 'CT', 'TC', 'CC'],
        help="The transpose options for matrices A and B (in that order) for which to generate output." \
             " If this is not supplied, all transpose options are used.")
    parser.add_argument('--python_script_dir', help='The directory path to the python directory required if using --makefile.', default=None)
    parser.add_argument('--makefile', help="The path of the makefile to be generated, requires --python_script_dir arg.", default=None)
    parser.add_argument('--dispatch', help="The path of the dispatch file to be generated", default=None)
    parser.add_argument('--sve', action='store_true',
        help="Indicates that kernels should be produced for SVE as well as for Neon.")
    parser.add_argument('--target-os', help="The target operating system to generate kernels for", required=True)
    return parser

if __name__ == '__main__':
    main(build_parser().parse_args(), sys.argv)
