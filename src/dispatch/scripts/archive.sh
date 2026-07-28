#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


AR=$1
ARGS=$2
OUTPUT_PATH=$3

tmp_dir=$(mktemp -d)

new_file_paths=" "

for ((i = 4; i <= $#; i++)); do
	#hash the path of the object
	file_suffix="$(echo "${!i}" | cksum | cut -f1 -d' ')"
	#mangle that has into the object name
	filename="$(basename ${!i})"
	new_filename="${filename%.*}_${file_suffix}.${filename##*.}"

	#copy new filename to tmp dir
	new_file_path="${tmp_dir}/${new_filename}"
	new_file_paths+="${new_file_path} "

	cp "${!i}" "${new_file_path}"
done

${AR} "${ARGS}" ${OUTPUT_PATH} ${new_file_paths}

rm -rf "${tmp_dir}"
