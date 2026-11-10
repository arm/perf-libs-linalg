/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

/*
 * Read benchmark requests from standard input or named files.
 * Pass each input stream to run_context_benchmark() in context_runner.cpp.
 *
 * In FILE mode, report file-open errors on standard error. Try the remaining
 * files even if one fails to open, parse, or run. Return failure if any file fails.
 */

#include "context_runner.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>

int main(int argc, char **argv) {
	const auto usage = [&] {
		std::cerr << "USAGE: " << argv[0] << " STDIN | FILE <file1> ... <fileN>\n";
	};

	if(argc < 2) {
		usage();
		return EXIT_FAILURE;
	}

	const std::string_view input_type { argv[1] };

	if(input_type == "STDIN" && argc == 2) {
		return perflibs::linalg::run_context_benchmark(std::cin) ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	else if(input_type == "FILE" && argc > 2) {
		bool success = true;
		for(const std::filesystem::path file_path : std::span ( argv + 2, argc - 2 )) {
			std::ifstream file { file_path };
			if(!file) {
				std::cerr << "Cannot open " << file_path << '\n';
				success = false;
				continue;
			}
			if(!perflibs::linalg::run_context_benchmark(file)) {
				success = false;
			}
		}
		return success ? EXIT_SUCCESS : EXIT_FAILURE;
	}
	else {
		usage();
		return EXIT_FAILURE;
	}
}
