# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import json
import sys

with open(sys.argv[1]) as f:
	j = json.load(f)

	arch = sys.argv[2] if len(sys.argv) >= 3 else j["default_architecture"]

	flags = j["architectures"][arch]["build_flags"]

print(" ".join(flags))
