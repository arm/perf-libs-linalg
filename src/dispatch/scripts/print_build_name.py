# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import json
import sys

for p in sys.argv[1:]:
	with open(p) as f:
		print(json.load(f)["build_name"])
