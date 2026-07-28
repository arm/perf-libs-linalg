# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import json
import sys

for a in sys.argv[1:]:
	with open(a) as f:
		print(" ".join([ ar for ar in json.load(f)["architectures"] ]))
