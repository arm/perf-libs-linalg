# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

import json
import sys
from pathlib import Path

systems = []
for config_path in sys.argv[1:]:
	with Path(config_path).open(encoding="utf-8") as f:
		config = json.load(f)
	systems.extend(
		system
		for arch in config["architectures"].values()
		for system, info in arch["systems"].items()
		if "inherit" not in info
	)

print(" ".join(dict.fromkeys(systems)))
