## SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
## SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

## function to produce assembly kernel prologue for a specified operating system
<%def name="prologue(func_name)">
	%if target_os == 'linux':
		.p2align 5
		.global ${func_name}
		.type ${func_name}, %function
		${func_name}:
	%elif target_os == 'mac':
		.p2align 2
		.global _${func_name}               ; -- Begin function ${func_name}
		_${func_name}:
	%elif target_os == 'windows':
		.global ${func_name}
		.def    ${func_name};
		.scl    2;
		.type   32;
		.endef
		${func_name}:
	%elif target_os == 'windows_arm64ec':
		.global "#${func_name}"
		.def    "#${func_name}";
		.scl    2;
		.type   32;
		.endef
		"#${func_name}":
	%else:
		<% raise Exception('Unrecognised operating system target in prologue(): ' + target_os) %>
	%endif
</%def>

## function to produce assembly kernel epilogue for a specified operating system
<%def name="epilogue(func_name)">
	%if target_os == 'linux':
		.size ${func_name}, .-${func_name}
	%elif target_os == 'mac':
	%elif target_os == 'windows':
	%elif target_os == 'windows_arm64ec':
	%else:
		<% raise Exception('Unrecognised operating system target in epilogue(): ' + target_os) %>
	%endif
</%def>

## function to insert the correct symbol for assert for a specified operating system
<%def name="assert_fn()">
	%if target_os == 'linux':
		bl __assert
	%elif target_os == 'mac':
		bl ___assert_rtn
	%elif target_os in ('windows', 'windows_arm64ec'):
		bl _wassert
	%else:
		<% raise Exception('Unrecognised operating system target in assert_fn(): ' + target_os) %>
	%endif
</%def>
