# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

#ie "typename T1, typename T2, int I"
def make_template_formal_args(args):
	return ", ".join( [ f"{a['type']} {a['name']}" for a in args ] )

#ie "T1, T2, i"
def make_template_actual_args(args):
	return ", ".join( [ a['name'] for a in args ] )


def make_resolved_template_actual_args(routine_data, templates):
	return ", ".join([
		routine_data[ x["name"] ]
			for x in templates ])


def resolve_template(typename, template_types, template_values):
	if typename in [ x["name"] for x in template_types ]:
		return template_values[ typename ]
	else:
		return typename


def make_function_parameter_unresolved(param_data, template_params):
	out = ""
	if param_data["is_const"]:
		out += "const "

	param_type = param_data["type"]

	if param_type == "promote_real":
		param_type = "remove_complex_t<promote_t<{}>>".format(
			", ".join(template_params))

	elif param_type == "promote":
		param_type = "promote_t<{}>".format(
			", ".join(template_params))

	out += f"{param_type} "

	if  param_data["is_pointer"]:
		out += "*"

	if "is_reference" in param_data and param_data["is_reference"]:
		out += "&"

	out += param_data["name"]

	return out

def make_function_parameters_unresolved(params_data, template_params):
	out = []

	for param in params_data:
		out.append(make_function_parameter_unresolved(param, template_params))

	return ", ".join(out)

def promote(types):
	is_complex = False;
	precision=16
	for t in types:
		if t.startswith("c"):
			is_complex = True

		precision=max(int(t[1:]), precision)

	return "f{}{precision".format("c" if is_complex else "r")

def make_function_parameter(param_data, template_types, template_values):
	out = ""
	if param_data["is_const"]:
		out += "const "

	if param_data["type"] == "promote_real":
		param_type = template_values[ param_data["type"] ]
	elif param_data["type"] == "promote":
		param_type = template_values[ param_data["type"] ]
	elif param_data["type"] in [ x["name"] for x in template_types ]:
		param_type = template_values[ param_data["type"] ]
	else:
		param_type = param_data["type"]

	out += f"{param_type} "

	if param_data["is_pointer"]:
		out += "*"

	if "is_reference" in param_data and param_data["is_reference"]:
		out += "&"

	out += param_data["name"]

	return out

def make_function_parameters(params_data, template_types, template_values):
	out = []
	for param in params_data:
		out.append(make_function_parameter(param, template_types, template_values))

	return ", ".join(out)

def make_actual_function_parameters(params_data):
	return ", ".join([ x["name"] for x in params_data ])


