# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent))

import utils
from json2cpp import JSONReader, generate_cpp
from json2cpp.passes import *




@dataclass
class KernelSpecGenerator:
    routine: dict
    file_name: str
    target: str

    def __post_init__(self):
        self.routine_name = self.routine["routine"]
        self.kernel_name = self.routine["tuned_routine_spec"]["kernel"]["kernel_name"]
        self.datatype = self.routine["datatype"]
        self.cpp_type = utils.to_cpp_typename(self.datatype)
        self.is_live = self.target == "live_target"
        self.kernel_type = f"{self.kernel_name}_kernel_t<{self.cpp_type}>*"
        self.return_type = self.kernel_type
        if self.is_live:
            self.return_type = f"std::optional<{self.return_type}>"
        self.is_complex = self.datatype in ["c32", "c64"]
        self.is_generic = self.target == "generic"
        if self.target != "live_target":
            ROOT_DIR = Path(__file__).parents[1]
            systems = utils.read_json(ROOT_DIR / "systems.json")
            self.architecture = systems[self.target]["architecture"]
        self.arch_spec = (
            "ArchitectureSpec"
            if self.is_generic or self.is_live
            else f"{self.architecture}_architecture_spec"
        )

    @staticmethod
    def _case_to_args(case):
        def value(v):
            return str(v).lower() if isinstance(v, bool) else f"value_support::{v}"

        return ", ".join(value(v) for v in case.values())

    @staticmethod
    def _case_to_params(case):
        def value(k, v):
            return f"bool {k}" if isinstance(v, bool) else f"value_support {k}"

        return ", ".join(value(k, v) for k, v in case.items())

    @staticmethod
    def _case_to_str(case):
        return "_".join(k + "_" + str(v).lower() for k, v in case.items())

    @staticmethod
    def _scalar_type(matrix_datatypes):
        return "promote_t<{}, {}, {}>".format(*matrix_datatypes)

    @classmethod
    def _axpby_match_requires(cls, matrix_datatypes, problem_context):
        a_type, _, c_type = matrix_datatypes
        scalar_type = cls._scalar_type(matrix_datatypes)

        return (
            "requires ( "
            "axpby_kernel_tag<StrategyTag>::template "
            f"match<{a_type}, {c_type}, {scalar_type}, {problem_context}>() )"
        )

    def _setup_masks(self):
        """Setup scalar masks for use in the upcoming jump table."""
        # TODO: this is all very heavily hardcoded for axpby; this will need to be re-written
        # if we want to expand this for use with other kernel types
        out = f"""
    using compute_type = promote_t<{self.cpp_type}>;
    const auto incx     = strat_tag.incx(pctx);
    const auto incy     = strat_tag.incy(pctx);
    const auto alpha    = strat_tag.alpha(pctx);
    const auto beta     = strat_tag.beta(pctx);"""

        if self.is_complex:
            out += "const auto conj_x = strat_tag.is_conj(pctx);"

        out += """
    constexpr auto alpha_zero = 0, alpha_one = 1, alpha_all = 2;
    constexpr auto incx_one = 0 << 4, incx_not_zero = 1 << 4, incx_zero = 2 << 4;
    constexpr auto incy_one = 0 << 6, incy_not_zero = 1 << 6, incy_zero = 2 << 6;"""
        if self.is_complex:
            out += "constexpr auto conj_x_false = 0 << 8, conj_x_true = 1 << 8;"

        out += """
    const std::uint8_t alpha_mask   = alpha  == zero<compute_type> ? 0 : alpha == one<compute_type> ? 1 : 2;
    const std::uint8_t incx_mask    = incx   == 1 ? 0 : incx != 0 ? 1 : 2;
    const std::uint8_t incy_mask    = incy   == 1 ? 0 : incy != 0 ? 1 : 2;"""

        if self.is_complex:
            out += "constexpr auto beta_zero = 0 << 2, beta_one = 1 << 2, beta_real = 2 << 2, beta_all = 3 << 2;"
            out += "const std::uint8_t beta_mask = beta == zero<compute_type> && pctx.beta_zero_mode == zero_mode::set ? 0 : beta  == one<compute_type> ? 1 : perflibs::imag(beta) == zero<compute_type> ? 2 : 3;"
            out += "const std::uint8_t conj_x_mask = conj_x == false ? 0 : 1;"
        else:
            out += "constexpr auto beta_zero = 0 << 2, beta_one = 1 << 2, beta_all = 3 << 2;"
            out += "const std::uint8_t beta_mask = beta == zero<compute_type> && pctx.beta_zero_mode == zero_mode::set ? 0 : beta  == one<compute_type> ? 1 : 3;"

        if self.is_complex:
            out += "const std::uint16_t val = alpha_mask | beta_mask << 2 | incx_mask << 4 | incy_mask << 6 | conj_x_mask << 8;"
        else:
            out += "const std::uint8_t val = alpha_mask | beta_mask << 2 | incx_mask << 4 | incy_mask << 6;"

        return out.strip()

    def generate_kernel_spec(self, kernel_spec, array_index=False):
        """Generate get_<routine>_kernel function according to the provided kernel_spec."""

        class Reader(JSONReader):
            def statement(self, s):
                match s:
                    case _, str(v):
                        return Return(self.expr(v))
                    case _:
                        return super().statement(s)

        input_json = {
            "kernel_spec": kernel_spec,
        }
        kernel_table = (
            f"{self.kernel_name}_kernels<{self.arch_spec}, {self.cpp_type}>"
        )
        passes = [KernelIndexer(kernel_table)] if array_index else []
        get_kernel_cases_cpp = generate_cpp(input_json, reader=Reader(), passes=passes)

        matrix_datatypes = utils.get_matrix_types(self.routine["datatype"])
        problem_context = f"problem_context<ProblemContextBase, {self.arch_spec}>"
        requires_clause = self._axpby_match_requires(
            matrix_datatypes, problem_context
        )

        func_params = [ "axpby_kernel_tag<StrategyTag> strat_tag" ]

        if self.is_generic:
            func_name = f"get_spec_system"

            func_params.append(f"const problem_context<ProblemContextBase, ArchitectureSpec>& pctx, system_t<machine::system::{self.target}>")
            template_params = "template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec>"

        elif self.is_live:
            func_name = f"get_{self.kernel_name}_kernel_live"
            func_params.append( f"const {problem_context}& pctx" )

            template_params = "template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec>"
        else:
            func_name = f"get_spec_system"
            func_params.append( f"const {problem_context}& pctx, system_t<machine::system::{self.target}>" )

            template_params = (
                "template<typename StrategyTag, typename ProblemContextBase>"
            )
        func_params = ", ".join(func_params)

        live_getter = ""
        live_fwd = ""
        if not self.is_live:
            live_problem_context = "problem_context<ProblemContextBase, ArchitectureSpec>"
            live_requires_clause = self._axpby_match_requires(
                matrix_datatypes, live_problem_context
            )
            live_guard = self.routine.get(
                "_live_target_guard", "defined(PL_LINALG_LIVE_TARGET)"
            )
            live_fwd = f"""
#if {live_guard}
template<typename StrategyTag, typename ProblemContextBase, typename ArchitectureSpec>
std::optional<{self.kernel_type}> get_{self.kernel_name}_kernel_live(
        axpby_kernel_tag<StrategyTag>,
        const problem_context<ProblemContextBase, ArchitectureSpec>&)
        {live_requires_clause};
#endif
"""
            live_getter = f"""
#if {live_guard}
            if constexpr (is_live()) {{
              if (auto kernel = get_{self.kernel_name}_kernel_live(strat_tag, pctx)) {{
                return *kernel;
              }}
            }}
#endif"""

        masks = self._setup_masks() if not isinstance(kernel_spec, str) else ""

        linkage = "" if self.is_live else "static "

        return f"""
{live_fwd}
{template_params}
{linkage}{self.return_type} {func_name}({func_params})
		{requires_clause} {{
    {live_getter}
    {masks}
    {get_kernel_cases_cpp}
}}"""

    def _generate_live_reader(self):
        default_kernel = self.routine["kernel_spec"]
        return f"""
  template<typename ArchitectureSpec>
  inline auto read_{self.file_name}_kernel_spec(const std::filesystem::path& json_dir, const verbosity_level_t verbosity_level, const std::string& case_str) {{
    static {self.return_type} kernel;
    if (kernel) return kernel;

    const auto json_file_path = json_dir / "{self.file_name}.json";
    static nlohmann::json json;
    static bool searched = false;

    if (json.is_null() && !searched) {{
        if (std::ifstream json_file_stream {{ json_file_path }}; json_file_stream) {{
            json = nlohmann::json::parse(json_file_stream, nullptr, false);
            searched = true;
        }}
    }}

    if (!json.is_null() && !json.is_discarded()) {{
        for (const auto& entry : json) {{
            if (entry["routine"] == "{self.routine_name}" && entry["datatype"] == "{self.datatype}") {{
                const auto& kernel_spec = entry["kernel_spec"];
                if (kernel_spec.type() == nlohmann::json::value_t::object) {{
                    kernel = try_get_kernel<{self.arch_spec}, {self.cpp_type}>(kernel_spec[case_str].get<std::string>()).value().kernel;
                }}
            }}
        }}
    }}
    return kernel;
}}
"""

    def _generate_live_kernels(self):
        live_kernel_template = """
template<typename ArchitectureSpec>
static {kernel_type} live_{file_name}_kernel_{case_str} = read_{file_name}_kernel_spec<ArchitectureSpec>(
        get_live_target_dir(),
        get_verbosity_level(),
        "{case_str}");
"""
        return "\n".join(
            live_kernel_template.format(
                file_name=self.file_name,
                case_str=self._case_to_str(case),
                kernel_type=self.return_type,
            )
            for case in self.routine["cases"]
        )

    def generate_live_header(self):
        kernel_spec = {
            "jump_table": [
                {
                    "condition": case,
                    "value": f"live::live_{self.file_name}_kernel_{self._case_to_str(case)}<ArchitectureSpec>",
                }
                for case in self.routine["cases"]
            ],
        }

        instantiations = "\n".join(
            [
                f"extern {self.return_type} live_{self.file_name}_kernel_{self._case_to_str(case)};"
                for case in self.routine["cases"]
            ]
        )
        return f"""
{self.generate_kernel_spec(kernel_spec)}
"""

    def generate_live_source(self):
        return f"""
        namespace live {{
            {self._generate_live_reader()}
            {self._generate_live_kernels()}
        }}
            {self.generate_live_header()}
        """
