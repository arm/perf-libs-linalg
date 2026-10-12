/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)
 */

#ifndef PERFLIBS_EXTRA_JSON_COMPLEX_H
#define PERFLIBS_EXTRA_JSON_COMPLEX_H

#include <nlohmann/json.hpp>

#include <complex>
namespace std {
  template<typename BasicJsonType, typename FloatType>
  void to_json(BasicJsonType& j, complex<FloatType> val) noexcept {
    j = {val.real(), val.imag()};
  }

  template<typename BasicJsonType, typename FloatType>
  void from_json(const BasicJsonType& j, complex<FloatType>& p) {
    p.real(j.at(0).template get<FloatType>());
    p.imag(j.at(1).template get<FloatType>());
  }
}

#endif
