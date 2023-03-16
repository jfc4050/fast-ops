#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#define AT_DISPATCH_CASE_HALF_TYPES(...)                                       \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_HALF_TYPES(TYPE, NAME, ...)                                \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))

template <typename scalar_t_pt> struct cutlass_t;

template <> struct cutlass_t<c10::Half> { using value = cutlass::half_t; };

template <> struct cutlass_t<c10::BFloat16> {
  using value = cutlass::bfloat16_t;
};
