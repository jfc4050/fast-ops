#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>

#define AT_DISPATCH_CASE_HALF_TYPES(...)                  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))
