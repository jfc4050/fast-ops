#include <ATen/Dispatch.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void lion_update_kernel(
    scalar_t *__restrict__ param,
    const scalar_t *__restrict__ grad,
    scalar_t *__restrict__ exp_avg,
    const int numel,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay) {
  ;
}

void lion_update(
    at::Tensor param,
    const at::Tensor grad,
    at::Tensor exp_avg,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay) {
  // TODO. assert dtypes
  // TODO. assert contiguous
}
