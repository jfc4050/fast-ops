#include <ATen/Dispatch.h>
#include <torch/extension.h>

#define PRAGMA_UNROLL _Pragma("unroll")

template <typename scalar_t> __device__ scalar_t sign(scalar_t x) {
  int t = x < 0 ? -1 : 0;
  return x > 0 ? 1 : t;
}

template <typename scalar_t, typename momentum_t>
__global__ void lion_update_kernel(
    scalar_t *__restrict__ param,
    const scalar_t *__restrict__ grad,
    momentum_t *__restrict__ exp_avg,
    const int numel,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay) {

  // 128bit/16byte access per thread
  constexpr int ACCESS_N = 16 / sizeof(scalar_t);
  using AccessType = scalar_t[ACCESS_N];

  AccessType *param_accesses = reinterpret_cast<AccessType *>(param);
  const AccessType *grad_accesses = reinterpret_cast<AccessType *>(grad);
  AccessType *momentum_accesses = reinterpret_cast<AccessType *>(grad);

  const scalar_t weight_decay_factor = 1.0 - lr * weight_decay;
  const scalar_t beta1_complement = 1.0 - beta1;
  const scalar_t neg_lr = -lr;
  for (int thread_start = blockIdx.x * blockDim.x + threadIdx.x * ACCESS_N;
       thread_start < numel;
       thread_start += blockDim.x) {
    AccessType param_access = param_accesses[thread_start];
    AccessType grad_access = grad_accesses[thread_start];
    AccessType momentum_access = momentum_accesses[thread_start];

    // apply weight decay
    // TODO. make sure this vectorizes
    PRAGMA_UNROLL
    for (int i = 0; i < ACCESS_N; ++i) {
      param_access[i] *= weight_decay_factor;
    }

    // compute update
    AccessType update = momentum_access;
    PRAGMA_UNROLL
    for (int i = 0; i < ACCESS_N; ++i) {
      update[i] *= beta1;
    }
    PRAGMA_UNROLL
    for (int i = 0; i < ACCESS_N; ++i) {
      update[i] = beta1_complement * grad_access[i] + update[i];
    }
    PRAGMA_UNROLL
    for (int i = 0; i < ACCESS_N; ++i) {
      update[i] = sign(update[i]);
    }

    // apply update
    PRAGMA_UNROLL
    for (int i = 0; i < ACCESS_N; ++i) {
      param_access[i] = update[i] * neg_lr + param_access[i];
    }

    // write back
    *param_accesses[thread_start] = param_access;

    // decay momentum
    for (int i = 0; i < ACCESS_N; ++i) {
      ;
    }

  }
}

void lion_update(
    at::Tensor param,
    const at::Tensor grad,
    at::Tensor exp_avg,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay) {

  const int param_numel = param.numel();
  // TODO. assert dtypes
  // TODO. assert contiguous
}
