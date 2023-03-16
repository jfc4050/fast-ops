#include "ATen/ops/zeros_like.h"
#include "c10/core/ScalarType.h"
#include <ATen/core/TensorAccessor.h>
#include <cutlass/cutlass.h>
#include <torch/extension.h>

#include "common/dispatch.h"

template <typename scalar_t>
__global__ void flash_attn_fwd_kernel(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> Q,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> K,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> V,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> O) {
  O[0][0][0][0] = 100;
}

at::Tensor flash_attn_fwd_cuda(at::Tensor Q, at::Tensor K, at::Tensor V) {
  const int head_dim = Q.size(-1);
  const float sm_scale = 1.0 / sqrt(head_dim);

  at::Tensor O = torch::zeros_like(Q);

  AT_DISPATCH_HALF_TYPES(
      Q.scalar_type(), "flash_attn_fwd", ([&] {
        // TODO. update blocks and threads
        flash_attn_fwd_kernel<scalar_t><<<1, 1>>>(
            Q.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
            K.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
            V.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
            O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>());
      }));

  return O;
}
