#include "ATen/ops/zeros_like.h"
#include "c10/core/ScalarType.h"
#include <ATen/core/TensorAccessor.h>
#include <cutlass/cutlass.h>
#include <torch/extension.h>

#include "common/dispatch.h"

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
__global__ void flash_attn_fwd_kernel(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> Q,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> K,
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> V,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> O) {

  const int seqlen_m = Q.size(-1); // number of queries
  const int seqlen_n = K.size(-1); // number of keys/values

  const int batch_idx = threadIdx.x;
  const int head_idx = threadIdx.y;
  const int seq_chunk_m_idx = threadIdx.z;

  const int start_m = seq_chunk_m_idx * BLOCK_M;

  for (int seq_chunk_n_start = 0; seq_chunk_n_start < seqlen_n;
       seq_chunk_n_start += BLOCK_N) {
    ;
  }
}

at::Tensor flash_attn_fwd_cuda(at::Tensor Q, at::Tensor K, at::Tensor V) {
  const int head_dim = Q.size(-1);
  const float sm_scale = 1.0 / sqrt(head_dim);

  at::Tensor O = torch::zeros_like(Q);

  const int batch_sz = O.size(0);
  const int n_heads = O.size(1);
  const int seqlen_m = O.size(2);

  constexpr int BLOCK = 128;
  const int n_blocks_m = (seqlen_m + BLOCK - 1) / BLOCK;

  AT_DISPATCH_HALF_TYPES(
      Q.scalar_type(), "flash_attn_fwd", ([&] {
        // TODO. double check block dim
        flash_attn_fwd_kernel<scalar_t, BLOCK, BLOCK>
            <<<dim3(batch_sz, n_heads, n_blocks_m), dim3(BLOCK, BLOCK)>>>(
                Q.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                K.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                V.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>());
      }));

  return O;
}
