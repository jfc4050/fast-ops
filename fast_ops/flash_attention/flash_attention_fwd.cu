#include <ATen/core/TensorAccessor.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/ScalarType.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <torch/extension.h>

#include "common/launch_utils.h"
#include "cute/config.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/swizzle_layout.hpp"

template <typename scalar_t_pt, int BLOCK_M, int BLOCK_N, int BLOCK_D>
__global__ void flash_attn_fwd_kernel(
    const at::PackedTensorAccessor32<scalar_t_pt, 4, at::RestrictPtrTraits>
        Q_accessor,
    const at::PackedTensorAccessor32<scalar_t_pt, 4, at::RestrictPtrTraits>
        K_accessor,
    const at::PackedTensorAccessor32<scalar_t_pt, 4, at::RestrictPtrTraits>
        V_accessor,
    at::PackedTensorAccessor32<scalar_t_pt, 4, at::RestrictPtrTraits>
        O_accessor) {

  // map PyTorch type to CUTLASS type
  using scalar_t = typename cutlass_t<scalar_t_pt>::value;

  __shared__ scalar_t Qi_smem[BLOCK_M * BLOCK_D];

  const int seqlen_m = Q_accessor.size(3); // number of queries
  const int seqlen_n = K_accessor.size(3); // number of keys/values

  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int seq_chunk_m_idx = blockIdx.z;
  const int start_m = seq_chunk_m_idx * BLOCK_M;

  const int warp_id = threadIdx.x;
  const int lane_id = threadIdx.y;
  const int thread_id = warp_id * blockDim.x + lane_id;

  // represent full tensors
  auto Q =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<scalar_t *>(
                            Q_accessor[batch_idx][head_idx].data())),
                        cute::make_layout(cute::make_shape(Q_accessor.size(3),
                                                           Q_accessor.size(4)),
                                          cute::GenRowMajor{}));
  // TODO. do K and V as well

  // represent SRAM tiles
  auto Qi =
      cute::make_tensor(cute::make_smem_ptr(Qi_smem),
                        cute::make_layout(cute::make_shape(BLOCK_M, BLOCK_D)));
  // make tile to be copied into SRAM
  auto Qi_gmem_tile = cute::local_tile(Q, cute::make_shape(BLOCK_M, BLOCK_D),
                                       cute::make_coord(start_m, 0));
  auto Qi_load_thread_layout =
      cute::make_shape(cute::Int<32>{}, cute::Int<8>{});
  auto Qi_load_partition_gmem =
      cute::local_partition(Qi_gmem_tile, Qi_load_thread_layout, thread_id);
  auto Qi_load_partition_smem =
      cute::local_partition(Qi, Qi_load_thread_layout, thread_id);
  cute::copy(Qi_load_partition_gmem, Qi_load_partition_smem);

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

  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 128;
  constexpr int BLOCK_D = 128; // TODO. dispatch based on runtime headdim
  const int n_blocks_m = (seqlen_m + BLOCK_M - 1) / BLOCK_M;

  AT_DISPATCH_HALF_TYPES(
      Q.scalar_type(), "flash_attn_fwd", ([&] {
        // TODO. double check block dim
        flash_attn_fwd_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_D>
            <<<dim3(batch_sz, n_heads, n_blocks_m), dim3(BLOCK_M, BLOCK_N)>>>(
                Q.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                K.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                V.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>());
      }));

  return O;
}
