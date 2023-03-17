#include <ATen/core/TensorAccessor.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/ScalarType.h>
#include <torch/extension.h>

#include "common/launch_utils.h"
#include "cute/config.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor.hpp"

template <typename scalar_t, int BLOCK_M, int BLOCK_N, int BLOCK_D>
struct flash_attn_fwd_smem {
  scalar_t Qi[BLOCK_M * BLOCK_D];
  scalar_t Sij[BLOCK_M * BLOCK_N];

  union {
    scalar_t Kj[BLOCK_N * BLOCK_D];
    scalar_t Vj[BLOCK_N * BLOCK_D];
  };
};

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
  // note the parallelization strategy and loop structure is different than
  // described in the flash attention paper.
  // the outer loop is over q sequence (parallelized over threadblocks/SMs)
  // and inner loop is over kv sequence, where the algorithm
  // described in the paper does the opposite.
  //
  // this means that for each threadblock:
  // - Qi is only loaded from DRAM once
  // - Oi, li and mi are only written to DRAM once
  // - Kj and Vj have to be loaded from DRAM multiple times

  // map PyTorch type to CUTLASS type
  using scalar_t = typename cutlass_t<scalar_t_pt>::value;

  __shared__ flash_attn_fwd_smem<scalar_t, BLOCK_M, BLOCK_N, BLOCK_D> smem;

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
  auto Q = cute::make_tensor(
      cute::make_gmem_ptr(
          reinterpret_cast<scalar_t *>(Q_accessor[batch_idx][head_idx].data())),
      cute::make_layout(
          cute::make_shape(Q_accessor.size(3), Q_accessor.size(4)),
          cute::GenRowMajor{}));
  auto K = cute::make_tensor(
      cute::make_gmem_ptr(
          reinterpret_cast<scalar_t *>(K_accessor[batch_idx][head_idx].data())),
      cute::make_layout(
          cute::make_shape(K_accessor.size(3), K_accessor.size(4)),
          cute::GenRowMajor{}));
  // TODO. do V as well

  // load Qi into SRAM - this is loop invariant and is only loaded once.
  auto Qi = cute::make_tensor(
      cute::make_smem_ptr(smem.Qi),
      cute::make_layout(cute::make_shape(BLOCK_M, BLOCK_D)));
  auto Qi_gmem_tile = cute::local_tile(
      Q, cute::make_shape(BLOCK_M, BLOCK_D), cute::make_coord(start_m, 0));
  auto Qi_load_thread_layout =
      cute::make_shape(cute::Int<32>{}, cute::Int<8>{});
  auto Qi_load_partition_gmem =
      cute::local_partition(Qi_gmem_tile, Qi_load_thread_layout, thread_id);
  auto Qi_load_partition_smem =
      cute::local_partition(Qi, Qi_load_thread_layout, thread_id);
  cute::copy(Qi_load_partition_gmem, Qi_load_partition_smem);

  for (int seq_block_n0 = 0; seq_block_n0 < seqlen_n; seq_block_n0 += BLOCK_N) {

    // load Kj into SRAM
    auto Kj = cute::make_tensor(
        cute::make_smem_ptr(smem.Kj),
        cute::make_layout(cute::make_shape(BLOCK_N, BLOCK_D)));
    auto Kj_gmem_tile = cute::local_tile(
        Kj, cute::make_shape(BLOCK_N, BLOCK_D),
        cute::make_coord(seq_block_n0, 0));
    auto Kj_load_thread_layout =
        cute::make_shape(cute::Int<32>{}, cute::Int<8>{});
    auto Kj_load_partition_gmem =
        cute::local_partition(Kj_gmem_tile, Kj_load_thread_layout, thread_id);
    auto Kj_load_partition_smem =
        cute::local_partition(Kj, Kj_load_thread_layout, thread_id);
    cute::copy(Kj_load_partition_gmem, Kj_load_partition_smem);

    // initialize accumulator tile for Sij (registers)
    auto Sij = cute::make_tensor(
        cute::make_smem_ptr(smem.Sij),
        cute::make_layout(cute::make_shape(BLOCK_M, BLOCK_N)));
    auto Sij_frag = cute::make_fragment_like(
        cute::make_layout(cute::make_shape(cute::Int<16>{}, cute::Int<16>{})));

    // do Sij = tau * Qi @ Kj.T
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
            <<<dim3(batch_sz, n_heads, n_blocks_m), dim3(8, 32)>>>(
                Q.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                K.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                V.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                O.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>());
      }));

  return O;
}
