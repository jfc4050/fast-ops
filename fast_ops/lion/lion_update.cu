#include <ATen/Dispatch.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include <torch/extension.h>

#define PRAGMA_UNROLL _Pragma("unroll")

template <typename scalar_t> __device__ scalar_t sign(scalar_t x) {
  int t = x < 0 ? -1 : 0;
  return x > 0 ? 1 : t;
}

template <typename scalar_t, typename momentum_t, typename IdxT>
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
  static_assert(sizeof(scalar_t) == sizeof(momentum_t), "restriction for now");
  constexpr int ACCESS_N = 16 / sizeof(scalar_t);
  using VectorT = at::native::memory::aligned_vector<scalar_t, ACCESS_N>;
  using MomentumVectorT =
      at::native::memory::aligned_vector<momentum_t, ACCESS_N>;

  VectorT *param_vectors = reinterpret_cast<VectorT *>(param);
  const VectorT *grad_vectors = reinterpret_cast<const VectorT *>(grad);
  MomentumVectorT *momentum_vectors =
      reinterpret_cast<MomentumVectorT *>(exp_avg);

  const scalar_t weight_decay_factor = 1.0 - lr * weight_decay;
  const scalar_t beta1_complement = 1.0 - beta1;
  const scalar_t beta2_complement = 1.0 - beta2;
  const scalar_t neg_lr = -lr;

  // grid-stride loop, with additional striding due to threads using
  // vectorized accesses
  for (IdxT i = (blockIdx.x * blockDim.x + threadIdx.x) * ACCESS_N; i < numel;
       i += blockDim.x * gridDim.x * ACCESS_N) {

    // load vectors into registers
    VectorT param_vector = param_vectors[i];
    const VectorT grad_vector = grad_vectors[i];
    MomentumVectorT momentum_vector = momentum_vectors[i];

    // apply weight decay
    PRAGMA_UNROLL
    for (int ii = 0; ii < ACCESS_N; ++i) {
      param_vector.val[ii] *= weight_decay_factor;
    }

    // compute update
    VectorT update_vector = momentum_vector;
    PRAGMA_UNROLL
    for (int ii = 0; ii < ACCESS_N; ++ii) {
      update_vector.val[ii] *= beta1;
    }
    PRAGMA_UNROLL
    for (int ii = 0; ii < ACCESS_N; ++ii) {
      // TODO. make sure compiler uses FMA here
      update_vector.val[ii] =
          beta1_complement * grad_vector.val[ii] + update_vector.val[ii];
    }
    PRAGMA_UNROLL
    for (int i = 0; i < ACCESS_N; ++i) {
      update_vector.val[i] = sign(update_vector.val[i]);
    }

    // apply update
    PRAGMA_UNROLL
    for (int ii = 0; ii < ACCESS_N; ++ii) {
      // TODO. make sure compiler uses FMA here
      param_vector.val[ii] =
          update_vector.val[ii] * neg_lr + param_vector.val[ii];
    }

    // write back
    param_vectors[i] = param_vector;

    // decay momentum
    for (int ii = 0; ii < ACCESS_N; ++ii) {
      momentum_vector.val[ii] *= beta2;
    }
    for (int ii = 0; ii < ACCESS_N; ++ii) {
      // TODO. make sure compiler uses FMA here
      momentum_vector.val[ii] =
          grad_vector.val[ii] * beta2_complement + momentum_vector.val[ii];
    }

    // write back momentum
    momentum_vectors[i] = momentum_vector;
  }
}

#define CHECK_CONTIGUOUS(tensor)                                               \
  if (!tensor.is_non_overlapping_and_dense()) {                                \
    throw std::runtime_error("expected tensor to be contiguous");              \
  }

#define AT_DISPATCH_CASE_HALF_TYPES(...)                                       \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_HALF_TYPES(TYPE, NAME, ...)                                \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))

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
  // TODO. assert sizes

  CHECK_CONTIGUOUS(param);
  CHECK_CONTIGUOUS(grad);
  CHECK_CONTIGUOUS(exp_avg);

  AT_DISPATCH_HALF_TYPES(param.scalar_type(), "lion_update", [&]() {
    const int numel = param.numel();
    constexpr int BLOCK_DIM = 256;
    const int N_BLOCKS = (numel + BLOCK_DIM - 1) / BLOCK_DIM;
    const dim3 blockdim(BLOCK_DIM);
    const dim3 griddim(N_BLOCKS);

    assert(BLOCK_DIM * N_BLOCKS == numel);

    // TODO. check if can use 32bit indexing
    // TODO. handle different momentum type
    lion_update_kernel<scalar_t, scalar_t, uint64_t>
        <<<griddim, blockdim, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<scalar_t *>(param.data_ptr()),
            reinterpret_cast<const scalar_t *>(grad.data_ptr()),
            reinterpret_cast<scalar_t *>(exp_avg.data_ptr()),
            numel,
            lr,
            beta1,
            beta2,
            weight_decay);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
