#include <torch/extension.h>

at::Tensor flash_attn_fwd_cuda(at::Tensor Q, at::Tensor K, at::Tensor V);

at::Tensor flash_attn_fwd(at::Tensor Q, at::Tensor K, at::Tensor V) {
  const int head_dim = Q.size(-1);
  const float sm_scale = 1.0 / sqrt(head_dim);

  return flash_attn_fwd_cuda(Q, K, V);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attn_fwd, "FlashAttention forward");
}
