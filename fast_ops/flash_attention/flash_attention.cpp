#include <torch/extension.h>

at::Tensor flash_attn_fwd(at::Tensor Q, at::Tensor K, at::Tensor V) {
  return Q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attn_fwd, "FlashAttention forward");
}
