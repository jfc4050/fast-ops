#include <torch/extension.h>

at::Tensor flash_attn_fwd(at::Tensor Q, at::Tensor K, at::Tensor V) {
  const int head_dim = Q.size(-1);
  const float sm_scale = 1.0 / sqrt(head_dim);

  at::Tensor qk = at::matmul(Q, K.transpose(-1, -2)) * sm_scale;
  at::Tensor sm = at::softmax(qk, -1);
  at::Tensor out = at::matmul(sm, V);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attn_fwd, "FlashAttention forward");
}
