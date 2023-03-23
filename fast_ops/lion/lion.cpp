#include <torch/extension.h>

// forward declaration, defined in .cu file
void lion_update(
    at::Tensor param,
    const at::Tensor grad,
    at::Tensor exp_avg,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lion_update", &lion_update, "Lion update");
}
