#include <torch/extension.h>
#include <vector>
#include "layernorm_kernel_util.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// std::vector<at::Tensor> mean_var(at::Tensor x) {
//   CHECK_INPUT(x);
//   return mean_var_cuda(x);
// }

std::vector<at::Tensor> layer_norm(
  at::Tensor input, at::Tensor gamma, at::Tensor beta, double epsilon) {

  auto size = input.sizes();

  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);

  // std::vector<at::Tensor> two_pass_mean_var = mean_var(input);
  // at::Tensor mean = two_pass_mean_var[0];
  // at::Tensor var = two_pass_mean_var[1];
  // at::Tensor var_sqrt = at::sqrt(at::add(var, epsilon));
  // at::Tensor y = at::div(at::sub(input, mean), var_sqrt);

  at::Tensor mean = at::mean(input, 1, true);
  at::Tensor input_sub_mean = at::sub(input, mean);
  at::Tensor var = at::mean(at::mul(input_sub_mean, input_sub_mean), 1, true);

  at::Tensor var_sqrt = at::sqrt(at::add(var, epsilon));
  at::Tensor y = at::div(input_sub_mean, var_sqrt);

  at::Tensor output = at::add(at::mul(gamma, y), beta);

  return {output, y, var_sqrt};
}

std::vector<at::Tensor> layer_norm_gradient(
  at::Tensor grad_output, at::Tensor y, at::Tensor var_sqrt, at::Tensor gamma) {

  CHECK_INPUT(grad_output);
  CHECK_INPUT(y);
  CHECK_INPUT(var_sqrt);

  at::Tensor g = at::mul(grad_output, gamma);
  at::Tensor mean_g = at::mean(g, 1, true);
  at::Tensor g_gamma = at::mul(g, y);
  at::Tensor mean_gy = at::mean(g_gamma, 1, true);
  at::Tensor gx = at::div(at::sub(at::sub(g, at::mul(y, mean_gy)), mean_g), var_sqrt);

  return {gx, g_gamma, grad_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layer_norm, "LayerNorm forward (CUDA)");
  m.def("backward", &layer_norm_gradient, "LayerNorm backward (CUDA)");
}
