#include <ATen/ATen.h>

#include <vector>

// std::vector<at::Tensor> mean_var_cuda(at::Tensor x);

static void get_dims(at::Tensor x, int64_t& num, int64_t& chn, int64_t& sp) {
  num = x.size(0);
  chn = x.size(1);
  sp = 1;
  for (int64_t i = 2; i < x.ndimension(); ++i)
    sp *= x.size(i);
}

// static void get_dims(at::Tensor x, int64_t& batch, int64_t& channel, int64_t& h, int64_t& w) {
//   batch = x.size(0);
//   channel = x.size(1);
//   h = x.size(2);
//   w = x.size(3);
// }
