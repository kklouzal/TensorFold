#include <torch/extension.h>
#include <vector>
void ds41_pack_planes(torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor,
                      torch::Tensor);
void ds41_restore_planes(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t,
                         int64_t, torch::Tensor);
void ds41_dense_rows(torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                     std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<int64_t>,
                     std::vector<int64_t>, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor,
                     int64_t, int64_t);
void ds41_q5k_rows(torch::Tensor, torch::Tensor, int64_t, torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_planes", &ds41_pack_planes);
  m.def("restore_planes", &ds41_restore_planes);
  m.def("dense_rows", &ds41_dense_rows);
  m.def("q5k_rows", &ds41_q5k_rows);
}
