#include <vector>

#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include "geometry.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"build_grad",
		py::overload_cast<
		    const torch::Tensor,
		    const torch::Tensor,
            const torch::Tensor,
		    const uint32_t
		>(&build_grad),
		"CUDA implementation of build_grad.",
		py::arg("verts"),
		py::arg("edges"),
		py::arg("edge_tangent_vectors"),
		py::arg("n_neigh")
	);
// Overload disabled for now. Might be useful in full cuda workflow later
/*	m.def(
		"build_grad",
		py::overload_cast<
		    const std::vector<torch::Tensor>,
		    const std::vector<torch::Tensor>,
            const std::vector<torch::Tensor>,
		    const uint32_t
		>(&build_grad),
		"CUDA implementation of build_grad.",
		py::arg("verts"),
		py::arg("edges"),
		py::arg("edge_tangent_vectors"),
		py::arg("n_neigh")
	);*/
}