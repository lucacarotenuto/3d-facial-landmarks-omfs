#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>

#include <torch/extension.h>

std::vector<torch::Tensor> build_grad(
    const torch::Tensor verts,
    const torch::Tensor edges,
    const torch::Tensor edge_tangent_vectors,
    const uint32_t n_neigh
);

void assign_vert_edge_outgoing_cuda(
    torch::Tensor edges,
    torch::Tensor vert_edge_outgoing,
    torch::Tensor vert_edge_outgoing_count
);

void build_grad_compressed_cuda(
        const int num_vertices,
        const int max_nhood,
        torch::Tensor edges,
        torch::Tensor edge_tangent_vectors,
        torch::Tensor vert_edge_outgoing,
        torch::Tensor vert_edge_outgoing_count,
        torch::Tensor row_inds,
        torch::Tensor col_inds,
        torch::Tensor data_vals_real,
        torch::Tensor data_vals_imag,
        const float eps_reg,
        const float w_e
);

#endif
