#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "macros.hpp"
#include "geometry.hpp"

//std::vector<torch::Tensor> build_grad(
std::vector<torch::Tensor> build_grad(
    const torch::Tensor verts,
    const torch::Tensor edges,
    const torch::Tensor edge_tangent_vectors,
    const uint32_t n_neigh
) {
    // Sanity check(s)
    if (n_neigh > MAX_NHOOD) {
        std::cout << "ERROR: n_neigh > MAX_NHOOD, n_neigh cannot be higher than: " << MAX_NHOOD << std::endl;
        std::cout << "Use a lower n_neigh value." << std::endl;
        std::cout << "Alternatively, recompile with a higher MAX_NHOOD (slower and requires more vram)." << std::endl;
        throw std::invalid_argument("MAX_NHOOD smaller than n_neigh");
    }

    // Get the amount of vertices
    uint32_t n_vertices = (uint32_t)verts.size(0);

    // allocate & initate vert_edge_outgoing
    torch::Tensor vert_edge_outgoing = torch::zeros({n_vertices,n_neigh}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor vert_edge_outgoing_count = torch::zeros(n_vertices, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    // Assign vert_edge_outgoing
    assign_vert_edge_outgoing_cuda(edges,vert_edge_outgoing,vert_edge_outgoing_count);

    // allocate & initate build_grad_compressed variables
    torch::Tensor row_ind = torch::zeros({n_vertices*(MAX_NHOOD+1)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor cols_ind = torch::zeros({n_vertices*(MAX_NHOOD+1)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    torch::Tensor data_vals_real = torch::zeros({n_vertices*(MAX_NHOOD+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    torch::Tensor data_vals_imag = torch::zeros({n_vertices*(MAX_NHOOD+1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // Build compressed grad
    build_grad_compressed_cuda(n_vertices, MAX_NHOOD,
        edges,
        edge_tangent_vectors,
        vert_edge_outgoing,
        vert_edge_outgoing_count,
        row_ind,
        cols_ind,
        data_vals_real,
        data_vals_imag,
        1e-5,
        1.0
    );

    return {row_ind, cols_ind, data_vals_real, data_vals_imag, vert_edge_outgoing_count};

}