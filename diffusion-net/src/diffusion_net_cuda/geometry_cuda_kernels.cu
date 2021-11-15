#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.hpp"
#include "geometry_cuda.hcu"


__global__ void kernel::assign_vert_edge_outgoing_cuda_kernel(
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edges,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> vert_edge_outgoing,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> vert_edge_outgoing_count
) {
    // Get edge ID
    int eId = blockDim.x * blockIdx.x + threadIdx.x;

    if (eId < edges.size(1)) { // Sanity check
        int tail_ind = edges[0][eId];
        int tip_ind = edges[1][eId];

        if (tip_ind != tail_ind) {
            // Ensure single location access per kernel
            int location = atomicAdd(&vert_edge_outgoing_count[tail_ind],1);
            // Assign edge id to vertex
            vert_edge_outgoing[tail_ind][location] = eId;
        }
    }
}

__global__ void kernel::build_grad_compressed_cuda_kernel(
    const int num_vertices,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> edges,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_tangent_vectors,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> vert_edge_outgoing,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> vert_edge_outgoing_count,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_inds,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> col_inds,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> data_vals_real,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> data_vals_imag,
    const float eps_reg,
    const float w_e
) {
    // Get vertex ID
    int iV = blockDim.x * blockIdx.x + threadIdx.x;

    if (iV < num_vertices) { // Sanity check
        // Get current n_neigh
        const int n_neigh = vert_edge_outgoing_count[iV];

        // Determine offset in the global arrays
        int offset = iV*(MAX_NHOOD+1);

        // Allocate arrays for matrix computations
        float lhs_mat[MAX_NHOOD][2];
        float rhs_mat[MAX_NHOOD][MAX_NHOOD+1];
        float lhs_inv[2][MAX_NHOOD];
        float sol_mat[2][MAX_NHOOD+1];

        // Init first row_inds & col_inds
        row_inds[offset] = iV;
        col_inds[offset] = iV;

        sol_mat[0][0] = 0;
        sol_mat[1][0] = 0;

        // iterate over i_neigh
        for (int i_neigh = 0; i_neigh < n_neigh; i_neigh++) {
            // offset for the global arrays
            int cur_index = offset+i_neigh+1;

            // Get edge id for the vertex at i_neigh
            int iE = vert_edge_outgoing[iV][i_neigh];
            // Get corresponding tip edge
            int jV = edges[1][iE];

            // use the for loop for (pre)assigning
            row_inds[cur_index] = iV;
            col_inds[cur_index] = jV;
            lhs_inv[0][i_neigh] = 0;
            lhs_inv[1][i_neigh] = 0;
            sol_mat[0][i_neigh+1] = 0;
            sol_mat[1][i_neigh+1] = 0;

            // w_e multiplication to lhs_mat
            lhs_mat[i_neigh][0] = edge_tangent_vectors[iE][0] * w_e;
            lhs_mat[i_neigh][1] = edge_tangent_vectors[iE][1] * w_e;

            // rhs_mat population
            rhs_mat[i_neigh][0] = w_e * (-1);
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1;
        }

        // Lazy init
        float lhs_mul[2][2];
        lhs_mul[0][0] = 0;
        lhs_mul[0][1] = 0;
        lhs_mul[1][0] = 0;
        lhs_mul[1][1] = 0;

        // simple matrix multiplication 1 (lhs_mat.T @ lhs_mat)
        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                for (int k=0;k<n_neigh;k++) {
                    lhs_mul[i][j] += lhs_mat[k][i] * lhs_mat[k][j];
                }
            }
        }

        // add eps_reg
        lhs_mul[0][0] += eps_reg;
        lhs_mul[1][1] += eps_reg;

        // simple inversion
        float lhs_mul_inv[2][2];
        float det = 1.0/(lhs_mul[0][0]*lhs_mul[1][1]-lhs_mul[0][1]*lhs_mul[1][0]);

        lhs_mul_inv[0][0] = det * lhs_mul[1][1];
        lhs_mul_inv[0][1] = det * -lhs_mul[0][1];
        lhs_mul_inv[1][0] = det * -lhs_mul[1][0];
        lhs_mul_inv[1][1] = det * lhs_mul[1][1];

        // simple matrix multiplication 2 (lhs_mul_inv @ lhs_mat.T)
        for (int i=0;i<2;i++) {
            for (int j=0;j<n_neigh;j++) {
                for (int k=0;k<2;k++) {
                    //lhs_inv[i][j] += lhs_mul[k][i] * lhs_mat[j][k];
                    lhs_inv[i][j] += lhs_mul_inv[k][i] * lhs_mat[j][k];
                }
            }
        }

        // simple matrix multiplication 3 (sol_mat = lhs_inv @ rhs_mat)
        for (int i=0;i<(n_neigh);i++) {
            for (int j=0;j<(n_neigh+1);j++) {
                for (int k=0;k<2;k++) {
                    sol_mat[k][j] += lhs_inv[k][i] * rhs_mat[i][j];
                }
            }
        }

        // Assign sol_mat
        for (int i_neigh=0; i_neigh<(n_neigh+1); i_neigh++) {
            data_vals_real[offset+i_neigh] = sol_mat[0][i_neigh];
            data_vals_imag[offset+i_neigh] = sol_mat[1][i_neigh];
        }
    }
}