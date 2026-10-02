// Stage 2 — compute-bound kernel. C[M,N] = A[M,K] @ B[K,N], float32, row-major.
//   naive : every thread reads a full row of A and column of B from DRAM
//   tiled : TILE x TILE blocks staged in shared memory -> each global load reused TILE times
// T4 fp32 peak is ~8.1 TFLOPS; cuBLAS (torch.matmul) is the reference ceiling.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> col keeps B reads coalesced
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

template <int TILE>
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, const std::string& variant) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "shape mismatch");
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = c10::cuda::getCurrentCUDAStream();

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    if (variant == "naive") {
        matmul_naive<<<grid, block, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                 C.data_ptr<float>(), M, N, K);
    } else if (variant == "tiled") {
        matmul_tiled<TILE><<<grid, block, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                       C.data_ptr<float>(), M, N, K);
    } else {
        TORCH_CHECK(false, "unknown variant: ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "C = A @ B (float32)",
          py::arg("A"), py::arg("B"), py::arg("variant") = "tiled");
}
