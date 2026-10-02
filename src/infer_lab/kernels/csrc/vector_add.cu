// Stage 1 — memory-bound kernel. Goal: approach T4 peak DRAM bandwidth (~320 GB/s).
//   scalar : grid-stride loop, 1 float per thread per iteration
//   vec4   : float4 loads/stores -> 4x fewer memory instructions
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void vector_add_scalar(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_vec4(const float4* __restrict__ a, const float4* __restrict__ b,
                                float4* __restrict__ c, int64_t n4) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n4;
         i += (int64_t)blockDim.x * gridDim.x) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

static bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b, bool vectorized) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    a = a.contiguous();
    b = b.contiguous();
    auto c = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) return c;

    const int threads = 256;
    const int max_blocks = 40 * 32;  // T4: 40 SMs; enough resident blocks to saturate DRAM
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (vectorized && n % 4 == 0 && aligned16(a.data_ptr()) && aligned16(b.data_ptr())) {
        const int64_t n4 = n / 4;
        const int blocks = (int)std::min<int64_t>((n4 + threads - 1) / threads, max_blocks);
        vector_add_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(a.data_ptr<float>()),
            reinterpret_cast<const float4*>(b.data_ptr<float>()),
            reinterpret_cast<float4*>(c.data_ptr<float>()), n4);
    } else {
        const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, max_blocks);
        vector_add_scalar<<<blocks, threads, 0, stream>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "c = a + b (float32)",
          py::arg("a"), py::arg("b"), py::arg("vectorized") = true);
}
