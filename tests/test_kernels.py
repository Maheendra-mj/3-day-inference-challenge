import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("n", [1, 3, 1024, 1 << 20, (1 << 20) + 3])
@pytest.mark.parametrize("vectorized", [False, True])
def test_vector_add(n, vectorized):
    from infer_lab.kernels import load_ext
    ext = load_ext("vector_add")
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    torch.testing.assert_close(ext.vector_add(a, b, vectorized), a + b)


@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 33, 9), (128, 256, 64), (500, 300, 700)])
@pytest.mark.parametrize("variant", ["naive", "tiled"])
def test_matmul(shape, variant):
    from infer_lab.kernels import load_ext
    ext = load_ext("matmul")
    M, K, N = shape
    A, B = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda")
    torch.testing.assert_close(ext.matmul(A, B, variant), A @ B, rtol=1e-3, atol=1e-3)
