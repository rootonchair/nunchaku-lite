import torch

from kernels.benchmark import Benchmark


class NunchakuLiteKernelsBenchmark(Benchmark):
    def setup(self):
        self.m = 16
        self.k = 1536
        self.n = 3072
        self.input = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.kernel_weights = torch.zeros(self.n // 4, (self.k // 64) * 32, device=self.device, dtype=torch.int32)
        self.scales = torch.ones(self.k // 64, self.n, device=self.device, dtype=torch.float16) * 0.01
        self.zeros = torch.zeros_like(self.scales)

    def benchmark_base(self):
        self.kernel.awq_gemm_w4a16_g64_int32(self.input, self.kernel_weights, self.scales, self.zeros)

    def verify_base(self) -> torch.Tensor:
        return torch.zeros(self.m, self.n, device=self.device, dtype=torch.float16)
