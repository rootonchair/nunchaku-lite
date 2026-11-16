// Local (setup.py) MPS build: registers the dispatcher ops with MPS implementations.
#define NUNCHAKU_LITE_REGISTER_MPS_DISPATCH

#include <torch/extension.h>

#include "../nunchaku_lite_kernels_cuda/torch_registration.h"

// Ops are exposed through torch.ops.nunchaku_lite_kernels; the module only needs to be importable.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
