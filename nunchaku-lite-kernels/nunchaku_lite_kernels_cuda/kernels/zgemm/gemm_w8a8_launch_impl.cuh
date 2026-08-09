#include "gemm_w8a8_launch.cuh"

namespace nunchaku::kernels {

// W8A8 launch, mirroring the core (non-FP4, non-attention-fusion) subset of
// GEMM_W4A4_Launch::gemm_w4a4: main GEMM output, optional bias/wcscales,
// optional up/down LoRA, optional GELU or SiLU activation, and an optional
// fused int8 requantize of the output for chaining into a next W8A8 layer.
// QKV/rotary-pack and LiteLA-attention epilogue fusions are intentionally not
// supported here; those remain int4/nvfp4-only for now.
#ifndef __INTELLISENSE__
template<typename Config>
void GEMM_W8A8_Launch<Config>::gemm_w8a8(
#else
template<>
void GEMM_W8A8_Launch<GEMMConfig_W8A8_FP16>::gemm_w8a8(
#endif
    Tensor act,
    Tensor wgt,
    Tensor out,
    Tensor qout,
    Tensor ascales,
    Tensor wscales,
    Tensor oscales,
    Tensor lora_act_in,
    Tensor lora_up,
    Tensor lora_down,
    Tensor lora_act_out,
    Tensor bias,
    Tensor smooth_factor,
    Tensor wcscales,
    std::vector<float> lora_scales,
    bool fuse_silu) {
    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1];
    assert(K == wgt.shape[1]);

    int actualM = 0;
    int actualN = 0;
    if (out.valid()) {
        actualM = out.numel() / out.shape[-1];
        actualN = out.shape[-1];

        assert(actualM <= M && M - actualM < GEMM::BLOCK_M);
        assert(actualN <= N && N - actualN < GEMM::BLOCK_N);
    }

    auto launch = [&]<typename Epilogue>(Epilogue::Arguments args) {
        assert(M % GEMM::BLOCK_M == 0);
        assert(N % GEMM::BLOCK_N == 0);
        dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

        bool swapBlockMN = M > N * 2;
        if (swapBlockMN) {
            std::swap(grid.x, grid.y);
        }

        auto func = invoke_kernel<typename GEMM::gemm_w8a8_kernel<Epilogue>,
                                  const packed_act_t *,
                                  const packed_wgt_t *,
                                  const packed_ascale_t *,
                                  const packed_wscale_t *,
                                  int,
                                  int,
                                  int,
                                  typename Epilogue::Arguments,
                                  bool,
                                  bool>;

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, getCurrentCUDAStream()>>>(
            act.data_ptr<packed_act_t>(),
            wgt.data_ptr<packed_wgt_t>(),
            ascales.data_ptr<packed_ascale_t>(),
            wscales.data_ptr<packed_wscale_t>(),
            M,
            N,
            K,
            args,
            swapBlockMN,
            false);
        checkCUDA(cudaGetLastError());
    };

    auto launch_bias = [&]<typename NextEpilogue>(NextEpilogue::Arguments nextArgs) {
        assert(!bias.valid() || bias.numel() == N);
        assert(!wcscales.valid() || wcscales.numel() == N);

        dispatchBool(bias.valid(), [&]<bool USE_BIAS>() {
            dispatchBool(wcscales.valid(), [&]<bool USE_SCALE>() {
                using EpilogueBias = typename GEMM::template EpilogueBias<USE_BIAS, USE_SCALE>;
                using Epilogue =
                    typename GEMM::template EpilogueCombination<EpilogueBias, NextEpilogue, typename GEMM::EpilogueNop>;
                return launch.template operator()<Epilogue>(
                    {typename EpilogueBias::Arguments{
                         .bias  = USE_BIAS ? bias.data_ptr<packed_wscale_t>() : nullptr,
                         .scale = USE_SCALE ? wcscales.data_ptr<packed_wscale_t>() : nullptr,
                     },
                     nextArgs,
                     {}});
            });
        });
    };

    auto launch_lora = [&]<typename NextEpilogue, typename MidEpilogue>(NextEpilogue::Arguments nextArgs,
                                                                        MidEpilogue::Arguments midArgs) {
        assert(lora_up.valid() == lora_act_in.valid());
        assert(lora_down.valid() == lora_act_out.valid());

        const int rank_up   = lora_up.valid() ? lora_up.shape[1] : 0;
        const int rank_down = lora_down.valid() ? lora_down.shape[1] : 0;

        if (rank_up == 0) {
            assert(rank_down == 0);
            return launch_bias.template operator()<typename GEMM::template EpilogueCombination<MidEpilogue, NextEpilogue>>(
                {midArgs, nextArgs});
        }

        assert(rank_up % 16 == 0);
        assert(lora_up.shape[0] == N);
        assert(lora_act_in.shape[0] == M);
        assert(lora_act_in.shape[1] == rank_up);

        using LoraUp  = Lora;
        using scale_t = typename LoraUp::scale_t;

        scale_t scales;
        if constexpr (scales.size() > 0) {
            for (size_t i = 0; i < scales.size(); i++) {
                scales[i] = i < lora_scales.size() ? lora_scales[i] : 0.0f;
            }
        }

        if (rank_down == 0) {
            using Epilogue = typename GEMM::template EpilogueCombination<typename LoraUp::EpilogueLoraUp,
                                                                         MidEpilogue,
                                                                         NextEpilogue,
                                                                         typename GEMM::EpilogueNop>;
            return launch_bias.template operator()<Epilogue>({typename LoraUp::EpilogueLoraUp::Arguments{
                                                                  .lora_act    = lora_act_in.data_ptr<float>(),
                                                                  .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
                                                                  .rank        = rank_up,
                                                                  .scales      = scales,
                                                                  .alwaysfalse = false,
                                                              },
                                                              midArgs,
                                                              nextArgs,
                                                              {}});
        }

        assert(rank_down % 16 == 0);
        assert(lora_down.shape[0] == N);
        assert(lora_act_out.shape[0] == M);
        assert(lora_act_out.shape[1] == rank_down);

        lora_act_out.zero_();

        using LoraDown = LoraUp;
        using Epilogue = typename GEMM::template EpilogueCombination<typename LoraUp::EpilogueLoraUp,
                                                                     MidEpilogue,
                                                                     typename LoraDown::EpilogueLoraDown,
                                                                     NextEpilogue,
                                                                     typename GEMM::EpilogueNop>;
        return launch_bias.template operator()<Epilogue>({typename LoraUp::EpilogueLoraUp::Arguments{
                                                              .lora_act    = lora_act_in.data_ptr<float>(),
                                                              .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
                                                              .rank        = rank_up,
                                                              .scales      = scales,
                                                              .alwaysfalse = false,
                                                          },
                                                          midArgs,
                                                          typename LoraDown::EpilogueLoraDown::Arguments{
                                                              .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
                                                              .lora_act      = lora_act_out.data_ptr<float>(),
                                                              .rank          = rank_down,
                                                              .alwaysfalse   = false,
                                                          },
                                                          nextArgs,
                                                          {}});
    };

    if (qout.valid() && oscales.valid()) {
        // fused activation-quantize-for-next-layer path (mirrors GEMM_W4A4_Launch::gemm_w4a4):
        // GELU is applied unconditionally here, matching the only current caller (fused_gelu_mlp).
        using EpilogueQuantize = typename GEMM::EpilogueQuantize;
        auto argsQuantize      = typename EpilogueQuantize::Arguments{
                 .qout          = qout.data_ptr<packed_act_t>(),
                 .oscales       = oscales.data_ptr<packed_ascale_t>(),
                 .smooth_factor = smooth_factor.data_ptr<packed_wscale_t>(),
        };

        if (out.valid()) {
            launch_lora.template
            operator()<typename GEMM::template EpilogueCombination<typename GEMM::EpilogueDefault, EpilogueQuantize>,
                       typename Epilogues::EpilogueGelu>({typename GEMM::EpilogueDefault::Arguments{
                                                              .out     = out.data_ptr<half_t>(),
                                                              .actualM = actualM,
                                                              .actualN = actualN,
                                                          },
                                                          argsQuantize},
                                                         {});
        } else {
            launch_lora.template operator()<EpilogueQuantize, typename Epilogues::EpilogueGelu>(argsQuantize, {});
        }

    } else if (out.valid()) {
        using Epilogue = typename GEMM::EpilogueDefault;
        typename Epilogue::Arguments args{
            .out     = out.data_ptr<half_t>(),
            .actualM = actualM,
            .actualN = actualN,
        };

        if (fuse_silu) {
            launch_lora.template operator()<Epilogue, typename GEMM::EpilogueSilu>(args, {});
        } else {
            launch_lora.template operator()<Epilogue, typename GEMM::EpilogueNop>(args, {});
        }
    } else {
        assert(false);
    }
}

#ifndef __INTELLISENSE__
template<typename Config>
void GEMM_W8A8_Launch<Config>::quantize_w8a8_act_fuse_lora(
#else
template<>
void GEMM_W8A8_Launch<GEMMConfig_W8A8_FP16>::quantize_w8a8_act_fuse_lora(
#endif
    Tensor input,
    Tensor output,
    Tensor oscales,
    Tensor lora_down,
    Tensor lora_act_out,
    Tensor smooth,
    bool fuse_glu) {
    const int actualM = input.numel() / input.shape[-1];
    const int actualN = input.shape[-1];

    const int M = ceilDiv(actualM, GEMM::BLOCK_M) * GEMM::BLOCK_M;
    const int N = ceilDiv(actualN / (fuse_glu ? 2 : 1), GEMM::BLOCK_N) * GEMM::BLOCK_N;

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == N);

    assert(isTypeMatch<half_t>(oscales.dtype()));
    assert(oscales.numel() == M * N / GEMM::WARP_K);

    const int rank = lora_down.shape[1];

    assert(rank % 16 == 0);
    assert(lora_down.shape[0] == N);
    assert(lora_act_out.shape[0] == M);
    assert(lora_act_out.shape[1] == rank);

    lora_act_out.zero_();

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    dispatchBool(fuse_glu, [&]<bool FUSE_GLU>() {
        using kernel = typename GEMM::template quantize_w8a8_fuse_lora_kernel<FUSE_GLU>;

        auto func = invoke_kernel<kernel, typename kernel::Arguments>;

        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
            typename kernel::Arguments{
                .input         = input.data_ptr<half_t>(),
                .smooth_factor = smooth.valid() ? smooth.data_ptr<packed_wscale_t>() : nullptr,
                .output        = output.data_ptr<packed_act_t>(),
                .oscales       = oscales.data_ptr<packed_ascale_t>(),
                .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
                .lora_act      = lora_act_out.data_ptr<float>(),
                .lora_rank     = rank,
                .M             = M,
                .N             = N,
                .actualM       = actualM,
                .actualN       = actualN,
                .alwaysfalse   = false,
            });
        checkCUDA(cudaGetLastError());
    });
}

}; // namespace nunchaku::kernels
