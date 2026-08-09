#pragma once

#include "gemm_base.cuh"
#include "lora.cuh"

namespace nunchaku::kernels {

// W8A8: int8 weights, int8 activations, symmetric quantization (signed only).
// Structurally parallels GEMM_W4A4<Config> in gemm_w4a4.cuh, but every packed
// value is a full byte instead of a nibble, and the native tensor-core MMA
// shape is m16n8k32 (INSN_K=32) instead of m16n8k64. Unlike the W4A4 kernel,
// there is no unsigned-activation or FP4 variant here -- both operands are
// always signed int8 in [-127, 127].
template<typename Config>
class GEMM_W8A8;

#ifndef __INTELLISENSE__
template<typename Config>
class GEMM_W8A8 : public GEMMBase<Config> {
#else
template<>
class GEMM_W8A8<GEMMConfig_W8A8_FP16> : public GEMMBase<GEMMConfig_W8A8_FP16> {
    using Config = GEMMConfig_W8A8_FP16;
#endif

public:
    IMPORT_GEMM_BASE(Config);

public:
    // single m16n8k32 s8xs8 MMA per N-8 half; packed_wgt_t (uint4) covers two
    // such halves (wgt.x/wgt.y and wgt.z/wgt.w), matching WARP_N_TILES=INSN_N/8=2
    __device__ __forceinline__ static packed_psum_t mma(packed_act_t act, packed_wgt_t wgt) {
        packed_psum_t psum;

        uint4 out1 =
            mma_m16n8kx_s32common<mma_helper::s8, mma_helper::s8>(act, uint2(wgt.x, wgt.y), uint4(0, 0, 0, 0));
        uint4 out2 =
            mma_m16n8kx_s32common<mma_helper::s8, mma_helper::s8>(act, uint2(wgt.z, wgt.w), uint4(0, 0, 0, 0));
        psum.data[0] = out1.x;
        psum.data[1] = out1.y;
        psum.data[2] = out1.z;
        psum.data[3] = out1.w;
        psum.data[4] = out2.x;
        psum.data[5] = out2.y;
        psum.data[6] = out2.z;
        psum.data[7] = out2.w;

        return psum;
    }

    /**
     * requantize a fp16/bf16 GEMM output tile back to int8 activations for a fused next layer.
     * mirrors GEMM_W4A4::quantize_w4a4_from_fpsum_warp, generalized to 8-bit symmetric packing.
     */
    __device__ __forceinline__ static void
    quantize_w8a8_from_fpsum_warp(const packed_fpsum_t (&fpsum)[INSN_K / INSN_N],
                                  packed_act_t &output,
                                  half_t *output_scale) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr float QVALUE_MAX       = 127.0f;
        constexpr float RECPI_QVALUE_MAX = 1 / QVALUE_MAX;

        // 0 for row 0-7; 1 for row 8-15
        half2_t input[2][INSN_K / INSN_N * 2];

#pragma unroll
        for (int i = 0; i < INSN_K / INSN_N; i++) {
            input[0][i * 2 + 0] = fpsum[i].data[0];
            input[0][i * 2 + 1] = fpsum[i].data[2];
            input[1][i * 2 + 0] = fpsum[i].data[1];
            input[1][i * 2 + 1] = fpsum[i].data[3];
        }

        half_t maxvalue[2];
        maxvalue[0] = 0;
        maxvalue[1] = 0;
#pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
            half2_t abs0 = __habs2(input[0][i]);
            half2_t abs1 = __habs2(input[1][i]);
            maxvalue[0]  = __hmax(maxvalue[0], __hmax(abs0.x, abs0.y));
            maxvalue[1]  = __hmax(maxvalue[1], __hmax(abs1.x, abs1.y));
        }
#pragma unroll
        for (int mask = 2; mask > 0; mask /= 2) {
            maxvalue[0] = __hmax(maxvalue[0], __shfl_xor_sync(~0, maxvalue[0], mask));
            maxvalue[1] = __hmax(maxvalue[1], __shfl_xor_sync(~0, maxvalue[1], mask));
        }
        maxvalue[0] = __shfl_sync(~0, maxvalue[0], laneId / 4 * 4);
        maxvalue[1] = __shfl_sync(~0, maxvalue[1], laneId / 4 * 4);

        float scale[2];
        scale[0] = float(maxvalue[0]) * RECPI_QVALUE_MAX;
        scale[1] = float(maxvalue[1]) * RECPI_QVALUE_MAX;
        if (laneId % 4 == 0) {
            output_scale[laneId / 4]     = half_t(scale[0]);
            output_scale[laneId / 4 + 8] = half_t(scale[1]);
        }

        float rscale[2];
        rscale[0] = cuda_frcp(scale[0]);
        rscale[1] = cuda_frcp(scale[1]);

        // PACK_SIZE = 4 int8 values per 32-bit register (vs 8 int4 values for W4A4)
        uint32_t qpacks[2][INSN_K / INSN_M * 2];
#pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                float2 fval  = half22float2(input[j][i]) * make_float2(rscale[j], rscale[j]);
                qpacks[j][i] = quantize_float2<8, false>(fval) << (laneId % 4 * 16);
            }
        }

#pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
#pragma unroll
            for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    qpacks[j][i] |= __shfl_xor_sync(~0, qpacks[j][i], mask);
                }
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical qpacks now

#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (laneId % 4 == i) {
                output.x = qpacks[0][0 + i];
                output.y = qpacks[1][0 + i];
                output.z = qpacks[0][4 + i];
                output.w = qpacks[1][4 + i];
            }
        }
    }

    /**
     * each warp quantizes a INSN_M * INSN_K (16 * 32) matrix.
     * mirrors GEMM_W4A4::quantize_w4a4_warp, generalized to 8-bit symmetric packing.
     * shmem must be at least INSN_M * INSN_K bytes (16 * 32 = 512 Bytes)
     * default to quantize activation; if quantize weight, input should be column-majored and output should be
     * transposed ({x, y, z, w} = {x, z, y, w})
     */
    __device__ __forceinline__ static void
    quantize_w8a8_warp(const half_t *input, int stride, packed_act_t &output, half_t *output_scale, void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int QUANTIZE_BITWIDTH = 8;
        constexpr int QVALUE_MAX        = 127; // 8 bit => [-127, 127]

        // 4 int8 values per 32-bit pack (vs 8 int4 values for W4A4)
        constexpr int PACK_SIZE             = 4;
        constexpr int NUM_PACKS_PER_ROW     = INSN_K / PACK_SIZE;
        constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE * WARP_SIZE / INSN_K;
        constexpr int NUM_PACKWARPS         = INSN_M / NUM_ROWS_PER_PACKWARP;
        using packed_input                  = std::array<half_t, PACK_SIZE>;

        packed_input packs[NUM_PACKWARPS];

        // load
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            int rowId = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            int colId = laneId % NUM_PACKS_PER_ROW * PACK_SIZE;
            packs[i]  = load(reinterpret_cast<const packed_input *>(input + rowId * stride + colId));
        }

        // find max
        half_t maxvalue[NUM_PACKWARPS];
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            maxvalue[i] = __habs(packs[i][0]);
#pragma unroll
            for (int j = 1; j < PACK_SIZE; j++) {
                maxvalue[i] = __hmax(maxvalue[i], __habs(packs[i][j]));
            }
        }

        // warp reduce (max)
#pragma unroll
        for (int mask = NUM_PACKS_PER_ROW / 2; mask > 0; mask /= 2) {
#pragma unroll
            for (int i = 0; i < NUM_PACKWARPS; i++) {
                maxvalue[i] = __hmax(maxvalue[i], __shfl_xor_sync(~0, maxvalue[i], mask));
            }
        }

        // broadcast (max)
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            maxvalue[i] = __shfl_sync(~0, maxvalue[i], laneId / NUM_PACKS_PER_ROW * NUM_PACKS_PER_ROW);
        }

        // quantize
        using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
        matrix_t &mat  = *reinterpret_cast<matrix_t *>(shmem);
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            half_t scale  = maxvalue[i] / half_t(QVALUE_MAX);
            half_t rscale = half_t(QVALUE_MAX) / maxvalue[i];
            if (laneId % NUM_PACKS_PER_ROW == 0) {
                output_scale[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW] = scale;
            }

            uint32_t qpack = 0;
#pragma unroll
            for (int j = 0; j < PACK_SIZE; j += 2) {
                half2_t hval = __hmul2(half2_t(rscale, rscale), half2_t(packs[i][j], packs[i][j + 1]));
                qpack |= quantize_float2<QUANTIZE_BITWIDTH, false>(half22float2(hval)) << (j * QUANTIZE_BITWIDTH);
            }
            mat[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW][laneId % NUM_PACKS_PER_ROW] = qpack;
        }
        __syncwarp();

        // convert to imma format
        int row = laneId % 16;
        int col = laneId / 16 * 4;
        ldmatrix(&mat[row][col], output);

        __syncwarp();
    }

    __device__ __forceinline__ static void
    compute(act_warp A, wgt_warp W, ascale_warp ascale, wscale_warp wscale, fpsum_warp &fpsum) {
        Base::template apply_scales<typename Base::i2f_normal>(
            [&](int i, int j) { return mma(A[i], W[j]); }, ascale, wscale, fpsum);
    }

    // out: [M / BLOCK_M, N / BLOCK_N, NUM_WARPS, 1, NUM_M_TILES, NUM_N_TILES, WARP_SIZE] of fpsum_warp
    template<typename Epilogue>
    __device__ __forceinline__ static void gemm_w8a8_block(const BlockInfo binfo,
                                                            const packed_act_t *act,
                                                            const packed_wgt_t *wgt,
                                                            const packed_ascale_t *ascales,
                                                            const packed_wscale_t *wscales,
                                                            int M,
                                                            int N,
                                                            int K,
                                                            const Epilogue::Arguments &epilogueArgs,
                                                            bool alwaysfalse) {
        constexpr int NUM_STAGES = 2;

        act_warp A[NUM_STAGES];
        wgt_warp W[NUM_STAGES];
        ascale_warp ascale[NUM_STAGES];
        wscale_warp wscale[NUM_STAGES];
        fpsum_warp fpsum;

        for (int k = 0; k < NUM_STAGES - 1; k++) {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
            load_ascale(ascales, k, M, ascale[k], true);
            load_wscale(wscales, k, N, wscale[k], true);
        }

        for (auto &pack : fpsum) {
            for (int i = 0; i < 4; i++) {
                pack.data[i].x = 0;
                pack.data[i].y = 0;
            }
        }

        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES) {
#pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx   = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                load_ascale(ascales, nextk, M, ascale[idx], pred);
                load_wscale(wscales, nextk, N, wscale[idx], pred);

                compute(A[k2], W[k2], ascale[k2], wscale[k2], fpsum);

                if (alwaysfalse) {
                    dummy = clock();
                }
            }
        }

        unused_var(dummy, alwaysfalse);

        Epilogue()(binfo, fpsum, M, N, K, epilogueArgs);
    }

    template<typename Epilogue>
    struct gemm_w8a8_kernel {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;

        __device__ void operator()(const packed_act_t *act,
                                   const packed_wgt_t *wgt,
                                   const packed_ascale_t *ascales,
                                   const packed_wscale_t *wscales,
                                   int M,
                                   int N,
                                   int K,
                                   Epilogue::Arguments epilogueArgs,
                                   bool swapBlockXY,
                                   bool alwaysfalse) {
            BlockInfo binfo = {
                .bm         = (int)blockIdx.x,
                .bn         = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            if (swapBlockXY) {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            const int bm = binfo.bm;
            const int bn = binfo.bn;

            gemm_w8a8_block<Epilogue>(binfo,
                                      act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                                      wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                                      ascales + bm * (K / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
                                      wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
                                      M,
                                      N,
                                      K,
                                      epilogueArgs,
                                      alwaysfalse);
        }
    };

    // requantizes the GEMM output into int8 activations for a fused next layer (chained W8A8 layers)
    struct EpilogueQuantize {
        struct Arguments {
            packed_act_t *qout;
            packed_ascale_t *oscales;
            const packed_wscale_t *smooth_factor;
        };

        static constexpr int NUM_PACKS  = INSN_K / INSN_N;
        static constexpr int NUM_GROUPS = WARP_N_TILES / NUM_PACKS;

        __device__ __forceinline__ void
        apply_quantize(fpsum_warp fpsum, int M, int N, int K, packed_act_t *qout, packed_ascale_t *oscales,
                       const packed_wscale_t *smooth_factor) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            __shared__ half_t oscale_shmem[NUM_WARPS][WARP_M];

            wscale_warp smooth;
            load_wscale(smooth_factor, 0, N, smooth, true);

#pragma unroll
            for (int group = 0; group < NUM_GROUPS; group++) {
#pragma unroll
                for (int i = 0; i < WARP_M_TILES; i++) {
                    packed_fpsum_t tmp[NUM_PACKS];

#pragma unroll
                    for (int j = 0; j < NUM_PACKS; j++) {
                        half2_t ws1 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4, laneId);
                        half2_t ws2 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4 + 2, laneId);
#pragma unroll
                        for (int k = 0; k < 4; k++) {
                            tmp[j].data[k] = fpsum[i * WARP_N_TILES + group * NUM_PACKS + j].data[k];
                        }

                        tmp[j].data[0] = h2div(tmp[j].data[0], ws1);
                        tmp[j].data[1] = h2div(tmp[j].data[1], ws1);
                        tmp[j].data[2] = h2div(tmp[j].data[2], ws2);
                        tmp[j].data[3] = h2div(tmp[j].data[3], ws2);
                    }

                    packed_act_t qresult;
                    quantize_w8a8_from_fpsum_warp(tmp, qresult, &oscale_shmem[warpId][i * INSN_M]);
                    store(&qout[((group * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SIZE + laneId], qresult);
                }

                __syncwarp();
                pack_ascales(&oscale_shmem[warpId][0],
                             &oscales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
                __syncwarp();
            }
        }

        __device__ __forceinline__ void
        operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, const Arguments &args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            apply_quantize(fpsum,
                           M,
                           N,
                           K,
                           args.qout + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                           args.oscales +
                               (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
                           args.smooth_factor + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES);
        }
    };

    template<bool fuse_glu>
    struct quantize_w8a8_fuse_lora_kernel {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
        static constexpr size_t SHMEM_PER_WARP =
            ceilDiv<size_t>(Base::template load_act_to_fpsum<fuse_glu>::SHMEM_SIZE, 128) * 128;
        static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * NUM_WARPS;

        struct Arguments {
            const half_t *input;
            const packed_wscale_t *smooth_factor;
            packed_act_t *output;
            packed_ascale_t *oscales;
            const packed_fpsum_t *lora_wgt_down;
            float *lora_act;

            int lora_rank;

            int M, N;
            int actualM, actualN;

            bool alwaysfalse;
        };

        __device__ __forceinline__ void operator()(Arguments args) {
            const BlockInfo binfo = {
                .bm         = (int)blockIdx.x,
                .bn         = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            const int bm     = binfo.bm;
            const int bn     = binfo.bn;
            const int warpId = threadIdx.x / WARP_SIZE;

            const int m_offset = bm * BLOCK_M + warpId * WARP_M;
            const int n_offset = bn * BLOCK_N * (fuse_glu ? 2 : 1);

            extern __shared__ uint8_t shmem[];

            fpsum_warp fpsum;

            Base::template load_act_to_fpsum<fuse_glu>()(args.input + m_offset * args.actualN + n_offset,
                                                         args.actualN,
                                                         args.actualM - m_offset,
                                                         args.actualN - n_offset,
                                                         fpsum,
                                                         shmem + warpId * SHMEM_PER_WARP);

            using EpilogueLoraDown = typename Lora<Config>::EpilogueLoraDown;

            EpilogueLoraDown()(binfo,
                               fpsum,
                               args.M,
                               args.N,
                               0,
                               typename EpilogueLoraDown::Arguments{
                                   .lora_wgt_down = args.lora_wgt_down,
                                   .lora_act      = args.lora_act,
                                   .rank          = args.lora_rank,
                                   .alwaysfalse   = args.alwaysfalse,
                               });

            EpilogueQuantize()(binfo,
                               fpsum,
                               args.M,
                               args.N,
                               0,
                               typename EpilogueQuantize::Arguments{
                                   .qout          = args.output,
                                   .oscales       = args.oscales,
                                   .smooth_factor = args.smooth_factor,
                               });
        }
    };
};

}; // namespace nunchaku::kernels
