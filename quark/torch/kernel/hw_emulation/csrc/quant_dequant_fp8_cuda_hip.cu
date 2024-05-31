//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include "custom_type.h"


__host__ __device__ uint8_t double_to_fp8(
                        const double x,
                        const saturation_t saturate,
                        const fp8_interpretation_t fp8_interpretation) {
    unsigned char res;
    unsigned long long int xbits;

#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&xbits, &x, sizeof(x));
#else
    (void)std::memcpy(&xbits, &x, sizeof(x));
#endif
    unsigned char FP8_MAXNORM;
    unsigned char FP8_MANTISSA_MASK;
    unsigned short int FP8_EXP_BIAS;
    unsigned long long int FP8_SIGNIFICAND_BITS;
    const unsigned long long int DP_INF_BITS = 0x7FF0000000000000ULL;
    unsigned long long int FP8_MINDENORM_O2;
    unsigned long long int FP8_OVERFLOW_THRESHOLD;
    unsigned long long int FP8_MINNORM;

    if (fp8_interpretation == E4M3) {
        FP8_EXP_BIAS = 7U;
        FP8_SIGNIFICAND_BITS = 4ULL;
        FP8_MANTISSA_MASK = 0x7U;
        FP8_MINDENORM_O2 = 0x3F50000000000000ULL; // mindenorm/2 = 2^-10
        FP8_OVERFLOW_THRESHOLD =
            0x407D000000000000ULL; // maxnorm + 1/2ulp = 0x1.Cp+8 + 0x1p+4
        FP8_MAXNORM = 0x7EU;
        FP8_MINNORM = 0x3F90000000000000ULL; // minnorm = 2^-6
    } else {                                 //E5M2
        FP8_EXP_BIAS = 15U;
        FP8_SIGNIFICAND_BITS = 3ULL;
        FP8_MANTISSA_MASK = 0x3U;
        FP8_MINDENORM_O2 = 0x3EE0000000000000ULL; // mindenorm/2 = 2^-17
        FP8_OVERFLOW_THRESHOLD =
            0x40EE000000000000ULL -
            1ULL; // maxnorm + 1/2ulp = 0x1.Ep+15, and -1 to have common code
        FP8_MAXNORM = 0x7BU;
        FP8_MINNORM = 0x3F10000000000000ULL; // minnorm = 2^-14
    }

    // 1/2 LSB of the target format, positioned in double precision mantissa
    // helpful in midpoints detection during round-to-nearest-even step
    const unsigned long long int FP8_DP_HALF_ULP =
        (unsigned long long int)1ULL << (53ULL - FP8_SIGNIFICAND_BITS - 1ULL);
    // prepare sign bit in target format
    unsigned char sign = (unsigned char)((xbits >> 63ULL) << 7U);
    // prepare exponent field in target format
    unsigned char exp =
        (unsigned char)((((unsigned short int)(xbits >> 52ULL)) & 0x7FFU) -
                        1023U + FP8_EXP_BIAS);
    // round mantissa to target format width, rounding towards zero
    unsigned char mantissa =
        (unsigned char)(xbits >> (53ULL - FP8_SIGNIFICAND_BITS)) &
        FP8_MANTISSA_MASK;
    unsigned long long int absx = xbits & 0x7FFFFFFFFFFFFFFFULL;

    if (absx <= FP8_MINDENORM_O2) {
        // zero or underflow
        res = 0U;
    } else if (absx > DP_INF_BITS) {
        // NaN
        if (fp8_interpretation == E4M3) {
            res = 0x7FU;
        } else {
            // NaN --> QNaN
            res = 0x7EU | mantissa;
        }
    } else if (absx > FP8_OVERFLOW_THRESHOLD) {
        if (saturate == SATFINITE) {
            res = FP8_MAXNORM;
        } else {
            // NOSAT
            if (fp8_interpretation == E4M3) {
                // no Inf in E4M3
                res = 0x7FU; // NaN
            } else {
                res = 0x7CU; // Inf in E5M2
            }
        }
    } else if (absx >= FP8_MINNORM) {
        res = (unsigned char)((exp << (FP8_SIGNIFICAND_BITS - 1U)) | mantissa);
        // rounded-off bits
        unsigned long long int round =
            xbits & ((FP8_DP_HALF_ULP << 1ULL) - 1ULL);
        // round-to-nearest-even adjustment
        if ((round > FP8_DP_HALF_ULP) ||
            ((round == FP8_DP_HALF_ULP) && (mantissa & 1U))) {
            res = (unsigned char)(res + 1U);
        }
    } else // Denormal range
    {
        unsigned char shift = (unsigned char)(1U - exp);
        // add implicit leading bit
        mantissa |= (unsigned char)(1U << (FP8_SIGNIFICAND_BITS - 1U));
        // additional round-off due to denormalization
        res = (unsigned char)(mantissa >> shift);

        // rounded-off bits, including implicit leading bit
        unsigned long long int round =
            (xbits | ((unsigned long long int)1ULL << (53ULL - 1ULL))) &
            ((FP8_DP_HALF_ULP << (shift + 1ULL)) - 1ULL);
        // round-to-nearest-even adjustment
        if ((round > (FP8_DP_HALF_ULP << shift)) ||
            ((round == (FP8_DP_HALF_ULP << shift)) && (res & 1U))) {
            res = (unsigned char)(res + 1U);
        }
    }

    res |= sign;

    return (uint8_t)res;
}




__host__ __device__ uint16_t fp8_to_halfraw(const uint8_t x,
                        const fp8_interpretation_t fp8_interpretation) {
    uint16_t res;
    res = 0U;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    res.x =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)x, fp8_interpretation)
            .x;
#else
    unsigned short int ur = (unsigned short int)x;
    ur = (unsigned short int)(ur << 8U);

    if (fp8_interpretation == E5M2) {
        if ((ur & 0x7FFFU) > 0x7C00U) {
            /* If NaN, return canonical NaN */
            ur = 0x7FFFU;
        }
    } else { // __NV_E4M3
        unsigned short int sign = ur & 0x8000U;
        unsigned short int exponent =
            (unsigned short int)(((ur & 0x7800U) >> 1U) + 0x2000U);
        unsigned short int mantissa = (ur & 0x0700U) >> 1U;
        unsigned char absx = 0x7FU & (unsigned char)x;

        if (absx == 0x7FU) // NaN
        {
            ur = 0x7FFFU; // fp16 canonical NaN, discard sign
        } else if (exponent == 0x2000U) {
            // zero or denormal
            if (mantissa != 0U) {
                // normalize
                mantissa = (unsigned short int)(mantissa << 1U);
                while ((mantissa & 0x0400U) == 0U) {
                    mantissa = (unsigned short int)(mantissa << 1U);
                    exponent = (unsigned short int)(exponent - 0x0400U);
                }
                // discard implicit leading bit
                mantissa &= 0x03FFU;
            } else { // Zero
                exponent = 0U;
            }

            ur = (sign | exponent) | mantissa;
        } else {
            ur = (sign | exponent) | mantissa;
        }
    }
    res = ur;
#endif
    return res;
}


__host__ __device__ float halfraw_to_float(const uint16_t x) {
    float f;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    asm("{cvt.f32.f16 %0, %1;}\n" : "=f"(f) : "h"(x));
#else
    const unsigned int ux = (unsigned int)x;
    unsigned int sign = (ux >> 15U) & 1U;
    unsigned int exponent = (ux >> 10U) & 0x1fU;
    unsigned int mantissa = (ux & 0x3ffU) << 13U;
    if (exponent == 0x1fU) { /* NaN or Inf */
        /* discard sign of a NaN */
        sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
        mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
        exponent = 0xffU;
    } else if (exponent == 0U) { /* Denorm or Zero */
        if (mantissa != 0U) {
            unsigned int msb;
            exponent = 0x71U;
            do {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1U; /* normalize */
                --exponent;
            } while (msb == 0U);
            mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70U;
    }
    const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&f, &u, sizeof(u));
#else
    (void)std::memcpy(&f, &u, sizeof(u));
#endif
#endif /* (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 530) */
    return f;
}


template <typename T>
__global__ void quant_fp8_e4m3_kernel(const T *inputs, size_t n, uint8_t *outputs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        double tmp = static_cast<double>(inputs[idx]);
        uint8_t tmp_u8 = double_to_fp8(
            tmp,
            saturation_t::NOSAT,
            fp8_interpretation_t::E4M3
        );
        outputs[idx] = tmp_u8;
    }
}

template <typename T>
__global__ void dequant_fp8_e4m3_kernel(const uint8_t *inputs, size_t n, T *outputs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        auto halfraw = fp8_to_halfraw(inputs[idx], fp8_interpretation_t::E4M3);
        outputs[idx] = halfraw_to_float(halfraw);
    }
}

template <typename T>
__global__ void quant_dequant_fp8_e4m3_kernel(const T *inputs, size_t n, T *outputs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
        double tmp = static_cast<double>(inputs[idx]);
        uint8_t tmp_u8 = double_to_fp8(
            tmp,
            saturation_t::NOSAT,
            fp8_interpretation_t::E4M3
        );
        auto halfraw = fp8_to_halfraw(tmp_u8, fp8_interpretation_t::E4M3);
        outputs[idx] = halfraw_to_float(halfraw);
    }
}




at::Tensor quant_fp8_e4m3_cuda(at::Tensor inputs) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs, torch::kUInt8);
  AT_DISPATCH_FLOATING_TYPES(
      inputs.type().scalarType(), "quant_fp8_e4m3_cuda", [&] {
        quant_fp8_e4m3_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE>>>(
            inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<uint8_t>());
      });
  return outputs;
}

at::Tensor dequant_fp8_e4m3_cuda(at::Tensor inputs) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs, torch::kFloat);
  AT_DISPATCH_FLOATING_TYPES(
      outputs.type().scalarType(), "dequant_fp8_e4m3_cuda", [&] {
        dequant_fp8_e4m3_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE>>>(
            inputs.data_ptr<uint8_t>(), numel, outputs.data_ptr<scalar_t>());
      });
  return outputs;
}

at::Tensor quant_dequant_fp8_e4m3_cuda(at::Tensor inputs) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs);
  AT_DISPATCH_FLOATING_TYPES(
      inputs.type().scalarType(), "quant_fp8_e4m3_cuda", [&] {
        quant_dequant_fp8_e4m3_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE>>>(
            inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>());
      });
  return outputs;
}
