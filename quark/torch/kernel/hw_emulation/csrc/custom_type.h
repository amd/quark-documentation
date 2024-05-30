//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef CUSTOM_TYPE_H
#define CUSTOM_TYPE_H


#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                        \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                            \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

typedef enum fp8_interpretation_t {
    E4M3,
    E5M2,
} fp8_interpretation_t;

typedef enum saturation_t {
    NOSAT,
    SATFINITE,
} saturation_t;


#endif // CUSTOM_TYPE_H
