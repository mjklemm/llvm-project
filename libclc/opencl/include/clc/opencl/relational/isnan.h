//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_RELATIONAL_ISNAN_H__
#define __CLC_OPENCL_RELATIONAL_ISNAN_H__

#include <clc/opencl/opencl-base.h>

#define _CLC_ISNAN_DECL(RET_TYPE, ARG_TYPE)                                    \
  _CLC_OVERLOAD _CLC_DECL RET_TYPE isnan(ARG_TYPE);

#define _CLC_VECTOR_ISNAN_DECL(RET_TYPE, ARG_TYPE)                             \
  _CLC_ISNAN_DECL(RET_TYPE##2, ARG_TYPE##2)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##3, ARG_TYPE##3)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##4, ARG_TYPE##4)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##8, ARG_TYPE##8)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##16, ARG_TYPE##16)

_CLC_ISNAN_DECL(int, float)
_CLC_VECTOR_ISNAN_DECL(int, float)

#ifdef cl_khr_fp64
_CLC_ISNAN_DECL(int, double)
_CLC_VECTOR_ISNAN_DECL(long, double)
#endif

#ifdef cl_khr_fp16
_CLC_ISNAN_DECL(int, half)
_CLC_VECTOR_ISNAN_DECL(short, half)
#endif

#undef _CLC_ISNAN_DECL
#undef _CLC_VECTOR_ISNAN_DECL

#endif // __CLC_OPENCL_RELATIONAL_ISNAN_H__
