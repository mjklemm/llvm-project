// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +reference-types -target-feature +simd128 -target-feature +relaxed-simd -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -target-feature +gc -target-feature +fp16 -flax-vector-conversions=none -O3 -emit-llvm -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-feature +reference-types -target-feature +simd128 -target-feature +relaxed-simd -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -target-feature +gc  -target-feature +fp16 -flax-vector-conversions=none -O3 -emit-llvm -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY64
// RUN: not %clang_cc1 -triple wasm64-unknown-unknown -target-feature +reference-types -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -target-feature +gc -flax-vector-conversions=none -O3 -emit-llvm -o - %s 2>&1 | FileCheck %s -check-prefixes MISSING-SIMD

// SIMD convenience types
typedef signed char i8x16 __attribute((vector_size(16)));
typedef short i16x8 __attribute((vector_size(16)));
typedef int i32x4 __attribute((vector_size(16)));
typedef long long i64x2 __attribute((vector_size(16)));
typedef unsigned char u8x16 __attribute((vector_size(16)));
typedef unsigned short u16x8 __attribute((vector_size(16)));
typedef unsigned int u32x4 __attribute((vector_size(16)));
typedef unsigned long long u64x2 __attribute((vector_size(16)));
typedef __fp16 f16x8 __attribute((vector_size(16)));
typedef float f32x4 __attribute((vector_size(16)));
typedef double f64x2 __attribute((vector_size(16)));

__SIZE_TYPE__ memory_size(void) {
  return __builtin_wasm_memory_size(0);
  // WEBASSEMBLY32: call {{i.*}} @llvm.wasm.memory.size.i32(i32 0)
  // WEBASSEMBLY64: call {{i.*}} @llvm.wasm.memory.size.i64(i32 0)
}

__SIZE_TYPE__ memory_grow(__SIZE_TYPE__ delta) {
  return __builtin_wasm_memory_grow(0, delta);
  // WEBASSEMBLY32: call i32 @llvm.wasm.memory.grow.i32(i32 0, i32 %{{.*}})
  // WEBASSEMBLY64: call i64 @llvm.wasm.memory.grow.i64(i32 0, i64 %{{.*}})
}

__SIZE_TYPE__ tls_size(void) {
  return __builtin_wasm_tls_size();
  // WEBASSEMBLY32: call i32 @llvm.wasm.tls.size.i32()
  // WEBASSEMBLY64: call i64 @llvm.wasm.tls.size.i64()
}

__SIZE_TYPE__ tls_align(void) {
  return __builtin_wasm_tls_align();
  // WEBASSEMBLY32: call i32 @llvm.wasm.tls.align.i32()
  // WEBASSEMBLY64: call i64 @llvm.wasm.tls.align.i64()
}

void *tls_base(void) {
  return __builtin_wasm_tls_base();
  // WEBASSEMBLY: call ptr @llvm.wasm.tls.base()
}

void throw(void *obj) {
  return __builtin_wasm_throw(0, obj);
  // WEBASSEMBLY: call void @llvm.wasm.throw(i32 0, ptr %{{.*}})
}

void rethrow(void) {
  return __builtin_wasm_rethrow();
  // WEBASSEMBLY32: call void @llvm.wasm.rethrow()
  // WEBASSEMBLY64: call void @llvm.wasm.rethrow()
}

int memory_atomic_wait32(int *addr, int expected, long long timeout) {
  return __builtin_wasm_memory_atomic_wait32(addr, expected, timeout);
  // WEBASSEMBLY: call i32 @llvm.wasm.memory.atomic.wait32(ptr %{{.*}}, i32 %{{.*}}, i64 %{{.*}})
}

int memory_atomic_wait64(long long *addr, long long expected, long long timeout) {
  return __builtin_wasm_memory_atomic_wait64(addr, expected, timeout);
  // WEBASSEMBLY: call i32 @llvm.wasm.memory.atomic.wait64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
}

unsigned int memory_atomic_notify(int *addr, unsigned int count) {
  return __builtin_wasm_memory_atomic_notify(addr, count);
  // WEBASSEMBLY: call i32 @llvm.wasm.memory.atomic.notify(ptr %{{.*}}, i32 %{{.*}})
}

int trunc_s_i32_f32(float f) {
  return __builtin_wasm_trunc_s_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.signed.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_u_i32_f32(float f) {
  return __builtin_wasm_trunc_u_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.unsigned.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_s_i32_f64(double f) {
  return __builtin_wasm_trunc_s_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.signed.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_u_i32_f64(double f) {
  return __builtin_wasm_trunc_u_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.unsigned.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_s_i64_f32(float f) {
  return __builtin_wasm_trunc_s_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.signed.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_u_i64_f32(float f) {
  return __builtin_wasm_trunc_u_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.unsigned.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_s_i64_f64(double f) {
  return __builtin_wasm_trunc_s_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.signed.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_u_i64_f64(double f) {
  return __builtin_wasm_trunc_u_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.unsigned.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_s_i32_f32(float f) {
  return __builtin_wasm_trunc_saturate_s_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.fptosi.sat.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_u_i32_f32(float f) {
  return __builtin_wasm_trunc_saturate_u_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.fptoui.sat.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_s_i32_f64(double f) {
  return __builtin_wasm_trunc_saturate_s_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.fptosi.sat.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_u_i32_f64(double f) {
  return __builtin_wasm_trunc_saturate_u_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.fptoui.sat.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_s_i64_f32(float f) {
  return __builtin_wasm_trunc_saturate_s_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.fptosi.sat.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_u_i64_f32(float f) {
  return __builtin_wasm_trunc_saturate_u_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.fptoui.sat.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_s_i64_f64(double f) {
  return __builtin_wasm_trunc_saturate_s_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.fptosi.sat.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_u_i64_f64(double f) {
  return __builtin_wasm_trunc_saturate_u_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.fptoui.sat.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

float min_f32(float x, float y) {
  return __builtin_wasm_min_f32(x, y);
  // WEBASSEMBLY: call float @llvm.minimum.f32(float %x, float %y)
  // WEBASSEMBLY-NEXT: ret
}

float max_f32(float x, float y) {
  return __builtin_wasm_max_f32(x, y);
  // WEBASSEMBLY: call float @llvm.maximum.f32(float %x, float %y)
  // WEBASSEMBLY-NEXT: ret
}

double min_f64(double x, double y) {
  return __builtin_wasm_min_f64(x, y);
  // WEBASSEMBLY: call double @llvm.minimum.f64(double %x, double %y)
  // WEBASSEMBLY-NEXT: ret
}

double max_f64(double x, double y) {
  return __builtin_wasm_max_f64(x, y);
  // WEBASSEMBLY: call double @llvm.maximum.f64(double %x, double %y)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 abs_i8x16(i8x16 v) {
  return __builtin_wasm_abs_i8x16(v);
  // MISSING-SIMD: error: '__builtin_wasm_abs_i8x16' needs target feature simd128
  // WEBASSEMBLY: call <16 x i8> @llvm.abs.v16i8(<16 x i8> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 abs_i16x8(i16x8 v) {
  return __builtin_wasm_abs_i16x8(v);
  // WEBASSEMBLY: call <8 x i16> @llvm.abs.v8i16(<8 x i16> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 abs_i32x4(i32x4 v) {
  return __builtin_wasm_abs_i32x4(v);
  // WEBASSEMBLY: call <4 x i32> @llvm.abs.v4i32(<4 x i32> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

i64x2 abs_i64x2(i64x2 v) {
  return __builtin_wasm_abs_i64x2(v);
  // WEBASSEMBLY: call <2 x i64> @llvm.abs.v2i64(<2 x i64> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

u8x16 avgr_u_i8x16(u8x16 x, u8x16 y) {
  return __builtin_wasm_avgr_u_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.avgr.unsigned.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 avgr_u_i16x8(u16x8 x, u16x8 y) {
  return __builtin_wasm_avgr_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.avgr.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 q15mulr_sat_s_i16x8(i16x8 x, i16x8 y) {
  return __builtin_wasm_q15mulr_sat_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.q15mulr.sat.signed(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 extadd_pairwise_i8x16_s_i16x8(i8x16 v) {
  return __builtin_wasm_extadd_pairwise_i8x16_s_i16x8(v);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extadd.pairwise.signed.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %v)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 extadd_pairwise_i8x16_u_i16x8(u8x16 v) {
  return __builtin_wasm_extadd_pairwise_i8x16_u_i16x8(v);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extadd.pairwise.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %v)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 extadd_pairwise_i16x8_s_i32x4(i16x8 v) {
  return __builtin_wasm_extadd_pairwise_i16x8_s_i32x4(v);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extadd.pairwise.signed.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %v)
  // WEBASSEMBLY-NEXT: ret
}

u32x4 extadd_pairwise_i16x8_u_i32x4(u16x8 v) {
  return __builtin_wasm_extadd_pairwise_i16x8_u_i32x4(v);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extadd.pairwise.unsigned.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %v)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 dot_i16x8_s(i16x8 x, i16x8 y) {
  return __builtin_wasm_dot_s_i32x4_i16x8(x, y);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.dot(<8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 bitselect(i32x4 x, i32x4 y, i32x4 c) {
  return __builtin_wasm_bitselect(x, y, c);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.bitselect.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %x, <4 x i32> %y, <4 x i32> %c)
  // WEBASSEMBLY-NEXT: ret
}

int any_true_v128(i8x16 x) {
  return __builtin_wasm_any_true_v128(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int all_true_i8x16(i8x16 x) {
  return __builtin_wasm_all_true_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int all_true_i16x8(i16x8 x) {
  return __builtin_wasm_all_true_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int all_true_i32x4(i32x4 x) {
  return __builtin_wasm_all_true_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int all_true_i64x2(i64x2 x) {
  return __builtin_wasm_all_true_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i8x16(i8x16 x) {
  return __builtin_wasm_bitmask_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i16x8(i16x8 x) {
  return __builtin_wasm_bitmask_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i32x4(i32x4 x) {
  return __builtin_wasm_bitmask_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i64x2(i64x2 x) {
  return __builtin_wasm_bitmask_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

f32x4 abs_f32x4(f32x4 x) {
  return __builtin_wasm_abs_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 abs_f64x2(f64x2 x) {
  return __builtin_wasm_abs_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.fabs.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f32x4 min_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_min_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.minimum.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 max_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_max_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.maximum.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 pmin_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_pmin_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.pmin.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 pmax_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_pmax_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.pmax.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 min_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_min_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.minimum.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 max_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_max_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.maximum.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 pmin_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_pmin_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.pmin.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 pmax_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_pmax_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.pmax.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 ceil_f32x4(f32x4 x) {
  return __builtin_wasm_ceil_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 floor_f32x4(f32x4 x) {
  return __builtin_wasm_floor_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.floor.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 trunc_f32x4(f32x4 x) {
  return __builtin_wasm_trunc_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.trunc.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 nearest_f32x4(f32x4 x) {
  return __builtin_wasm_nearest_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 ceil_f64x2(f64x2 x) {
  return __builtin_wasm_ceil_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.ceil.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 floor_f64x2(f64x2 x) {
  return __builtin_wasm_floor_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.floor.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 trunc_f64x2(f64x2 x) {
  return __builtin_wasm_trunc_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.trunc.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 nearest_f64x2(f64x2 x) {
  return __builtin_wasm_nearest_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f32x4 sqrt_f32x4(f32x4 x) {
  return __builtin_wasm_sqrt_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 sqrt_f64x2(f64x2 x) {
  return __builtin_wasm_sqrt_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

i32x4 trunc_saturate_s_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_trunc_saturate_s_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.fptosi.sat.v4i32.v4f32(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 trunc_saturate_u_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_trunc_saturate_u_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.fptoui.sat.v4i32.v4f32(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 narrow_s_i8x16_i16x8(i16x8 low, i16x8 high) {
  return __builtin_wasm_narrow_s_i8x16_i16x8(low, high);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %low, <8 x i16> %high)
  // WEBASSEMBLY: ret
}

u8x16 narrow_u_i8x16_i16x8(i16x8 low, i16x8 high) {
  return __builtin_wasm_narrow_u_i8x16_i16x8(low, high);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.narrow.unsigned.v16i8.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %low, <8 x i16> %high)
  // WEBASSEMBLY: ret
}

i16x8 narrow_s_i16x8_i32x4(i32x4 low, i32x4 high) {
  return __builtin_wasm_narrow_s_i16x8_i32x4(low, high);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.narrow.signed.v8i16.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %low, <4 x i32> %high)
  // WEBASSEMBLY: ret
}

u16x8 narrow_u_i16x8_i32x4(i32x4 low, i32x4 high) {
  return __builtin_wasm_narrow_u_i16x8_i32x4(low, high);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %low, <4 x i32> %high)
  // WEBASSEMBLY: ret
}

i32x4 trunc_sat_s_zero_f64x2_i32x4(f64x2 x) {
  return __builtin_wasm_trunc_sat_s_zero_f64x2_i32x4(x);
  // WEBASSEMBLY: %0 = tail call <2 x i32> @llvm.fptosi.sat.v2i32.v2f64(<2 x double> %x)
  // WEBASSEMBLY: %1 = shufflevector <2 x i32> %0, <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // WEBASSEMBLY: ret <4 x i32> %1
}

u32x4 trunc_sat_u_zero_f64x2_i32x4(f64x2 x) {
  return __builtin_wasm_trunc_sat_u_zero_f64x2_i32x4(x);
  // WEBASSEMBLY: %0 = tail call <2 x i32> @llvm.fptoui.sat.v2i32.v2f64(<2 x double> %x)
  // WEBASSEMBLY: %1 = shufflevector <2 x i32> %0, <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // WEBASSEMBLY: ret <4 x i32> %1
}

i8x16 swizzle_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_swizzle_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %x, <16 x i8> %y)
}

i8x16 shuffle(i8x16 x, i8x16 y) {
  return __builtin_wasm_shuffle_i8x16(x, y, 0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.shuffle(<16 x i8> %x, <16 x i8> %y,
  // WEBASSEMBLY-SAME: i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
  // WEBASSEMBLY-SAME: i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14,
  // WEBASSEMBLY-SAME: i32 15
  // WEBASSEMBLY-NEXT: ret
}

f32x4 madd_f32x4(f32x4 a, f32x4 b, f32x4 c) {
  return __builtin_wasm_relaxed_madd_f32x4(a, b, c);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.relaxed.madd.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %a, <4 x float> %b, <4 x float> %c)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 nmadd_f32x4(f32x4 a, f32x4 b, f32x4 c) {
  return __builtin_wasm_relaxed_nmadd_f32x4(a, b, c);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.relaxed.nmadd.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %a, <4 x float> %b, <4 x float> %c)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 madd_f64x2(f64x2 a, f64x2 b, f64x2 c) {
  return __builtin_wasm_relaxed_madd_f64x2(a, b, c);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.relaxed.madd.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %a, <2 x double> %b, <2 x double> %c)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 nmadd_f64x2(f64x2 a, f64x2 b, f64x2 c) {
  return __builtin_wasm_relaxed_nmadd_f64x2(a, b, c);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.relaxed.nmadd.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %a, <2 x double> %b, <2 x double> %c)
  // WEBASSEMBLY-NEXT: ret
}

f16x8 madd_f16x8(f16x8 a, f16x8 b, f16x8 c) {
  return __builtin_wasm_relaxed_madd_f16x8(a, b, c);
  // WEBASSEMBLY: call <8 x half> @llvm.wasm.relaxed.madd.v8f16(
  // WEBASSEMBLY-SAME: <8 x half> %a, <8 x half> %b, <8 x half> %c)
  // WEBASSEMBLY-NEXT: ret
}

f16x8 nmadd_f16x8(f16x8 a, f16x8 b, f16x8 c) {
  return __builtin_wasm_relaxed_nmadd_f16x8(a, b, c);
  // WEBASSEMBLY: call <8 x half> @llvm.wasm.relaxed.nmadd.v8f16(
  // WEBASSEMBLY-SAME: <8 x half> %a, <8 x half> %b, <8 x half> %c)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 laneselect_i8x16(i8x16 a, i8x16 b, i8x16 c) {
  return __builtin_wasm_relaxed_laneselect_i8x16(a, b, c);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.relaxed.laneselect.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %a, <16 x i8> %b, <16 x i8> %c)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 laneselect_i16x8(i16x8 a, i16x8 b, i16x8 c) {
  return __builtin_wasm_relaxed_laneselect_i16x8(a, b, c);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.relaxed.laneselect.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %a, <8 x i16> %b, <8 x i16> %c)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 laneselect_i32x4(i32x4 a, i32x4 b, i32x4 c) {
  return __builtin_wasm_relaxed_laneselect_i32x4(a, b, c);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.relaxed.laneselect.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  // WEBASSEMBLY-NEXT: ret
}

i64x2 laneselect_i64x2(i64x2 a, i64x2 b, i64x2 c) {
  return __builtin_wasm_relaxed_laneselect_i64x2(a, b, c);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.relaxed.laneselect.v2i64(
  // WEBASSEMBLY-SAME: <2 x i64> %a, <2 x i64> %b, <2 x i64> %c)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 relaxed_swizzle_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_relaxed_swizzle_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.relaxed.swizzle(<16 x i8> %x, <16 x i8> %y)
}

f32x4 relaxed_min_f32x4(f32x4 a, f32x4 b) {
  return __builtin_wasm_relaxed_min_f32x4(a, b);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.relaxed.min.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %a, <4 x float> %b)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 relaxed_max_f32x4(f32x4 a, f32x4 b) {
  return __builtin_wasm_relaxed_max_f32x4(a, b);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.relaxed.max.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %a, <4 x float> %b)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 relaxed_min_f64x2(f64x2 a, f64x2 b) {
  return __builtin_wasm_relaxed_min_f64x2(a, b);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.relaxed.min.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %a, <2 x double> %b)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 relaxed_max_f64x2(f64x2 a, f64x2 b) {
  return __builtin_wasm_relaxed_max_f64x2(a, b);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.relaxed.max.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %a, <2 x double> %b)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 relaxed_trunc_s_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_relaxed_trunc_s_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.relaxed.trunc.signed(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

u32x4 relaxed_trunc_u_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_relaxed_trunc_u_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.relaxed.trunc.unsigned(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 relaxed_trunc_s_zero_i32x4_f64x2(f64x2 x) {
  return __builtin_wasm_relaxed_trunc_s_zero_i32x4_f64x2(x);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.relaxed.trunc.signed.zero(<2 x double> %x)
  // WEBASSEMBLY-NEXT: ret
}

u32x4 relaxed_trunc_u_zero_i32x4_f64x2(f64x2 x) {
  return __builtin_wasm_relaxed_trunc_u_zero_i32x4_f64x2(x);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.relaxed.trunc.unsigned.zero(<2 x double> %x)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 relaxed_q15mulr_s_i16x8(i16x8 a, i16x8 b) {
  return __builtin_wasm_relaxed_q15mulr_s_i16x8(a, b);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.relaxed.q15mulr.signed(
  // WEBASSEMBLY-SAME: <8 x i16> %a, <8 x i16> %b)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 dot_i8x16_i7x16_s_i16x8(i8x16 a, i8x16 b) {
  return __builtin_wasm_relaxed_dot_i8x16_i7x16_s_i16x8(a, b);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.relaxed.dot.i8x16.i7x16.signed(
  // WEBASSEMBLY-SAME: <16 x i8> %a, <16 x i8> %b)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 dot_i8x16_i7x16_add_s_i32x4(i8x16 a, i8x16 b, i32x4 c) {
  return __builtin_wasm_relaxed_dot_i8x16_i7x16_add_s_i32x4(a, b, c);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.relaxed.dot.i8x16.i7x16.add.signed(
  // WEBASSEMBLY-SAME: <16 x i8> %a, <16 x i8> %b, <4 x i32> %c)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 relaxed_dot_bf16x8_add_f32_f32x4(u16x8 a, u16x8 b, f32x4 c) {
  return __builtin_wasm_relaxed_dot_bf16x8_add_f32_f32x4(a, b, c);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.relaxed.dot.bf16x8.add.f32
  // WEBASSEMBLY-SAME: <8 x i16> %a, <8 x i16> %b, <4 x float> %c)
  // WEBASSEMBLY-NEXT: ret
}

float load_f16_f32(__fp16 *addr) {
  return __builtin_wasm_loadf16_f32(addr);
  // WEBASSEMBLY: call float @llvm.wasm.loadf16.f32(ptr %{{.*}})
}

void store_f16_f32(float val, __fp16 *addr) {
  return __builtin_wasm_storef16_f32(val, addr);
  // WEBASSEMBLY: tail call void @llvm.wasm.storef16.f32(float %val, ptr %{{.*}})
  // WEBASSEMBLY-NEXT: ret
}

f16x8 splat_f16x8(float a) {
  // WEBASSEMBLY: %0 = tail call <8 x half> @llvm.wasm.splat.f16x8(float %a)
  // WEBASSEMBLY-NEXT: ret <8 x half> %0
  return __builtin_wasm_splat_f16x8(a);
}

float extract_lane_f16x8(f16x8 a) {
  // WEBASSEMBLY:  %0 = tail call float @llvm.wasm.extract.lane.f16x8(<8 x half> %a, i32 7)
  // WEBASSEMBLY-NEXT: ret float %0
  return __builtin_wasm_extract_lane_f16x8(a, 7);
}

f16x8 replace_lane_f16x8(f16x8 a, float v) {
  // WEBASSEMBLY:  %0 = tail call <8 x half> @llvm.wasm.replace.lane.f16x8(<8 x half> %a, i32 7, float %v)
  // WEBASSEMBLY-NEXT: ret <8 x half> %0
  return __builtin_wasm_replace_lane_f16x8(a, 7, v);
}

f16x8 min_f16x8(f16x8 a, f16x8 b) {
  // WEBASSEMBLY:  %0 = tail call <8 x half> @llvm.minimum.v8f16(<8 x half> %a, <8 x half> %b)
  // WEBASSEMBLY-NEXT: ret <8 x half> %0
  return __builtin_wasm_min_f16x8(a, b);
}

f16x8 max_f16x8(f16x8 a, f16x8 b) {
  // WEBASSEMBLY:  %0 = tail call <8 x half> @llvm.maximum.v8f16(<8 x half> %a, <8 x half> %b)
  // WEBASSEMBLY-NEXT: ret <8 x half> %0
  return __builtin_wasm_max_f16x8(a, b);
}

f16x8 pmin_f16x8(f16x8 a, f16x8 b) {
  // WEBASSEMBLY:  %0 = tail call <8 x half> @llvm.wasm.pmin.v8f16(<8 x half> %a, <8 x half> %b)
  // WEBASSEMBLY-NEXT: ret <8 x half> %0
  return __builtin_wasm_pmin_f16x8(a, b);
}

f16x8 pmax_f16x8(f16x8 a, f16x8 b) {
  // WEBASSEMBLY:  %0 = tail call <8 x half> @llvm.wasm.pmax.v8f16(<8 x half> %a, <8 x half> %b)
  // WEBASSEMBLY-NEXT: ret <8 x half> %0
  return __builtin_wasm_pmax_f16x8(a, b);
}
__externref_t externref_null() {
  return __builtin_wasm_ref_null_extern();
  // WEBASSEMBLY: tail call ptr addrspace(10) @llvm.wasm.ref.null.extern()
  // WEBASSEMBLY-NEXT: ret
}

int externref_is_null(__externref_t arg) {
  return __builtin_wasm_ref_is_null_extern(arg);
  // WEBASSEMBLY: tail call i32 @llvm.wasm.ref.is_null.extern(ptr addrspace(10) %arg)
  // WEBASSEMBLY-NEXT: ret
}

void *tp (void) {
  return __builtin_thread_pointer ();
  // WEBASSEMBLY: call {{.*}} @llvm.thread.pointer.p0()
}

typedef void (*Fvoid)(void);
typedef float (*Ffloats)(float, double, int);
typedef void (*Fpointers)(Fvoid, Ffloats, void*, int*, int***, char[5]);

void use(int);

void test_function_pointer_signature_void(Fvoid func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

void test_function_pointer_signature_floats(Ffloats func) {
  // WEBASSEMBLY:  tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, float 0.000000e+00, token poison, float 0.000000e+00, double 0.000000e+00, i32 0)
  use(__builtin_wasm_test_function_pointer_signature(func));
}

void test_function_pointer_signature_pointers(Fpointers func) {
  // WEBASSEMBLY:  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null)
  use(__builtin_wasm_test_function_pointer_signature(func));
}
