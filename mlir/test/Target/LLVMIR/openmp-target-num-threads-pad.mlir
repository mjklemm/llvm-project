// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that when 'num_threads' is specified with fewer dimensions than the
// kernel rank (decided by 'num_teams' / 'thread_limit'), the unspecified
// trailing dimensions are implicitly 1.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  // Setup:
  //   num_teams    = (8, 4, 2)   3D
  //   thread_limit = (32, 4, 2)  3D
  //   num_threads  = (16, 8)     2D, dim 2 unspecified -> implicitly 1
  //
  // Combined per-dim min(thread_limit, num_threads-padded-with-1):
  //   dim 0: min(32, 16) = 16
  //   dim 1: min(4,  8)  = 4
  //   dim 2: min(2,  1)  = 1
  //
  // The resulting [3 x i32] num_threads array stored at slot 11 of
  // __tgt_kernel_arguments must therefore be [i32 16, i32 4, i32 1].

  // CHECK-LABEL: define void @num_threads_pad_missing_dim
  // CHECK: %[[KARGS:.*]] = alloca %struct.__tgt_kernel_arguments
  // CHECK: %[[NT_PTR:.*]] = getelementptr inbounds {{(nuw )?}}%struct.__tgt_kernel_arguments, ptr %[[KARGS]], i32 0, i32 11
  // CHECK-NEXT: store [3 x i32] [i32 16, i32 4, i32 1], ptr %[[NT_PTR]], align 4
  llvm.func @num_threads_pad_missing_dim() {
    %lb    = llvm.mlir.constant(0)  : i32
    %ub    = llvm.mlir.constant(10) : i32
    %step  = llvm.mlir.constant(1)  : i32
    %nt_x  = llvm.mlir.constant(8)  : i32
    %nt_y  = llvm.mlir.constant(4)  : i32
    %nt_z  = llvm.mlir.constant(2)  : i32
    %tl_x  = llvm.mlir.constant(32) : i32
    %tl_y  = llvm.mlir.constant(4)  : i32
    %tl_z  = llvm.mlir.constant(2)  : i32
    %ntr_x = llvm.mlir.constant(16) : i32
    %ntr_y = llvm.mlir.constant(8)  : i32
    omp.target host_eval(%lb    -> %arg_lb,    %ub    -> %arg_ub,    %step  -> %arg_step,
                         %nt_x  -> %arg_nt_x,  %nt_y  -> %arg_nt_y,  %nt_z  -> %arg_nt_z,
                         %tl_x  -> %arg_tl_x,  %tl_y  -> %arg_tl_y,  %tl_z  -> %arg_tl_z,
                         %ntr_x -> %arg_ntr_x, %ntr_y -> %arg_ntr_y
                         : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
      omp.teams num_teams(to %arg_nt_x, %arg_nt_y, %arg_nt_z : i32, i32, i32)
                thread_limit(%arg_tl_x, %arg_tl_y, %arg_tl_z : i32, i32, i32) {
        omp.parallel num_threads(%arg_ntr_x, %arg_ntr_y : i32, i32) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%i) : i32 = (%arg_lb) to (%arg_ub) inclusive step (%arg_step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }

  // Same setup, but 'num_threads' has *all* dims supplied. This is the baseline
  // to confirm the per-dim min pipeline is still correct.
  //
  //   num_teams    = (8, 4, 2)
  //   thread_limit = (32, 4, 2)
  //   num_threads  = (16, 8, 2)
  //
  //   dim 0: min(32, 16) = 16
  //   dim 1: min(4,  8)  = 4
  //   dim 2: min(2,  2)  = 2

  // CHECK-LABEL: define void @num_threads_full_dim
  // CHECK: %[[KARGS2:.*]] = alloca %struct.__tgt_kernel_arguments
  // CHECK: %[[NT_PTR2:.*]] = getelementptr inbounds {{(nuw )?}}%struct.__tgt_kernel_arguments, ptr %[[KARGS2]], i32 0, i32 11
  // CHECK-NEXT: store [3 x i32] [i32 16, i32 4, i32 2], ptr %[[NT_PTR2]], align 4
  llvm.func @num_threads_full_dim() {
    %lb    = llvm.mlir.constant(0)  : i32
    %ub    = llvm.mlir.constant(10) : i32
    %step  = llvm.mlir.constant(1)  : i32
    %nt_x  = llvm.mlir.constant(8)  : i32
    %nt_y  = llvm.mlir.constant(4)  : i32
    %nt_z  = llvm.mlir.constant(2)  : i32
    %tl_x  = llvm.mlir.constant(32) : i32
    %tl_y  = llvm.mlir.constant(4)  : i32
    %tl_z  = llvm.mlir.constant(2)  : i32
    %ntr_x = llvm.mlir.constant(16) : i32
    %ntr_y = llvm.mlir.constant(8)  : i32
    %ntr_z = llvm.mlir.constant(2)  : i32
    omp.target host_eval(%lb    -> %arg_lb,    %ub    -> %arg_ub,    %step  -> %arg_step,
                         %nt_x  -> %arg_nt_x,  %nt_y  -> %arg_nt_y,  %nt_z  -> %arg_nt_z,
                         %tl_x  -> %arg_tl_x,  %tl_y  -> %arg_tl_y,  %tl_z  -> %arg_tl_z,
                         %ntr_x -> %arg_ntr_x, %ntr_y -> %arg_ntr_y, %ntr_z -> %arg_ntr_z
                         : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
      omp.teams num_teams(to %arg_nt_x, %arg_nt_y, %arg_nt_z : i32, i32, i32)
                thread_limit(%arg_tl_x, %arg_tl_y, %arg_tl_z : i32, i32, i32) {
        omp.parallel num_threads(%arg_ntr_x, %arg_ntr_y, %arg_ntr_z : i32, i32, i32) {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%i) : i32 = (%arg_lb) to (%arg_ub) inclusive step (%arg_step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}
