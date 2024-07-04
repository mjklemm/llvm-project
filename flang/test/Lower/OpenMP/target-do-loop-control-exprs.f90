! Verifies that if expressions are used to compute a target parallel loop, that
! no values escape the target region when flang emits the ops corresponding to
! these expressions (for example the compute the trip count for the target region).

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine foo(upper_bound)
  implicit none
  integer :: upper_bound
  integer :: nodes(1 : upper_bound)
  integer :: i

  !$omp target teams distribute parallel do
    do i = 1, ubound(nodes,1)
      nodes(i) = i
    end do
  !$omp end target teams distribute parallel do
end subroutine

! CHECK: func.func @_QPfoo(%[[FUNC_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "upper_bound"}) {
! CHECK:   %[[UB_ALLOC:.*]] = fir.alloca i32
! CHECK:   fir.dummy_scope : !fir.dscope
! CHECK:   %[[UB_DECL:.*]]:2 = hlfir.declare %[[FUNC_ARG]] {{.*}} {uniq_name = "_QFfooEupper_bound"}

! CHECK:   omp.map.info
! CHECK:   omp.map.info
! CHECK:   omp.map.info

! Verify that we load from the original/host allocation of the `upper_bound`
! variable rather than the corresponding target region arg.

! CHECK:   fir.load %[[UB_ALLOC]] : !fir.ref<i32>
! CHECK:   omp.target

! CHECK: }
