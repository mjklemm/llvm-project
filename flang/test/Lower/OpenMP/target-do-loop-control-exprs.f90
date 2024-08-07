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

subroutine foo_with_dummy_arg(nodes)
  implicit none
  integer, intent(inout) :: nodes( : )
  integer :: i

  !$omp target teams distribute parallel do
    do i = 1, ubound(nodes, 1)
      nodes(i) = i
    end do
  !$omp end target teams distribute parallel do
end subroutine

! CHECK: func.func @_QPfoo_with_dummy_arg(%[[FUNC_ARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "nodes"}) {

! CHECK:   %[[ARR_DECL:.*]]:2 = hlfir.declare %[[FUNC_ARG]] dummy_scope

! CHECK:   omp.map.info
! CHECK:   omp.map.info
! CHECK:   omp.map.info

! Verify that we get the box dims of the host array declaration not the target
! one.

! CHECK:   fir.box_dims %[[ARR_DECL]]

! CHECK:   omp.target

! CHECK: }


subroutine bounds_expr_in_loop_control(array)
  real, intent(out) :: array(:,:)
  integer :: bounds(2), i, j
  bounds = shape(array)

  !$omp target teams distribute parallel do simd collapse(2)
  do j = 1,bounds(2)
    do i = 1,bounds(1)
      array(i,j) = 0.
    enddo
  enddo
end subroutine bounds_expr_in_loop_control


! CHECK: func.func @_QPbounds_expr_in_loop_control(%[[FUNC_ARG:.*]]: {{.*}}) {

! CHECK:   %[[BOUNDS_DECL:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "{{.*}}Ebounds"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> ({{.*}})

! Verify that the host declaration of `bounds` (i.e. not the target/mapped one)
! is used for the trip count calculation. Trip count is calculation ops are emitted
! directly before the `omp.target` op and after all `omp.map.info` op; hence the
! `CHECK-NOT: ...` line.

! CHECK: hlfir.designate %[[BOUNDS_DECL:.*]]#0 (%c2{{.*}})
! CHECK: hlfir.designate %[[BOUNDS_DECL:.*]]#0 (%c1{{.*}})
! CHECK-NOT: omp.map.info
! CHECK: omp.target

! CHECK: }
