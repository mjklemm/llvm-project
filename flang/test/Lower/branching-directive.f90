!RUN: not flang-new -fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91526

! TODO Test temporarily disabled
!CHECK: error: verification of lowering to FIR failed

!COM: CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!COM: CHECK: ^bb[[THEN]]:
!COM: CHECK:   cf.br ^bb[[EXIT:[0-9]+]]
!COM: CHECK: ^bb[[ELSE]]:
!COM: CHECK:   fir.call @_FortranAStopStatement
!COM: CHECK:   fir.unreachable
!COM: CHECK: ^bb[[EXIT]]:

subroutine simple(y)
  implicit none
  logical, intent(in) :: y
  integer :: i
  if (y) then
!$omp parallel
    i = 1
!$omp end parallel
  else
    stop 1
  end if
end subroutine simple

