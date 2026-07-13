!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52 -Werror -Wno-experimental-option
!RUN: %flang -fopenmp -fopenmp-version=52 -Wno-experimental-option -Wno-openmp-map-save-without-always -fsyntax-only %s 2>&1 | FileCheck --allow-empty --check-prefix=NOWARN %s
!NOWARN-NOT: warning:

! Verify that mapping a variable with the SAVE attribute (implicit or explicit)
! produces a warning when the ALWAYS map modifier is not specified.

subroutine explicit_save_warn
  integer, save :: s
!WARNING: Variable 's' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine explicit_save_with_always
  integer, save :: s
  !$omp target map(always, tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine data_init_implicit_save
  ! A variable initialized in a DATA statement has an implicit SAVE attribute.
  integer :: s
  data s /0/
!WARNING: Variable 's' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine initializer_implicit_save
  ! A variable with a default initializer has an implicit SAVE attribute.
  integer :: s = 0
!WARNING: Variable 's' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s)
  s = s + 1
  !$omp end target
end

subroutine non_save_no_warn(x)
  integer :: x
  ! No SAVE attribute -> no warning.
  !$omp target map(tofrom: x)
  x = x + 1
  !$omp end target
end

subroutine mixed_list
  integer, save :: s
  integer :: x
!WARNING: Variable 's' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
  !$omp target map(tofrom: s, x)
  s = s + x
  !$omp end target
end

module m
  ! Module variables have implicit SAVE.
  integer :: mv = 0
  integer, save :: mve = 0
contains
  subroutine use_module_var
!WARNING: Variable 'mv' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
!WARNING: Variable 'mve' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
    !$omp target map(tofrom: mv, mve)
    mv = mv + 1
    !$omp end target
  end
  subroutine use_module_var_always
    !$omp target map(always, tofrom: mv,mve)
    mv = mv + 1
    !$omp end target
  end
end module
