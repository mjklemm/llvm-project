! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

type :: t1
   integer :: y
end type

type :: t2
   integer :: y
end type

!ERROR: DECLARE_MAPPER directive should have a single argument
!WARNING: Variable 'x' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
!$omp declare mapper(m1:t1::x, m2:t2::x) map(x, x%y)

integer :: x(10)
!ERROR: The argument to the DECLARE_MAPPER directive should be a mapper-specifier
!WARNING: Variable 'x' has the SAVE attribute and is mapped without the ALWAYS map modifier; the host value may not be synchronized with the device on subsequent target regions [-Wopenmp-map-save-without-always]
!$omp declare mapper(x) map(to: x)

end
