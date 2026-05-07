! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51

! The "teams parallel" and "target teams parallel" combined constructs were
! added in OpenMP 6.0. They must be rejected on earlier versions.

subroutine f00(a)
  integer :: a
  !ERROR: TEAMS PARALLEL is not allowed in OpenMP v5.1, requires OpenMP v6.0 or later
  !$omp teams parallel
  a = a + 1
  !$omp end teams parallel
end

subroutine f01(a)
  integer :: a
  !ERROR: TARGET TEAMS PARALLEL is not allowed in OpenMP v5.1, requires OpenMP v6.0 or later
  !$omp target teams parallel
  a = a + 1
  !$omp end target teams parallel
end
