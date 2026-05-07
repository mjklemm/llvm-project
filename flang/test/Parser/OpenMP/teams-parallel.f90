!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(a)
  integer :: a
  !$omp teams parallel
  a = a + 1
  !$omp end teams parallel
end

subroutine f01(a)
  integer :: a
  !$omp target teams parallel
  a = a + 1
  !$omp end target teams parallel
end

!UNPARSE: !$OMP TEAMS PARALLEL
!UNPARSE: !$OMP END TEAMS PARALLEL
!UNPARSE: !$OMP TARGET TEAMS PARALLEL
!UNPARSE: !$OMP END TARGET TEAMS PARALLEL

!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = teams parallel
!PARSE-TREE: OmpDirectiveName -> llvm::omp::Directive = target teams parallel
