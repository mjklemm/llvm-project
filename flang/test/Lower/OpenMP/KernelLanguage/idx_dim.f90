! RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

module mod1
contains
  subroutine sub1()
    use omp_lib
    integer :: tmp
    tmp = ompx_thread_id_x()
    tmp = ompx_thread_id_y()
    tmp = ompx_thread_id_z()
    tmp = ompx_block_id_x()
    tmp = ompx_block_id_y()
    tmp = ompx_block_id_z()
    tmp = ompx_block_dim_x()
    tmp = ompx_block_dim_y()
    tmp = ompx_block_dim_z()
    tmp = ompx_grid_dim_x()
    tmp = ompx_grid_dim_y()
    tmp = ompx_grid_dim_z()
  end subroutine sub1

  subroutine host()
    !$omp target teams num_teams(1, 2, 3)
    call sub1()
    !$omp end target teams
  end subroutine host
end module
