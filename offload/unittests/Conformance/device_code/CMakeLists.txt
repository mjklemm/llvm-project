add_offload_test_device_code(LLVMLibm.c llvm-libm -stdlib -fno-builtin)

add_custom_target(conformance_device_binaries DEPENDS llvm-libm.bin)
set(OFFLOAD_CONFORMANCE_DEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
