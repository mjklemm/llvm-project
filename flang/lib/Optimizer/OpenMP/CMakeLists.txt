get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  DoConcurrentConversion.cpp
  FunctionFiltering.cpp
  GenericLoopConversion.cpp
  GlobalFiltering.cpp
  MapsForPrivatizedSymbols.cpp
  MapInfoFinalization.cpp
  MarkDeclareTarget.cpp
  LowerWorkshare.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen
  FlangOpenMPPassesIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRCodeGen
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranCommon
  FortranEvaluate
  MLIRFuncDialect
  MLIRMathTransforms
  MLIROpenMPDialect
  HLFIRDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  ${dialect_libs}
)
