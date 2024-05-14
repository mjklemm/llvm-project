add_flang_library(FIRTransforms
  AbstractResult.cpp
  AddAliasTags.cpp
  AffinePromotion.cpp
  AffineDemotion.cpp
  AnnotateConstant.cpp
  CharacterConversion.cpp
  ControlFlowConverter.cpp
  ArrayValueCopy.cpp
  ExternalNameConversion.cpp
  MemoryAllocation.cpp
  StackArrays.cpp
  MemRefDataFlowOpt.cpp
  SimplifyRegionLite.cpp
  AlgebraicSimplification.cpp
  SimplifyIntrinsics.cpp
  AddDebugInfo.cpp
  PolymorphicOpConversion.cpp
  LoopVersioning.cpp
  OMPFunctionFiltering.cpp
  OMPMapInfoFinalization.cpp
  OMPMarkDeclareTarget.cpp
  VScaleAttr.cpp
  FunctionAttr.cpp
  DebugTypeGenerator.cpp
  DoConcurrentConversion.cpp
  # TODO Find a cleaner solution for this. This is a workaround to expose 
  # `Utils.cpp` so that it be used the `DoConcurrentConversion` pass. We should
  # probably split this is a shared lib.
  ../../Lower/OpenMP/Utils.cpp

  DEPENDS
  FIRDialect
  FIROptTransformsPassIncGen
  HLFIROpsIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRCodeGen
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranCommon
  HLFIRDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMathTransforms
  MLIROpenACCDialect
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPDialect
)
