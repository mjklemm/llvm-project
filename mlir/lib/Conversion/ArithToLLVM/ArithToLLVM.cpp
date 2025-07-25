//===- ArithToLLVM.cpp - Arithmetic to LLVM dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_ARITHTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Operations whose conversion will depend on whether they are passed a
/// rounding mode attribute or not.
///
/// `SourceOp` is the source operation; `TargetOp`, the operation it will lower
/// to; `AttrConvert` is the attribute conversion to convert the rounding mode
/// attribute.
template <typename SourceOp, typename TargetOp, bool Constrained,
          template <typename, typename> typename AttrConvert =
              AttrConvertPassThrough>
struct ConstrainedVectorConvertToLLVMPattern
    : public VectorConvertToLLVMPattern<SourceOp, TargetOp, AttrConvert> {
  using VectorConvertToLLVMPattern<SourceOp, TargetOp,
                                   AttrConvert>::VectorConvertToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (Constrained != static_cast<bool>(op.getRoundingModeAttr()))
      return failure();
    return VectorConvertToLLVMPattern<SourceOp, TargetOp,
                                      AttrConvert>::matchAndRewrite(op, adaptor,
                                                                    rewriter);
  }
};

/// No-op bitcast. Propagate type input arg if converted source and dest types
/// are the same.
struct IdentityBitcastLowering final
    : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value src = adaptor.getIn();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (src.getType() != resultType)
      return rewriter.notifyMatchFailure(op, "Types are different");

    rewriter.replaceOp(op, src);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

using AddFOpLowering =
    VectorConvertToLLVMPattern<arith::AddFOp, LLVM::FAddOp,
                               arith::AttrConvertFastMathToLLVM>;
using AddIOpLowering =
    VectorConvertToLLVMPattern<arith::AddIOp, LLVM::AddOp,
                               arith::AttrConvertOverflowToLLVM>;
using AndIOpLowering = VectorConvertToLLVMPattern<arith::AndIOp, LLVM::AndOp>;
using BitcastOpLowering =
    VectorConvertToLLVMPattern<arith::BitcastOp, LLVM::BitcastOp>;
using DivFOpLowering =
    VectorConvertToLLVMPattern<arith::DivFOp, LLVM::FDivOp,
                               arith::AttrConvertFastMathToLLVM>;
using DivSIOpLowering =
    VectorConvertToLLVMPattern<arith::DivSIOp, LLVM::SDivOp>;
using DivUIOpLowering =
    VectorConvertToLLVMPattern<arith::DivUIOp, LLVM::UDivOp>;
using ExtFOpLowering = VectorConvertToLLVMPattern<arith::ExtFOp, LLVM::FPExtOp>;
using ExtSIOpLowering =
    VectorConvertToLLVMPattern<arith::ExtSIOp, LLVM::SExtOp>;
using ExtUIOpLowering =
    VectorConvertToLLVMPattern<arith::ExtUIOp, LLVM::ZExtOp>;
using FPToSIOpLowering =
    VectorConvertToLLVMPattern<arith::FPToSIOp, LLVM::FPToSIOp>;
using FPToUIOpLowering =
    VectorConvertToLLVMPattern<arith::FPToUIOp, LLVM::FPToUIOp>;
using MaximumFOpLowering =
    VectorConvertToLLVMPattern<arith::MaximumFOp, LLVM::MaximumOp,
                               arith::AttrConvertFastMathToLLVM>;
using MaxNumFOpLowering =
    VectorConvertToLLVMPattern<arith::MaxNumFOp, LLVM::MaxNumOp,
                               arith::AttrConvertFastMathToLLVM>;
using MaxSIOpLowering =
    VectorConvertToLLVMPattern<arith::MaxSIOp, LLVM::SMaxOp>;
using MaxUIOpLowering =
    VectorConvertToLLVMPattern<arith::MaxUIOp, LLVM::UMaxOp>;
using MinimumFOpLowering =
    VectorConvertToLLVMPattern<arith::MinimumFOp, LLVM::MinimumOp,
                               arith::AttrConvertFastMathToLLVM>;
using MinNumFOpLowering =
    VectorConvertToLLVMPattern<arith::MinNumFOp, LLVM::MinNumOp,
                               arith::AttrConvertFastMathToLLVM>;
using MinSIOpLowering =
    VectorConvertToLLVMPattern<arith::MinSIOp, LLVM::SMinOp>;
using MinUIOpLowering =
    VectorConvertToLLVMPattern<arith::MinUIOp, LLVM::UMinOp>;
using MulFOpLowering =
    VectorConvertToLLVMPattern<arith::MulFOp, LLVM::FMulOp,
                               arith::AttrConvertFastMathToLLVM>;
using MulIOpLowering =
    VectorConvertToLLVMPattern<arith::MulIOp, LLVM::MulOp,
                               arith::AttrConvertOverflowToLLVM>;
using NegFOpLowering =
    VectorConvertToLLVMPattern<arith::NegFOp, LLVM::FNegOp,
                               arith::AttrConvertFastMathToLLVM>;
using OrIOpLowering = VectorConvertToLLVMPattern<arith::OrIOp, LLVM::OrOp>;
using RemFOpLowering =
    VectorConvertToLLVMPattern<arith::RemFOp, LLVM::FRemOp,
                               arith::AttrConvertFastMathToLLVM>;
using RemSIOpLowering =
    VectorConvertToLLVMPattern<arith::RemSIOp, LLVM::SRemOp>;
using RemUIOpLowering =
    VectorConvertToLLVMPattern<arith::RemUIOp, LLVM::URemOp>;
using SelectOpLowering =
    VectorConvertToLLVMPattern<arith::SelectOp, LLVM::SelectOp>;
using ShLIOpLowering =
    VectorConvertToLLVMPattern<arith::ShLIOp, LLVM::ShlOp,
                               arith::AttrConvertOverflowToLLVM>;
using ShRSIOpLowering =
    VectorConvertToLLVMPattern<arith::ShRSIOp, LLVM::AShrOp>;
using ShRUIOpLowering =
    VectorConvertToLLVMPattern<arith::ShRUIOp, LLVM::LShrOp>;
using SIToFPOpLowering =
    VectorConvertToLLVMPattern<arith::SIToFPOp, LLVM::SIToFPOp>;
using SubFOpLowering =
    VectorConvertToLLVMPattern<arith::SubFOp, LLVM::FSubOp,
                               arith::AttrConvertFastMathToLLVM>;
using SubIOpLowering =
    VectorConvertToLLVMPattern<arith::SubIOp, LLVM::SubOp,
                               arith::AttrConvertOverflowToLLVM>;
using TruncFOpLowering =
    ConstrainedVectorConvertToLLVMPattern<arith::TruncFOp, LLVM::FPTruncOp,
                                          false>;
using ConstrainedTruncFOpLowering = ConstrainedVectorConvertToLLVMPattern<
    arith::TruncFOp, LLVM::ConstrainedFPTruncIntr, true,
    arith::AttrConverterConstrainedFPToLLVM>;
using TruncIOpLowering =
    VectorConvertToLLVMPattern<arith::TruncIOp, LLVM::TruncOp,
                               arith::AttrConvertOverflowToLLVM>;
using UIToFPOpLowering =
    VectorConvertToLLVMPattern<arith::UIToFPOp, LLVM::UIToFPOp>;
using XOrIOpLowering = VectorConvertToLLVMPattern<arith::XOrIOp, LLVM::XOrOp>;

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

/// Directly lower to LLVM op.
struct ConstantOpLowering : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
template <typename OpTy, typename ExtCastTy>
struct IndexCastOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

using IndexCastOpSILowering =
    IndexCastOpLowering<arith::IndexCastOp, LLVM::SExtOp>;
using IndexCastOpUILowering =
    IndexCastOpLowering<arith::IndexCastUIOp, LLVM::ZExtOp>;

struct AddUIExtendedOpLowering
    : public ConvertOpToLLVMPattern<arith::AddUIExtendedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename ArithMulOp, bool IsSigned>
struct MulIExtendedOpLowering : public ConvertOpToLLVMPattern<ArithMulOp> {
  using ConvertOpToLLVMPattern<ArithMulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArithMulOp op, typename ArithMulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

using MulSIExtendedOpLowering =
    MulIExtendedOpLowering<arith::MulSIExtendedOp, true>;
using MulUIExtendedOpLowering =
    MulIExtendedOpLowering<arith::MulUIExtendedOp, false>;

struct CmpIOpLowering : public ConvertOpToLLVMPattern<arith::CmpIOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CmpFOpLowering : public ConvertOpToLLVMPattern<arith::CmpFOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOpLowering::matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  return LLVM::detail::oneToOneRewrite(op, LLVM::ConstantOp::getOperationName(),
                                       adaptor.getOperands(), op->getAttrs(),
                                       *getTypeConverter(), rewriter);
}

//===----------------------------------------------------------------------===//
// IndexCastOpLowering
//===----------------------------------------------------------------------===//

template <typename OpTy, typename ExtCastTy>
LogicalResult IndexCastOpLowering<OpTy, ExtCastTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type resultType = op.getResult().getType();
  Type targetElementType =
      this->typeConverter->convertType(getElementTypeOrSelf(resultType));
  Type sourceElementType =
      this->typeConverter->convertType(getElementTypeOrSelf(op.getIn()));
  unsigned targetBits = targetElementType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceElementType.getIntOrFloatBitWidth();

  if (targetBits == sourceBits) {
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }

  // Handle the scalar and 1D vector cases.
  Type operandType = adaptor.getIn().getType();
  if (!isa<LLVM::LLVMArrayType>(operandType)) {
    Type targetType = this->typeConverter->convertType(resultType);
    if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, targetType,
                                                 adaptor.getIn());
    else
      rewriter.replaceOpWithNewOp<ExtCastTy>(op, targetType, adaptor.getIn());
    return success();
  }

  if (!isa<VectorType>(resultType))
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return LLVM::detail::handleMultidimensionalVectors(
      op.getOperation(), adaptor.getOperands(), *(this->getTypeConverter()),
      [&](Type llvm1DVectorTy, ValueRange operands) -> Value {
        typename OpTy::Adaptor adaptor(operands);
        if (targetBits < sourceBits) {
          return LLVM::TruncOp::create(rewriter, op.getLoc(), llvm1DVectorTy,
                                       adaptor.getIn());
        }
        return ExtCastTy::create(rewriter, op.getLoc(), llvm1DVectorTy,
                                 adaptor.getIn());
      },
      rewriter);
}

//===----------------------------------------------------------------------===//
// AddUIExtendedOpLowering
//===----------------------------------------------------------------------===//

LogicalResult AddUIExtendedOpLowering::matchAndRewrite(
    arith::AddUIExtendedOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type operandType = adaptor.getLhs().getType();
  Type sumResultType = op.getSum().getType();
  Type overflowResultType = op.getOverflow().getType();

  if (!LLVM::isCompatibleType(operandType))
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();

  // Handle the scalar and 1D vector cases.
  if (!isa<LLVM::LLVMArrayType>(operandType)) {
    Type newOverflowType = typeConverter->convertType(overflowResultType);
    Type structType =
        LLVM::LLVMStructType::getLiteral(ctx, {sumResultType, newOverflowType});
    Value addOverflow = LLVM::UAddWithOverflowOp::create(
        rewriter, loc, structType, adaptor.getLhs(), adaptor.getRhs());
    Value sumExtracted =
        LLVM::ExtractValueOp::create(rewriter, loc, addOverflow, 0);
    Value overflowExtracted =
        LLVM::ExtractValueOp::create(rewriter, loc, addOverflow, 1);
    rewriter.replaceOp(op, {sumExtracted, overflowExtracted});
    return success();
  }

  if (!isa<VectorType>(sumResultType))
    return rewriter.notifyMatchFailure(loc, "expected vector result types");

  return rewriter.notifyMatchFailure(loc,
                                     "ND vector types are not supported yet");
}

//===----------------------------------------------------------------------===//
// MulIExtendedOpLowering
//===----------------------------------------------------------------------===//

template <typename ArithMulOp, bool IsSigned>
LogicalResult MulIExtendedOpLowering<ArithMulOp, IsSigned>::matchAndRewrite(
    ArithMulOp op, typename ArithMulOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type resultType = adaptor.getLhs().getType();

  if (!LLVM::isCompatibleType(resultType))
    return failure();

  Location loc = op.getLoc();

  // Handle the scalar and 1D vector cases. Because LLVM does not have a
  // matching extended multiplication intrinsic, perform regular multiplication
  // on operands zero-extended to i(2*N) bits, and truncate the results back to
  // iN types.
  if (!isa<LLVM::LLVMArrayType>(resultType)) {
    // Shift amount necessary to extract the high bits from widened result.
    TypedAttr shiftValAttr;

    if (auto intTy = dyn_cast<IntegerType>(resultType)) {
      unsigned resultBitwidth = intTy.getWidth();
      auto attrTy = rewriter.getIntegerType(resultBitwidth * 2);
      shiftValAttr = rewriter.getIntegerAttr(attrTy, resultBitwidth);
    } else {
      auto vecTy = cast<VectorType>(resultType);
      unsigned resultBitwidth = vecTy.getElementTypeBitWidth();
      auto attrTy = VectorType::get(
          vecTy.getShape(), rewriter.getIntegerType(resultBitwidth * 2));
      shiftValAttr = SplatElementsAttr::get(
          attrTy, APInt(resultBitwidth * 2, resultBitwidth));
    }
    Type wideType = shiftValAttr.getType();
    assert(LLVM::isCompatibleType(wideType) &&
           "LLVM dialect should support all signless integer types");

    using LLVMExtOp = std::conditional_t<IsSigned, LLVM::SExtOp, LLVM::ZExtOp>;
    Value lhsExt = LLVMExtOp::create(rewriter, loc, wideType, adaptor.getLhs());
    Value rhsExt = LLVMExtOp::create(rewriter, loc, wideType, adaptor.getRhs());
    Value mulExt = LLVM::MulOp::create(rewriter, loc, wideType, lhsExt, rhsExt);

    // Split the 2*N-bit wide result into two N-bit values.
    Value low = LLVM::TruncOp::create(rewriter, loc, resultType, mulExt);
    Value shiftVal = LLVM::ConstantOp::create(rewriter, loc, shiftValAttr);
    Value highExt = LLVM::LShrOp::create(rewriter, loc, mulExt, shiftVal);
    Value high = LLVM::TruncOp::create(rewriter, loc, resultType, highExt);

    rewriter.replaceOp(op, {low, high});
    return success();
  }

  if (!isa<VectorType>(resultType))
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return rewriter.notifyMatchFailure(op,
                                     "ND vector types are not supported yet");
}

//===----------------------------------------------------------------------===//
// CmpIOpLowering
//===----------------------------------------------------------------------===//

// Convert arith.cmp predicate into the LLVM dialect CmpPredicate. The two enums
// share numerical values so just cast.
template <typename LLVMPredType, typename PredType>
static LLVMPredType convertCmpPredicate(PredType pred) {
  return static_cast<LLVMPredType>(pred);
}

LogicalResult
CmpIOpLowering::matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Type operandType = adaptor.getLhs().getType();
  Type resultType = op.getResult().getType();

  // Handle the scalar and 1D vector cases.
  if (!isa<LLVM::LLVMArrayType>(operandType)) {
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }

  if (!isa<VectorType>(resultType))
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return LLVM::detail::handleMultidimensionalVectors(
      op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
      [&](Type llvm1DVectorTy, ValueRange operands) {
        OpAdaptor adaptor(operands);
        return LLVM::ICmpOp::create(
            rewriter, op.getLoc(), llvm1DVectorTy,
            convertCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
            adaptor.getLhs(), adaptor.getRhs());
      },
      rewriter);
}

//===----------------------------------------------------------------------===//
// CmpFOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOpLowering::matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Type operandType = adaptor.getLhs().getType();
  Type resultType = op.getResult().getType();
  LLVM::FastmathFlags fmf =
      arith::convertArithFastMathFlagsToLLVM(op.getFastmath());

  // Handle the scalar and 1D vector cases.
  if (!isa<LLVM::LLVMArrayType>(operandType)) {
    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, typeConverter->convertType(resultType),
        convertCmpPredicate<LLVM::FCmpPredicate>(op.getPredicate()),
        adaptor.getLhs(), adaptor.getRhs(), fmf);
    return success();
  }

  if (!isa<VectorType>(resultType))
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  return LLVM::detail::handleMultidimensionalVectors(
      op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
      [&](Type llvm1DVectorTy, ValueRange operands) {
        OpAdaptor adaptor(operands);
        return LLVM::FCmpOp::create(
            rewriter, op.getLoc(), llvm1DVectorTy,
            convertCmpPredicate<LLVM::FCmpPredicate>(op.getPredicate()),
            adaptor.getLhs(), adaptor.getRhs(), fmf);
      },
      rewriter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ArithToLLVMConversionPass
    : public impl::ArithToLLVMConversionPassBase<ArithToLLVMConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(&getContext(), options);
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert MemRef to LLVM.
struct ArithToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::arith::registerConvertArithToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    dialect->addInterfaces<ArithToLLVMDialectInterface>();
  });
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {

  // Set a higher pattern benefit for IdentityBitcastLowering so it will run
  // before BitcastOpLowering.
  patterns.add<IdentityBitcastLowering>(converter, patterns.getContext(),
                                        /*patternBenefit*/ 10);

  // clang-format off
  patterns.add<
    AddFOpLowering,
    AddIOpLowering,
    AndIOpLowering,
    AddUIExtendedOpLowering,
    BitcastOpLowering,
    ConstantOpLowering,
    CmpFOpLowering,
    CmpIOpLowering,
    DivFOpLowering,
    DivSIOpLowering,
    DivUIOpLowering,
    ExtFOpLowering,
    ExtSIOpLowering,
    ExtUIOpLowering,
    FPToSIOpLowering,
    FPToUIOpLowering,
    IndexCastOpSILowering,
    IndexCastOpUILowering,
    MaximumFOpLowering,
    MaxNumFOpLowering,
    MaxSIOpLowering,
    MaxUIOpLowering,
    MinimumFOpLowering,
    MinNumFOpLowering,
    MinSIOpLowering,
    MinUIOpLowering,
    MulFOpLowering,
    MulIOpLowering,
    MulSIExtendedOpLowering,
    MulUIExtendedOpLowering,
    NegFOpLowering,
    OrIOpLowering,
    RemFOpLowering,
    RemSIOpLowering,
    RemUIOpLowering,
    SelectOpLowering,
    ShLIOpLowering,
    ShRSIOpLowering,
    ShRUIOpLowering,
    SIToFPOpLowering,
    SubFOpLowering,
    SubIOpLowering,
    TruncFOpLowering,
    ConstrainedTruncFOpLowering,
    TruncIOpLowering,
    UIToFPOpLowering,
    XOrIOpLowering
  >(converter);
  // clang-format on
}
