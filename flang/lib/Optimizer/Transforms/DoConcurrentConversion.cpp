//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <memory>
#include <utility>

namespace fir {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fopenmp-do-concurrent-conversion"

namespace Fortran {
namespace lower {
namespace omp {
mlir::omp::MapInfoOp
createMapInfoOp(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                llvm::ArrayRef<mlir::Value> bounds,
                llvm::ArrayRef<mlir::Value> members,
                mlir::DenseIntElementsAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap = false) {
  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, membersIndex, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name), builder.getBoolAttr(partialMap));

  return op;
}

mlir::Value calculateTripCount(fir::FirOpBuilder &builder, mlir::Location loc,
                               const mlir::omp::CollapseClauseOps &ops) {
  using namespace mlir::arith;
  assert(ops.loopLBVar.size() == ops.loopUBVar.size() &&
         ops.loopLBVar.size() == ops.loopStepVar.size() &&
         !ops.loopLBVar.empty() && "Invalid bounds or step");

  // Get the bit width of an integer-like type.
  auto widthOf = [](mlir::Type ty) -> unsigned {
    if (mlir::isa<mlir::IndexType>(ty)) {
      return mlir::IndexType::kInternalStorageBitWidth;
    }
    if (auto tyInt = mlir::dyn_cast<mlir::IntegerType>(ty)) {
      return tyInt.getWidth();
    }
    llvm_unreachable("Unexpected type");
  };

  // For a type that is either IntegerType or IndexType, return the
  // equivalent IntegerType. In the former case this is a no-op.
  auto asIntTy = [&](mlir::Type ty) -> mlir::IntegerType {
    if (ty.isIndex()) {
      return mlir::IntegerType::get(ty.getContext(), widthOf(ty));
    }
    assert(ty.isIntOrIndex() && "Unexpected type");
    return mlir::cast<mlir::IntegerType>(ty);
  };

  // For two given values, establish a common signless IntegerType
  // that can represent any value of type of x and of type of y,
  // and return the pair of x, y converted to the new type.
  auto unifyToSignless =
      [&](fir::FirOpBuilder &b, mlir::Value x,
          mlir::Value y) -> std::pair<mlir::Value, mlir::Value> {
    auto tyX = asIntTy(x.getType()), tyY = asIntTy(y.getType());
    unsigned width = std::max(widthOf(tyX), widthOf(tyY));
    auto wideTy = mlir::IntegerType::get(b.getContext(), width,
                                         mlir::IntegerType::Signless);
    return std::make_pair(b.createConvert(loc, wideTy, x),
                          b.createConvert(loc, wideTy, y));
  };

  // Start with signless i32 by default.
  auto tripCount = builder.createIntegerConstant(loc, builder.getI32Type(), 1);

  for (auto [origLb, origUb, origStep] :
       llvm::zip(ops.loopLBVar, ops.loopUBVar, ops.loopStepVar)) {
    auto tmpS0 = builder.createIntegerConstant(loc, origStep.getType(), 0);
    auto [step, step0] = unifyToSignless(builder, origStep, tmpS0);
    auto reverseCond =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, step, step0);
    auto negStep = builder.create<SubIOp>(loc, step0, step);
    mlir::Value absStep =
        builder.create<SelectOp>(loc, reverseCond, negStep, step);

    auto [lb, ub] = unifyToSignless(builder, origLb, origUb);
    auto start = builder.create<SelectOp>(loc, reverseCond, ub, lb);
    auto end = builder.create<SelectOp>(loc, reverseCond, lb, ub);

    mlir::Value range = builder.create<SubIOp>(loc, end, start);
    auto rangeCond =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, end, start);
    std::tie(range, absStep) = unifyToSignless(builder, range, absStep);
    // numSteps = (range /u absStep) + 1
    auto numSteps = builder.create<AddIOp>(
        loc, builder.create<DivUIOp>(loc, range, absStep),
        builder.createIntegerConstant(loc, range.getType(), 1));

    auto trip0 = builder.createIntegerConstant(loc, numSteps.getType(), 0);
    auto loopTripCount =
        builder.create<SelectOp>(loc, rangeCond, trip0, numSteps);
    auto [totalTC, thisTC] = unifyToSignless(builder, tripCount, loopTripCount);
    tripCount = builder.create<MulIOp>(loc, totalTC, thisTC);
  }

  return tripCount;
}
} // namespace omp
} // namespace lower
} // namespace Fortran

namespace {
class DoConcurrentConversion : public mlir::OpConversionPattern<fir::DoLoopOp> {
public:
  using mlir::OpConversionPattern<fir::DoLoopOp>::OpConversionPattern;

  DoConcurrentConversion(mlir::MLIRContext *context, bool mapToDevice)
      : OpConversionPattern(context), mapToDevice(mapToDevice) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Operation *lbOp = doLoop.getLowerBound().getDefiningOp();
    mlir::Operation *ubOp = doLoop.getUpperBound().getDefiningOp();
    mlir::Operation *stepOp = doLoop.getStep().getDefiningOp();

    if (lbOp == nullptr || ubOp == nullptr || stepOp == nullptr) {
      return rewriter.notifyMatchFailure(
          doLoop, "At least one of the loop's LB, UB, or step doesn't have a "
                  "defining operation.");
    }

    std::function<bool(mlir::Operation *)> isOpUltimatelyConstant =
        [&](mlir::Operation *operation) {
          if (mlir::isa_and_present<mlir::arith::ConstantOp>(operation))
            return true;

          if (auto convertOp =
                  mlir::dyn_cast_if_present<fir::ConvertOp>(operation))
            return isOpUltimatelyConstant(convertOp.getValue().getDefiningOp());

          return false;
        };

    if (!isOpUltimatelyConstant(lbOp) || !isOpUltimatelyConstant(ubOp) ||
        !isOpUltimatelyConstant(stepOp)) {
      return rewriter.notifyMatchFailure(
          doLoop, "`do concurrent` conversion is currently only supported for "
                  "constant LB, UB, and step values.");
    }

    llvm::SmallVector<mlir::Value> liveIns;
    collectLoopLiveIns(doLoop, liveIns);
    assert(!liveIns.empty());

    mlir::IRMapping mapper;
    mlir::omp::TargetOp targetOp;
    mlir::omp::LoopNestClauseOps loopNestClauseOps;

    if (mapToDevice) {
      mlir::omp::TargetClauseOps clauseOps;
      for (mlir::Value liveIn : liveIns)
        clauseOps.mapVars.push_back(genMapInfoOpForLiveIn(rewriter, liveIn));
      targetOp =
          genTargetOp(doLoop.getLoc(), rewriter, mapper, liveIns, clauseOps);
      genTeamsOp(doLoop.getLoc(), rewriter, doLoop, liveIns, mapper,
                 loopNestClauseOps);
      genDistributeOp(doLoop.getLoc(), rewriter);
    }

    genParallelOp(doLoop.getLoc(), rewriter, doLoop, liveIns, mapper,
                  loopNestClauseOps);
    genWsLoopOp(rewriter, doLoop, mapper, loopNestClauseOps);

    // Now that we created the nested `ws.loop` op, we set can the `target` op's
    // trip count.
    if (mapToDevice) {
      rewriter.setInsertionPoint(targetOp);
      auto parentModule = doLoop->getParentOfType<mlir::ModuleOp>();
      fir::FirOpBuilder firBuilder(rewriter, fir::getKindMapping(parentModule));

      mlir::omp::CollapseClauseOps collapseClauseOps;
      collapseClauseOps.loopLBVar.push_back(lbOp->getResult(0));
      collapseClauseOps.loopUBVar.push_back(ubOp->getResult(0));
      collapseClauseOps.loopStepVar.push_back(stepOp->getResult(0));

      mlir::cast<mlir::omp::TargetOp>(targetOp).getTripCountMutable().assign(
          Fortran::lower::omp::calculateTripCount(firBuilder, doLoop.getLoc(),
                                                  collapseClauseOps));
    }

    rewriter.eraseOp(doLoop);
    return mlir::success();
  }

private:
  /// Collect the list of values used inside the loop but defined outside of it.
  /// The first item in the returned list is always the loop's induction
  /// variable.
  void collectLoopLiveIns(fir::DoLoopOp doLoop,
                          llvm::SmallVectorImpl<mlir::Value> &liveIns) const {
    // Given an operation `op`, this lambda returns true if `op`'s operand is
    // ultimately the loop's induction variable. Detecting this helps finding
    // the live-in value corresponding to the induction variable in case the
    // induction variable is indirectly used in the loop (e.g. throught a cast
    // op).
    std::function<bool(mlir::Operation * op)> isIndVarUltimateOperand =
        [&](mlir::Operation *op) {
          if (auto storeOp = mlir::dyn_cast_if_present<fir::StoreOp>(op)) {
            return (storeOp.getValue() == doLoop.getInductionVar()) ||
                   isIndVarUltimateOperand(storeOp.getValue().getDefiningOp());
          }

          if (auto convertOp = mlir::dyn_cast_if_present<fir::ConvertOp>(op)) {
            return convertOp.getOperand() == doLoop.getInductionVar() ||
                   isIndVarUltimateOperand(
                       convertOp.getValue().getDefiningOp());
          }

          return false;
        };

    llvm::SmallDenseSet<mlir::Value> seenValues;
    llvm::SmallDenseSet<mlir::Operation *> seenOps;

    mlir::visitUsedValuesDefinedAbove(
        doLoop.getRegion(), [&](mlir::OpOperand *operand) {
          if (!seenValues.insert(operand->get()).second)
            return;

          mlir::Operation *definingOp = operand->get().getDefiningOp();
          // We want to collect ops corresponding to live-ins only once.
          if (definingOp && !seenOps.insert(definingOp).second)
            return;

          liveIns.push_back(operand->get());

          if (isIndVarUltimateOperand(operand->getOwner()))
            std::swap(*liveIns.begin(), *liveIns.rbegin());
        });
  }

  void genBoundsOps(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, hlfir::DeclareOp declareOp,
                    llvm::SmallVectorImpl<mlir::Value> &boundsOps) const {
    if (declareOp.getShape() == nullptr) {
      return;
    }

    auto shapeOp = mlir::dyn_cast_if_present<fir::ShapeOp>(
        declareOp.getShape().getDefiningOp());

    if (shapeOp == nullptr)
      TODO(loc, "Shapes not defined by shape op's are not supported yet.");

    auto extents = shapeOp.getExtents();

    auto genBoundsOp = [&](mlir::Value extent) {
      mlir::Type extentType = extent.getType();
      auto lb = rewriter.create<mlir::arith::ConstantOp>(
          loc, extentType, rewriter.getIntegerAttr(extentType, 0));
      // TODO I think this caluclation might not be correct. But this is how
      // it is done in PFT->OpenMP lowering. So keeping it like this until we
      // double check.
      mlir::Value ub = rewriter.create<mlir::arith::SubIOp>(loc, extent, lb);

      return rewriter.create<mlir::omp::MapBoundsOp>(
          loc, rewriter.getType<mlir::omp::MapBoundsType>(), lb, ub, extent,
          mlir::Value{}, false, mlir::Value{});
    };

    for (auto extent : extents)
      boundsOps.push_back(genBoundsOp(extent));
  }

  mlir::omp::MapInfoOp
  genMapInfoOpForLiveIn(mlir::ConversionPatternRewriter &rewriter,
                        mlir::Value liveIn) const {
    auto declareOp =
        mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveIn.getDefiningOp());

    if (declareOp == nullptr)
      TODO(liveIn.getLoc(),
           "Values not defined by declare op's are not supported yet.");

    mlir::Type liveInType = liveIn.getType();
    mlir::Type eleType = liveInType;
    if (auto refType = mlir::dyn_cast<fir::ReferenceType>(liveInType))
      eleType = refType.getElementType();

    llvm::omp::OpenMPOffloadMappingFlags mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
    mlir::omp::VariableCaptureKind captureKind =
        mlir::omp::VariableCaptureKind::ByRef;

    if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
      captureKind = mlir::omp::VariableCaptureKind::ByCopy;
    } else if (!fir::isa_builtin_cptr_type(eleType)) {
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
    }

    llvm::SmallVector<mlir::Value> boundsOps;
    genBoundsOps(rewriter, liveIn.getLoc(), declareOp, boundsOps);

    return Fortran::lower::omp::createMapInfoOp(
        rewriter, liveIn.getLoc(), declareOp.getBase(), /*varPtrPtr=*/{},
        declareOp.getUniqName().str(), boundsOps, /*members=*/{},
        /*membersIndex=*/mlir::DenseIntElementsAttr{},
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            mapFlag),
        captureKind, liveInType);
  }

  mlir::omp::TargetOp genTargetOp(mlir::Location loc,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  mlir::IRMapping &mapper,
                                  llvm::ArrayRef<mlir::Value> liveIns,
                                  mlir::omp::TargetClauseOps &clauseOps) const {
    auto targetOp = rewriter.create<mlir::omp::TargetOp>(loc, clauseOps);

    mlir::Region &region = targetOp.getRegion();

    llvm::SmallVector<mlir::Type> liveInTypes;
    llvm::SmallVector<mlir::Location> liveInLocs;

    for (mlir::Value liveIn : liveIns) {
      liveInTypes.push_back(liveIn.getType());
      liveInLocs.push_back(liveIn.getLoc());
    }

    rewriter.createBlock(&region, {}, liveInTypes, liveInLocs);

    for (auto [arg, mapInfoOp] :
         llvm::zip_equal(region.getArguments(), clauseOps.mapVars)) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      hlfir::DeclareOp liveInDeclare = genLiveInDeclare(rewriter, arg, miOp);
      mapper.map(miOp.getVariableOperand(0), liveInDeclare.getBase());
    }

    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(targetOp.getLoc()));

    return targetOp;
  }

  hlfir::DeclareOp
  genLiveInDeclare(mlir::ConversionPatternRewriter &rewriter,
                   mlir::Value liveInArg,
                   mlir::omp::MapInfoOp liveInMapInfoOp) const {
    mlir::Type liveInType = liveInArg.getType();

    if (fir::isa_ref_type(liveInType))
      liveInType = fir::unwrapRefType(liveInType);

    mlir::Value shape = [&]() -> mlir::Value {
      if (hlfir::isFortranScalarNumericalType(liveInType))
        return {};

      if (hlfir::isFortranArrayObject(liveInType)) {
        llvm::SmallVector<mlir::Value> shapeOpOperands;

        for (auto boundsOperand : liveInMapInfoOp.getBounds()) {
          auto boundsOp =
              mlir::cast<mlir::omp::MapBoundsOp>(boundsOperand.getDefiningOp());
          mlir::Operation *localExtentDef =
              boundsOp.getExtent().getDefiningOp()->clone();
          rewriter.getInsertionBlock()->push_back(localExtentDef);
          assert(localExtentDef->getNumResults() == 1);

          shapeOpOperands.push_back(localExtentDef->getResult(0));
        }

        return rewriter.create<fir::ShapeOp>(liveInArg.getLoc(),
                                             shapeOpOperands);
      }

      std::string opStr;
      llvm::raw_string_ostream opOs(opStr);
      opOs << "Unsupported type: " << liveInType;
      llvm_unreachable(opOs.str().c_str());
    }();

    return rewriter.create<hlfir::DeclareOp>(liveInArg.getLoc(), liveInArg,
                                             liveInMapInfoOp.getName().value(),
                                             shape);
  }

  mlir::omp::TeamsOp
  genTeamsOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
             fir::DoLoopOp doLoop, llvm::ArrayRef<mlir::Value> liveIns,
             mlir::IRMapping &mapper,
             mlir::omp::LoopNestClauseOps &loopNestClauseOps) const {
    auto teamsOp = rewriter.create<mlir::omp::TeamsOp>(
        loc, /*clauses=*/mlir::omp::TeamsClauseOps{});

    rewriter.createBlock(&teamsOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    genInductionVariableAlloc(rewriter, liveIns, mapper);
    genLoopNestClauseOps(loc, rewriter, doLoop, mapper, loopNestClauseOps);

    return teamsOp;
  }

  void
  genLoopNestClauseOps(mlir::Location loc,
                       mlir::ConversionPatternRewriter &rewriter,
                       fir::DoLoopOp doLoop, mlir::IRMapping &mapper,
                       mlir::omp::LoopNestClauseOps &loopNestClauseOps) const {
    assert(loopNestClauseOps.loopLBVar.empty() &&
           "Loop nest bounds were already emitted!");

    // Clones the chain of ops defining a certain loop bound or its step into
    // the parallel region. For example, if the value of a bound is defined by a
    // `fir.convert`op, this lambda clones the `fir.convert` as well as the
    // value it converts from. We do this since `omp.target` regions are
    // isolated from above.
    std::function<mlir::Operation *(mlir::Operation *)>
        cloneBoundOrStepDefChain = [&](mlir::Operation *operation) {
          if (mlir::isa_and_present<mlir::arith::ConstantOp>(operation))
            return rewriter.clone(*operation, mapper);

          if (auto convertOp =
                  mlir::dyn_cast_if_present<fir::ConvertOp>(operation)) {
            cloneBoundOrStepDefChain(convertOp.getValue().getDefiningOp());
            return rewriter.clone(*operation, mapper);
          }

          std::string opStr;
          llvm::raw_string_ostream opOs(opStr);
          opOs << "Unexpected operation: " << *operation;
          llvm_unreachable(opOs.str().c_str());
        };

    mlir::Operation *lbOp = doLoop.getLowerBound().getDefiningOp();
    mlir::Operation *ubOp = doLoop.getUpperBound().getDefiningOp();
    mlir::Operation *stepOp = doLoop.getStep().getDefiningOp();

    loopNestClauseOps.loopLBVar.push_back(
        cloneBoundOrStepDefChain(lbOp)->getResult(0));
    loopNestClauseOps.loopLBVar.push_back(
        cloneBoundOrStepDefChain(ubOp)->getResult(0));
    loopNestClauseOps.loopLBVar.push_back(
        cloneBoundOrStepDefChain(stepOp)->getResult(0));
    loopNestClauseOps.loopInclusiveAttr = rewriter.getUnitAttr();
  }

  mlir::omp::DistributeOp
  genDistributeOp(mlir::Location loc,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto distOp = rewriter.create<mlir::omp::DistributeOp>(
        loc, /*clauses=*/mlir::omp::DistributeClauseOps{});

    rewriter.createBlock(&distOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    return distOp;
  }

  void genInductionVariableAlloc(mlir::ConversionPatternRewriter &rewriter,
                                 llvm::ArrayRef<mlir::Value> liveIns,
                                 mlir::IRMapping &mapper) const {
    mlir::Operation *indVarMemDef = liveIns.front().getDefiningOp();

    assert(
        indVarMemDef != nullptr &&
        "Induction variable memdef is expected to have a defining operation.");

    llvm::SmallSetVector<mlir::Operation *, 2> indVarDeclareAndAlloc;
    for (auto operand : indVarMemDef->getOperands())
      indVarDeclareAndAlloc.insert(operand.getDefiningOp());
    indVarDeclareAndAlloc.insert(indVarMemDef);

    for (mlir::Operation *opToClone : indVarDeclareAndAlloc)
      rewriter.clone(*opToClone, mapper);
  }

  mlir::omp::ParallelOp
  genParallelOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                fir::DoLoopOp doLoop, llvm::ArrayRef<mlir::Value> liveIns,
                mlir::IRMapping &mapper,
                mlir::omp::LoopNestClauseOps &loopNestClauseOps) const {
    auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
    rewriter.createBlock(&parallelOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    // If mapping to host, the local induction variable and loop bounds need to
    // be emitted as part of the `omp.parallel` op.
    if (!mapToDevice) {
      genInductionVariableAlloc(rewriter, liveIns, mapper);
      genLoopNestClauseOps(loc, rewriter, doLoop, mapper, loopNestClauseOps);
    }

    return parallelOp;
  }

  mlir::omp::LoopNestOp
  genWsLoopOp(mlir::ConversionPatternRewriter &rewriter, fir::DoLoopOp doLoop,
              mlir::IRMapping &mapper,
              const mlir::omp::LoopNestClauseOps &clauseOps) const {

    auto wsloopOp = rewriter.create<mlir::omp::WsloopOp>(doLoop.getLoc());
    rewriter.createBlock(&wsloopOp.getRegion());
    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(wsloopOp.getLoc()));

    auto loopNestOp =
        rewriter.create<mlir::omp::LoopNestOp>(doLoop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(doLoop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    mlir::Operation *terminator = loopNestOp.getRegion().back().getTerminator();
    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    rewriter.create<mlir::omp::YieldOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);

    return loopNestOp;
  }

  bool mapToDevice;
};

class DoConcurrentConversionPass
    : public fir::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  using fir::impl::DoConcurrentConversionPassBase<
      DoConcurrentConversionPass>::DoConcurrentConversionPassBase;

  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const fir::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration()) {
      return;
    }

    auto *context = &getContext();

    if (mapTo != "host" && mapTo != "device") {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }

    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(context, mapTo == "device");
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<fir::FIROpsDialect, hlfir::hlfirDialect,
                           mlir::arith::ArithDialect, mlir::func::FuncDialect,
                           mlir::omp::OpenMPDialect>();

    target.addDynamicallyLegalOp<fir::DoLoopOp>(
        [](fir::DoLoopOp op) { return !op.getUnordered(); });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting do-concurrent op");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
fir::createDoConcurrentConversionPass(bool mapToDevice) {
  DoConcurrentConversionPassOptions options;
  options.mapTo = mapToDevice ? "device" : "host";

  return std::make_unique<DoConcurrentConversionPass>(options);
}
