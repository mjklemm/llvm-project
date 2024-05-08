//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenMP/Utils.h"
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
    mlir::omp::TargetOp targetOp = nullptr;
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
  /// The first item in the retunred list is always the loop's induction
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
    if (auto refType = liveInType.dyn_cast<fir::ReferenceType>())
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

    genBodyOfTargetOp(rewriter, targetOp, liveIns, clauseOps.mapVars, mapper);
    return targetOp;
  }

  void genBodyOfTargetOp(mlir::ConversionPatternRewriter &rewriter,
                         mlir::omp::TargetOp targetOp,
                         llvm::ArrayRef<mlir::Value> liveIns,
                         llvm::ArrayRef<mlir::Value> liveInMapInfoOps,
                         mlir::IRMapping &mapper) const {
    mlir::Region &region = targetOp.getRegion();

    llvm::SmallVector<mlir::Type> liveInTypes;
    llvm::SmallVector<mlir::Location> liveInLocs;

    for (mlir::Value liveIn : liveIns) {
      liveInTypes.push_back(liveIn.getType());
      liveInLocs.push_back(liveIn.getLoc());
    }

    rewriter.createBlock(&region, {}, liveInTypes, liveInLocs);

    for (auto [arg, mapInfoOp] :
         llvm::zip_equal(region.getArguments(), liveInMapInfoOps)) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      hlfir::DeclareOp liveInDeclare = genLiveInDeclare(rewriter, arg, miOp);
      mapper.map(miOp.getVariableOperand(0), liveInDeclare.getBase());
    }

    auto terminator =
        rewriter.create<mlir::omp::TerminatorOp>(targetOp.getLoc());
    rewriter.setInsertionPoint(terminator);
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

    mlir::Block *teamsBlock = rewriter.createBlock(&teamsOp.getRegion());
    rewriter.create<mlir::omp::TerminatorOp>(loc);
    rewriter.setInsertionPointToStart(teamsBlock);

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

    mlir::Block *distBlock = rewriter.createBlock(&distOp.getRegion());
    rewriter.create<mlir::omp::TerminatorOp>(loc);
    rewriter.setInsertionPointToStart(distBlock);

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
    mlir::Block *parRegion = rewriter.createBlock(&parallelOp.getRegion());
    rewriter.create<mlir::omp::TerminatorOp>(loc);
    rewriter.setInsertionPointToStart(parRegion);

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

    // Clone the loop's body inside the worksharing construct using the
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
