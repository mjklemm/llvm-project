//===-- Utils..cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include <flang/Lower/OpenMP/Utils.h>

#include <flang/Lower/AbstractConverter.h>
#include <flang/Lower/ConvertType.h>
#include <flang/Lower/OpenMP/Clauses.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

llvm::cl::opt<bool> treatIndexAsSection(
    "openmp-treat-index-as-section",
    llvm::cl::desc("In the OpenMP data clauses treat `a(N)` as `a(N:N)`."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableDelayedPrivatization(
    "openmp-enable-delayed-privatization",
    llvm::cl::desc(
        "Emit `[first]private` variables as clauses on the MLIR ops."),
    llvm::cl::init(false));

namespace Fortran {
namespace lower {
namespace omp {

int64_t getCollapseValue(const List<Clause> &clauses) {
  auto iter = llvm::find_if(clauses, [](const Clause &clause) {
    return clause.id == llvm::omp::Clause::OMPC_collapse;
  });
  if (iter != clauses.end()) {
    const auto &collapse = std::get<clause::Collapse>(iter->u);
    return evaluate::ToInt64(collapse.v).value();
  }
  return 1;
}

void genObjectList(const ObjectList &objects,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands) {
  for (const Object &object : objects) {
    const Fortran::semantics::Symbol *sym = object.id();
    assert(sym && "Expected Symbol");
    if (mlir::Value variable = converter.getSymbolAddress(*sym)) {
      operands.push_back(variable);
    } else if (const auto *details =
                   sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      operands.push_back(converter.getSymbolAddress(details->symbol()));
      converter.copySymbolBinding(details->symbol(), *sym);
    }
  }
}

mlir::Type getLoopVarType(Fortran::lower::AbstractConverter &converter,
                          std::size_t loopVarTypeSize) {
  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  loopVarTypeSize = loopVarTypeSize * 8;
  if (loopVarTypeSize < 32) {
    loopVarTypeSize = 32;
  } else if (loopVarTypeSize > 64) {
    loopVarTypeSize = 64;
    mlir::emitWarning(converter.getCurrentLocation(),
                      "OpenMP loop iteration variable cannot have more than 64 "
                      "bits size and will be narrowed into 64 bits.");
  }
  assert((loopVarTypeSize == 32 || loopVarTypeSize == 64) &&
         "OpenMP loop iteration variable size must be transformed into 32-bit "
         "or 64-bit");
  return converter.getFirOpBuilder().getIntegerType(loopVarTypeSize);
}

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  for (const Object &object : objects)
    symbolAndClause.emplace_back(clause, *object.id());
}

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject) {
  Fortran::semantics::Symbol *sym = nullptr;
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto *arrayEle =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator)) {
              sym = GetFirstName(arrayEle->base).symbol;
            } else if (auto *structComp = Fortran::parser::Unwrap<
                           Fortran::parser::StructureComponent>(designator)) {
              sym = structComp->component.symbol;
            } else if (const Fortran::parser::Name *name =
                           Fortran::semantics::getDesignatorNameIfDataRef(
                               designator)) {
              sym = name->symbol;
            }
          },
          [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
      ompObject.u);
  return sym;
}

mlir::omp::MapInfoOp
createMapInfoOp(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                llvm::ArrayRef<mlir::Value> bounds,
                llvm::ArrayRef<mlir::Value> members, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool isVal) {
  if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>()) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name));

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
