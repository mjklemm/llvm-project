//===- PassRegistry.cpp - Pass Registration Utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassRegistry.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace detail;

/// Static mapping of all of the registered passes.
static llvm::ManagedStatic<llvm::StringMap<PassInfo>> passRegistry;

/// A mapping of the above pass registry entries to the corresponding TypeID
/// of the pass that they generate.
static llvm::ManagedStatic<llvm::StringMap<TypeID>> passRegistryTypeIDs;

/// Static mapping of all of the registered pass pipelines.
static llvm::ManagedStatic<llvm::StringMap<PassPipelineInfo>>
    passPipelineRegistry;

/// Utility to create a default registry function from a pass instance.
static PassRegistryFunction
buildDefaultRegistryFn(const PassAllocatorFunction &allocator) {
  return [=](OpPassManager &pm, StringRef options,
             function_ref<LogicalResult(const Twine &)> errorHandler) {
    std::unique_ptr<Pass> pass = allocator();
    LogicalResult result = pass->initializeOptions(options, errorHandler);

    std::optional<StringRef> pmOpName = pm.getOpName();
    std::optional<StringRef> passOpName = pass->getOpName();
    if ((pm.getNesting() == OpPassManager::Nesting::Explicit) && pmOpName &&
        passOpName && *pmOpName != *passOpName) {
      return errorHandler(llvm::Twine("Can't add pass '") + pass->getName() +
                          "' restricted to '" + *pass->getOpName() +
                          "' on a PassManager intended to run on '" +
                          pm.getOpAnchorName() + "', did you intend to nest?");
    }
    pm.addPass(std::move(pass));
    return result;
  };
}

/// Utility to print the help string for a specific option.
static void printOptionHelp(StringRef arg, StringRef desc, size_t indent,
                            size_t descIndent, bool isTopLevel) {
  size_t numSpaces = descIndent - indent - 4;
  llvm::outs().indent(indent)
      << "--" << llvm::left_justify(arg, numSpaces) << "-   " << desc << '\n';
}

//===----------------------------------------------------------------------===//
// PassRegistry
//===----------------------------------------------------------------------===//

/// Prints the passes that were previously registered and stored in passRegistry
void mlir::printRegisteredPasses() {
  size_t maxWidth = 0;
  for (auto &entry : *passRegistry)
    maxWidth = std::max(maxWidth, entry.second.getOptionWidth() + 4);

  // Functor used to print the ordered entries of a registration map.
  auto printOrderedEntries = [&](StringRef header, auto &map) {
    llvm::SmallVector<PassRegistryEntry *, 32> orderedEntries;
    for (auto &kv : map)
      orderedEntries.push_back(&kv.second);
    llvm::array_pod_sort(
        orderedEntries.begin(), orderedEntries.end(),
        [](PassRegistryEntry *const *lhs, PassRegistryEntry *const *rhs) {
          return (*lhs)->getPassArgument().compare((*rhs)->getPassArgument());
        });

    llvm::outs().indent(0) << header << ":\n";
    for (PassRegistryEntry *entry : orderedEntries)
      entry->printHelpStr(/*indent=*/2, maxWidth);
  };

  // Print the available passes.
  printOrderedEntries("Passes", *passRegistry);
}

/// Print the help information for this pass. This includes the argument,
/// description, and any pass options. `descIndent` is the indent that the
/// descriptions should be aligned.
void PassRegistryEntry::printHelpStr(size_t indent, size_t descIndent) const {
  printOptionHelp(getPassArgument(), getPassDescription(), indent, descIndent,
                  /*isTopLevel=*/true);
  // If this entry has options, print the help for those as well.
  optHandler([=](const PassOptions &options) {
    options.printHelp(indent, descIndent);
  });
}

/// Return the maximum width required when printing the options of this
/// entry.
size_t PassRegistryEntry::getOptionWidth() const {
  size_t maxLen = 0;
  optHandler([&](const PassOptions &options) mutable {
    maxLen = options.getOptionWidth() + 2;
  });
  return maxLen;
}

//===----------------------------------------------------------------------===//
// PassPipelineInfo
//===----------------------------------------------------------------------===//

void mlir::registerPassPipeline(
    StringRef arg, StringRef description, const PassRegistryFunction &function,
    std::function<void(function_ref<void(const PassOptions &)>)> optHandler) {
  PassPipelineInfo pipelineInfo(arg, description, function,
                                std::move(optHandler));
  bool inserted = passPipelineRegistry->try_emplace(arg, pipelineInfo).second;
#ifndef NDEBUG
  if (!inserted)
    report_fatal_error("Pass pipeline " + arg + " registered multiple times");
#endif
  (void)inserted;
}

//===----------------------------------------------------------------------===//
// PassInfo
//===----------------------------------------------------------------------===//

PassInfo::PassInfo(StringRef arg, StringRef description,
                   const PassAllocatorFunction &allocator)
    : PassRegistryEntry(
          arg, description, buildDefaultRegistryFn(allocator),
          // Use a temporary pass to provide an options instance.
          [=](function_ref<void(const PassOptions &)> optHandler) {
            optHandler(allocator()->passOptions);
          }) {}

void mlir::registerPass(const PassAllocatorFunction &function) {
  std::unique_ptr<Pass> pass = function();
  StringRef arg = pass->getArgument();
  if (arg.empty())
    llvm::report_fatal_error(llvm::Twine("Trying to register '") +
                             pass->getName() +
                             "' pass that does not override `getArgument()`");
  StringRef description = pass->getDescription();
  PassInfo passInfo(arg, description, function);
  passRegistry->try_emplace(arg, passInfo);

  // Verify that the registered pass has the same ID as any registered to this
  // arg before it.
  TypeID entryTypeID = pass->getTypeID();
  auto it = passRegistryTypeIDs->try_emplace(arg, entryTypeID).first;
  if (it->second != entryTypeID)
    llvm::report_fatal_error(
        "pass allocator creates a different pass than previously "
        "registered for pass " +
        arg);
}

/// Returns the pass info for the specified pass argument or null if unknown.
const PassInfo *mlir::PassInfo::lookup(StringRef passArg) {
  auto it = passRegistry->find(passArg);
  return it == passRegistry->end() ? nullptr : &it->second;
}

/// Returns the pass pipeline info for the specified pass pipeline argument or
/// null if unknown.
const PassPipelineInfo *mlir::PassPipelineInfo::lookup(StringRef pipelineArg) {
  auto it = passPipelineRegistry->find(pipelineArg);
  return it == passPipelineRegistry->end() ? nullptr : &it->second;
}

//===----------------------------------------------------------------------===//
// PassOptions
//===----------------------------------------------------------------------===//

/// Attempt to find the next occurance of character 'c' in the string starting
/// from the `index`-th position , omitting any occurances that appear within
/// intervening ranges or literals.
static size_t findChar(StringRef str, size_t index, char c) {
  for (size_t i = index, e = str.size(); i < e; ++i) {
    if (str[i] == c)
      return i;
    // Check for various range characters.
    if (str[i] == '{')
      i = findChar(str, i + 1, '}');
    else if (str[i] == '(')
      i = findChar(str, i + 1, ')');
    else if (str[i] == '[')
      i = findChar(str, i + 1, ']');
    else if (str[i] == '\"')
      i = str.find_first_of('\"', i + 1);
    else if (str[i] == '\'')
      i = str.find_first_of('\'', i + 1);
    if (i == StringRef::npos)
      return StringRef::npos;
  }
  return StringRef::npos;
}

/// Extract an argument from 'options' and update it to point after the arg.
/// Returns the cleaned argument string.
static StringRef extractArgAndUpdateOptions(StringRef &options,
                                            size_t argSize) {
  StringRef str = options.take_front(argSize).trim();
  options = options.drop_front(argSize).ltrim();

  // Early exit if there's no escape sequence.
  if (str.size() <= 1)
    return str;

  const auto escapePairs = {std::make_pair('\'', '\''),
                            std::make_pair('"', '"')};
  for (const auto &escape : escapePairs) {
    if (str.front() == escape.first && str.back() == escape.second) {
      // Drop the escape characters and trim.
      // Don't process additional escape sequences.
      return str.drop_front().drop_back().trim();
    }
  }

  // Arguments may be wrapped in `{...}`. Unlike the quotation markers that
  // denote literals, we respect scoping here. The outer `{...}` should not
  // be stripped in cases such as "arg={...},{...}", which can be used to denote
  // lists of nested option structs.
  if (str.front() == '{') {
    unsigned match = findChar(str, 1, '}');
    if (match == str.size() - 1)
      str = str.drop_front().drop_back().trim();
  }

  return str;
}

LogicalResult detail::pass_options::parseCommaSeparatedList(
    llvm::cl::Option &opt, StringRef argName, StringRef optionStr,
    function_ref<LogicalResult(StringRef)> elementParseFn) {
  if (optionStr.empty())
    return success();

  size_t nextElePos = findChar(optionStr, 0, ',');
  while (nextElePos != StringRef::npos) {
    // Process the portion before the comma.
    if (failed(
            elementParseFn(extractArgAndUpdateOptions(optionStr, nextElePos))))
      return failure();

    // Drop the leading ','
    optionStr = optionStr.drop_front();
    nextElePos = findChar(optionStr, 0, ',');
  }
  return elementParseFn(
      extractArgAndUpdateOptions(optionStr, optionStr.size()));
}

/// Out of line virtual function to provide home for the class.
void detail::PassOptions::OptionBase::anchor() {}

/// Copy the option values from 'other'.
void detail::PassOptions::copyOptionValuesFrom(const PassOptions &other) {
  assert(options.size() == other.options.size());
  if (options.empty())
    return;
  for (auto optionsIt : llvm::zip(options, other.options))
    std::get<0>(optionsIt)->copyValueFrom(*std::get<1>(optionsIt));
}

/// Parse in the next argument from the given options string. Returns a tuple
/// containing [the key of the option, the value of the option, updated
/// `options` string pointing after the parsed option].
static std::tuple<StringRef, StringRef, StringRef>
parseNextArg(StringRef options) {
  // Try to process the given punctuation, properly escaping any contained
  // characters.
  auto tryProcessPunct = [&](size_t &currentPos, char punct) {
    if (options[currentPos] != punct)
      return false;
    size_t nextIt = options.find_first_of(punct, currentPos + 1);
    if (nextIt != StringRef::npos)
      currentPos = nextIt;
    return true;
  };

  // Parse the argument name of the option.
  StringRef argName;
  for (size_t argEndIt = 0, optionsE = options.size();; ++argEndIt) {
    // Check for the end of the full option.
    if (argEndIt == optionsE || options[argEndIt] == ' ') {
      argName = extractArgAndUpdateOptions(options, argEndIt);
      return std::make_tuple(argName, StringRef(), options);
    }

    // Check for the end of the name and the start of the value.
    if (options[argEndIt] == '=') {
      argName = extractArgAndUpdateOptions(options, argEndIt);
      options = options.drop_front();
      break;
    }
  }

  // Parse the value of the option.
  for (size_t argEndIt = 0, optionsE = options.size();; ++argEndIt) {
    // Handle the end of the options string.
    if (argEndIt == optionsE || options[argEndIt] == ' ') {
      StringRef value = extractArgAndUpdateOptions(options, argEndIt);
      return std::make_tuple(argName, value, options);
    }

    // Skip over escaped sequences.
    char c = options[argEndIt];
    if (tryProcessPunct(argEndIt, '\'') || tryProcessPunct(argEndIt, '"'))
      continue;
    // '{...}' is used to specify options to passes, properly escape it so
    // that we don't accidentally split any nested options.
    if (c == '{') {
      size_t braceCount = 1;
      for (++argEndIt; argEndIt != optionsE; ++argEndIt) {
        // Allow nested punctuation.
        if (tryProcessPunct(argEndIt, '\'') || tryProcessPunct(argEndIt, '"'))
          continue;
        if (options[argEndIt] == '{')
          ++braceCount;
        else if (options[argEndIt] == '}' && --braceCount == 0)
          break;
      }
      // Account for the increment at the top of the loop.
      --argEndIt;
    }
  }
  llvm_unreachable("unexpected control flow in pass option parsing");
}

LogicalResult detail::PassOptions::parseFromString(StringRef options,
                                                   raw_ostream &errorStream) {
  // NOTE: `options` is modified in place to always refer to the unprocessed
  // part of the string.
  while (!options.empty()) {
    StringRef key, value;
    std::tie(key, value, options) = parseNextArg(options);
    if (key.empty())
      continue;

    auto it = OptionsMap.find(key);
    if (it == OptionsMap.end()) {
      errorStream << "<Pass-Options-Parser>: no such option " << key << "\n";
      return failure();
    }
    if (llvm::cl::ProvidePositionalOption(it->second, value, 0))
      return failure();
  }

  return success();
}

/// Print the options held by this struct in a form that can be parsed via
/// 'parseFromString'.
void detail::PassOptions::print(raw_ostream &os) const {
  // If there are no options, there is nothing left to do.
  if (OptionsMap.empty())
    return;

  // Sort the options to make the ordering deterministic.
  SmallVector<OptionBase *, 4> orderedOps(options.begin(), options.end());
  auto compareOptionArgs = [](OptionBase *const *lhs, OptionBase *const *rhs) {
    return (*lhs)->getArgStr().compare((*rhs)->getArgStr());
  };
  llvm::array_pod_sort(orderedOps.begin(), orderedOps.end(), compareOptionArgs);

  // Interleave the options with ' '.
  os << '{';
  llvm::interleave(
      orderedOps, os, [&](OptionBase *option) { option->print(os); }, " ");
  os << '}';
}

/// Print the help string for the options held by this struct. `descIndent` is
/// the indent within the stream that the descriptions should be aligned.
void detail::PassOptions::printHelp(size_t indent, size_t descIndent) const {
  // Sort the options to make the ordering deterministic.
  SmallVector<OptionBase *, 4> orderedOps(options.begin(), options.end());
  auto compareOptionArgs = [](OptionBase *const *lhs, OptionBase *const *rhs) {
    return (*lhs)->getArgStr().compare((*rhs)->getArgStr());
  };
  llvm::array_pod_sort(orderedOps.begin(), orderedOps.end(), compareOptionArgs);
  for (OptionBase *option : orderedOps) {
    // TODO: printOptionInfo assumes a specific indent and will
    // print options with values with incorrect indentation. We should add
    // support to llvm::cl::Option for passing in a base indent to use when
    // printing.
    llvm::outs().indent(indent);
    option->getOption()->printOptionInfo(descIndent - indent);
  }
}

/// Return the maximum width required when printing the help string.
size_t detail::PassOptions::getOptionWidth() const {
  size_t max = 0;
  for (auto *option : options)
    max = std::max(max, option->getOption()->getOptionWidth());
  return max;
}

//===----------------------------------------------------------------------===//
// MLIR Options
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OpPassManager: OptionValue
//===----------------------------------------------------------------------===//

llvm::cl::OptionValue<OpPassManager>::OptionValue() = default;
llvm::cl::OptionValue<OpPassManager>::OptionValue(
    const mlir::OpPassManager &value) {
  setValue(value);
}
llvm::cl::OptionValue<OpPassManager>::OptionValue(
    const llvm::cl::OptionValue<mlir::OpPassManager> &rhs) {
  if (rhs.hasValue())
    setValue(rhs.getValue());
}
llvm::cl::OptionValue<OpPassManager> &
llvm::cl::OptionValue<OpPassManager>::operator=(
    const mlir::OpPassManager &rhs) {
  setValue(rhs);
  return *this;
}

llvm::cl::OptionValue<OpPassManager>::~OptionValue<OpPassManager>() = default;

void llvm::cl::OptionValue<OpPassManager>::setValue(
    const OpPassManager &newValue) {
  if (hasValue())
    *value = newValue;
  else
    value = std::make_unique<mlir::OpPassManager>(newValue);
}
void llvm::cl::OptionValue<OpPassManager>::setValue(StringRef pipelineStr) {
  FailureOr<OpPassManager> pipeline = parsePassPipeline(pipelineStr);
  assert(succeeded(pipeline) && "invalid pass pipeline");
  setValue(*pipeline);
}

bool llvm::cl::OptionValue<OpPassManager>::compare(
    const mlir::OpPassManager &rhs) const {
  std::string lhsStr, rhsStr;
  {
    raw_string_ostream lhsStream(lhsStr);
    value->printAsTextualPipeline(lhsStream);

    raw_string_ostream rhsStream(rhsStr);
    rhs.printAsTextualPipeline(rhsStream);
  }

  // Use the textual format for pipeline comparisons.
  return lhsStr == rhsStr;
}

void llvm::cl::OptionValue<OpPassManager>::anchor() {}

//===----------------------------------------------------------------------===//
// OpPassManager: Parser
//===----------------------------------------------------------------------===//

namespace llvm {
namespace cl {
template class basic_parser<OpPassManager>;
} // namespace cl
} // namespace llvm

bool llvm::cl::parser<OpPassManager>::parse(Option &, StringRef, StringRef arg,
                                            ParsedPassManager &value) {
  FailureOr<OpPassManager> pipeline = parsePassPipeline(arg);
  if (failed(pipeline))
    return true;
  value.value = std::make_unique<OpPassManager>(std::move(*pipeline));
  return false;
}

void llvm::cl::parser<OpPassManager>::print(raw_ostream &os,
                                            const OpPassManager &value) {
  value.printAsTextualPipeline(os);
}

void llvm::cl::parser<OpPassManager>::printOptionDiff(
    const Option &opt, OpPassManager &pm, const OptVal &defaultValue,
    size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= ";
  pm.printAsTextualPipeline(outs());

  if (defaultValue.hasValue()) {
    outs().indent(2) << " (default: ";
    defaultValue.getValue().printAsTextualPipeline(outs());
    outs() << ")";
  }
  outs() << "\n";
}

void llvm::cl::parser<OpPassManager>::anchor() {}

llvm::cl::parser<OpPassManager>::ParsedPassManager::ParsedPassManager() =
    default;
llvm::cl::parser<OpPassManager>::ParsedPassManager::ParsedPassManager(
    ParsedPassManager &&) = default;
llvm::cl::parser<OpPassManager>::ParsedPassManager::~ParsedPassManager() =
    default;

//===----------------------------------------------------------------------===//
// TextualPassPipeline Parser
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a textual description of a pass pipeline.
class TextualPipeline {
public:
  /// Try to initialize this pipeline with the given pipeline text.
  /// `errorStream` is the output stream to emit errors to.
  LogicalResult initialize(StringRef text, raw_ostream &errorStream);

  /// Add the internal pipeline elements to the provided pass manager.
  LogicalResult
  addToPipeline(OpPassManager &pm,
                function_ref<LogicalResult(const Twine &)> errorHandler) const;

private:
  /// A functor used to emit errors found during pipeline handling. The first
  /// parameter corresponds to the raw location within the pipeline string. This
  /// should always return failure.
  using ErrorHandlerT = function_ref<LogicalResult(const char *, Twine)>;

  /// A struct to capture parsed pass pipeline names.
  ///
  /// A pipeline is defined as a series of names, each of which may in itself
  /// recursively contain a nested pipeline. A name is either the name of a pass
  /// (e.g. "cse") or the name of an operation type (e.g. "buitin.module"). If
  /// the name is the name of a pass, the InnerPipeline is empty, since passes
  /// cannot contain inner pipelines.
  struct PipelineElement {
    PipelineElement(StringRef name) : name(name) {}

    StringRef name;
    StringRef options;
    const PassRegistryEntry *registryEntry = nullptr;
    std::vector<PipelineElement> innerPipeline;
  };

  /// Parse the given pipeline text into the internal pipeline vector. This
  /// function only parses the structure of the pipeline, and does not resolve
  /// its elements.
  LogicalResult parsePipelineText(StringRef text, ErrorHandlerT errorHandler);

  /// Resolve the elements of the pipeline, i.e. connect passes and pipelines to
  /// the corresponding registry entry.
  LogicalResult
  resolvePipelineElements(MutableArrayRef<PipelineElement> elements,
                          ErrorHandlerT errorHandler);

  /// Resolve a single element of the pipeline.
  LogicalResult resolvePipelineElement(PipelineElement &element,
                                       ErrorHandlerT errorHandler);

  /// Add the given pipeline elements to the provided pass manager.
  LogicalResult
  addToPipeline(ArrayRef<PipelineElement> elements, OpPassManager &pm,
                function_ref<LogicalResult(const Twine &)> errorHandler) const;

  std::vector<PipelineElement> pipeline;
};

} // namespace

/// Try to initialize this pipeline with the given pipeline text. An option is
/// given to enable accurate error reporting.
LogicalResult TextualPipeline::initialize(StringRef text,
                                          raw_ostream &errorStream) {
  if (text.empty())
    return success();

  // Build a source manager to use for error reporting.
  llvm::SourceMgr pipelineMgr;
  pipelineMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(text, "MLIR Textual PassPipeline Parser",
                                       /*RequiresNullTerminator=*/false),
      SMLoc());
  auto errorHandler = [&](const char *rawLoc, Twine msg) {
    pipelineMgr.PrintMessage(errorStream, SMLoc::getFromPointer(rawLoc),
                             llvm::SourceMgr::DK_Error, msg);
    return failure();
  };

  // Parse the provided pipeline string.
  if (failed(parsePipelineText(text, errorHandler)))
    return failure();
  return resolvePipelineElements(pipeline, errorHandler);
}

/// Add the internal pipeline elements to the provided pass manager.
LogicalResult TextualPipeline::addToPipeline(
    OpPassManager &pm,
    function_ref<LogicalResult(const Twine &)> errorHandler) const {
  // Temporarily disable implicit nesting while we append to the pipeline. We
  // want the created pipeline to exactly match the parsed text pipeline, so
  // it's preferrable to just error out if implicit nesting would be required.
  OpPassManager::Nesting nesting = pm.getNesting();
  pm.setNesting(OpPassManager::Nesting::Explicit);
  auto restore = llvm::make_scope_exit([&]() { pm.setNesting(nesting); });

  return addToPipeline(pipeline, pm, errorHandler);
}

/// Parse the given pipeline text into the internal pipeline vector. This
/// function only parses the structure of the pipeline, and does not resolve
/// its elements.
LogicalResult TextualPipeline::parsePipelineText(StringRef text,
                                                 ErrorHandlerT errorHandler) {
  SmallVector<std::vector<PipelineElement> *, 4> pipelineStack = {&pipeline};
  for (;;) {
    std::vector<PipelineElement> &pipeline = *pipelineStack.back();
    size_t pos = text.find_first_of(",(){");
    pipeline.emplace_back(/*name=*/text.substr(0, pos).trim());

    // If we have a single terminating name, we're done.
    if (pos == StringRef::npos)
      break;

    text = text.substr(pos);
    char sep = text[0];

    // Handle pulling ... from 'pass{...}' out as PipelineElement.options.
    if (sep == '{') {
      text = text.substr(1);

      // Skip over everything until the closing '}' and store as options.
      size_t close = StringRef::npos;
      for (unsigned i = 0, e = text.size(), braceCount = 1; i < e; ++i) {
        if (text[i] == '{') {
          ++braceCount;
          continue;
        }
        if (text[i] == '}' && --braceCount == 0) {
          close = i;
          break;
        }
      }

      // Check to see if a closing options brace was found.
      if (close == StringRef::npos) {
        return errorHandler(
            /*rawLoc=*/text.data() - 1,
            "missing closing '}' while processing pass options");
      }
      pipeline.back().options = text.substr(0, close);
      text = text.substr(close + 1);

      // Consume space characters that an user might add for readability.
      text = text.ltrim();

      // Skip checking for '(' because nested pipelines cannot have options.
    } else if (sep == '(') {
      text = text.substr(1);

      // Push the inner pipeline onto the stack to continue processing.
      pipelineStack.push_back(&pipeline.back().innerPipeline);
      continue;
    }

    // When handling the close parenthesis, we greedily consume them to avoid
    // empty strings in the pipeline.
    while (text.consume_front(")")) {
      // If we try to pop the outer pipeline we have unbalanced parentheses.
      if (pipelineStack.size() == 1)
        return errorHandler(/*rawLoc=*/text.data() - 1,
                            "encountered extra closing ')' creating unbalanced "
                            "parentheses while parsing pipeline");

      pipelineStack.pop_back();
      // Consume space characters that an user might add for readability.
      text = text.ltrim();
    }

    // Check if we've finished parsing.
    if (text.empty())
      break;

    // Otherwise, the end of an inner pipeline always has to be followed by
    // a comma, and then we can continue.
    if (!text.consume_front(","))
      return errorHandler(text.data(), "expected ',' after parsing pipeline");
  }

  // Check for unbalanced parentheses.
  if (pipelineStack.size() > 1)
    return errorHandler(
        text.data(),
        "encountered unbalanced parentheses while parsing pipeline");

  assert(pipelineStack.back() == &pipeline &&
         "wrong pipeline at the bottom of the stack");
  return success();
}

/// Resolve the elements of the pipeline, i.e. connect passes and pipelines to
/// the corresponding registry entry.
LogicalResult TextualPipeline::resolvePipelineElements(
    MutableArrayRef<PipelineElement> elements, ErrorHandlerT errorHandler) {
  for (auto &elt : elements)
    if (failed(resolvePipelineElement(elt, errorHandler)))
      return failure();
  return success();
}

/// Resolve a single element of the pipeline.
LogicalResult
TextualPipeline::resolvePipelineElement(PipelineElement &element,
                                        ErrorHandlerT errorHandler) {
  // If the inner pipeline of this element is not empty, this is an operation
  // pipeline.
  if (!element.innerPipeline.empty())
    return resolvePipelineElements(element.innerPipeline, errorHandler);

  // Otherwise, this must be a pass or pass pipeline.
  // Check to see if a pipeline was registered with this name.
  if ((element.registryEntry = PassPipelineInfo::lookup(element.name)))
    return success();

  // If not, then this must be a specific pass name.
  if ((element.registryEntry = PassInfo::lookup(element.name)))
    return success();

  // Emit an error for the unknown pass.
  auto *rawLoc = element.name.data();
  return errorHandler(rawLoc, "'" + element.name +
                                  "' does not refer to a "
                                  "registered pass or pass pipeline");
}

/// Add the given pipeline elements to the provided pass manager.
LogicalResult TextualPipeline::addToPipeline(
    ArrayRef<PipelineElement> elements, OpPassManager &pm,
    function_ref<LogicalResult(const Twine &)> errorHandler) const {
  for (auto &elt : elements) {
    if (elt.registryEntry) {
      if (failed(elt.registryEntry->addToPipeline(pm, elt.options,
                                                  errorHandler))) {
        return errorHandler("failed to add `" + elt.name + "` with options `" +
                            elt.options + "`");
      }
    } else if (failed(addToPipeline(elt.innerPipeline, pm.nest(elt.name),
                                    errorHandler))) {
      return errorHandler("failed to add `" + elt.name + "` with options `" +
                          elt.options + "` to inner pipeline");
    }
  }
  return success();
}

LogicalResult mlir::parsePassPipeline(StringRef pipeline, OpPassManager &pm,
                                      raw_ostream &errorStream) {
  TextualPipeline pipelineParser;
  if (failed(pipelineParser.initialize(pipeline, errorStream)))
    return failure();
  auto errorHandler = [&](Twine msg) {
    errorStream << msg << "\n";
    return failure();
  };
  if (failed(pipelineParser.addToPipeline(pm, errorHandler)))
    return failure();
  return success();
}

FailureOr<OpPassManager> mlir::parsePassPipeline(StringRef pipeline,
                                                 raw_ostream &errorStream) {
  pipeline = pipeline.trim();
  // Pipelines are expected to be of the form `<op-name>(<pipeline>)`.
  size_t pipelineStart = pipeline.find_first_of('(');
  if (pipelineStart == 0 || pipelineStart == StringRef::npos ||
      !pipeline.consume_back(")")) {
    errorStream << "expected pass pipeline to be wrapped with the anchor "
                   "operation type, e.g. 'builtin.module(...)'";
    return failure();
  }

  StringRef opName = pipeline.take_front(pipelineStart).rtrim();
  OpPassManager pm(opName);
  if (failed(parsePassPipeline(pipeline.drop_front(1 + pipelineStart), pm,
                               errorStream)))
    return failure();
  return pm;
}

//===----------------------------------------------------------------------===//
// PassNameParser
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents the possible data entries in a parsed pass pipeline
/// list.
struct PassArgData {
  PassArgData() = default;
  PassArgData(const PassRegistryEntry *registryEntry)
      : registryEntry(registryEntry) {}

  /// This field is used when the parsed option corresponds to a registered pass
  /// or pass pipeline.
  const PassRegistryEntry *registryEntry{nullptr};

  /// This field is set when instance specific pass options have been provided
  /// on the command line.
  StringRef options;
};
} // namespace

namespace llvm {
namespace cl {
/// Define a valid OptionValue for the command line pass argument.
template <>
struct OptionValue<PassArgData> final
    : OptionValueBase<PassArgData, /*isClass=*/true> {
  OptionValue(const PassArgData &value) { this->setValue(value); }
  OptionValue() = default;
  void anchor() override {}

  bool hasValue() const { return true; }
  const PassArgData &getValue() const { return value; }
  void setValue(const PassArgData &value) { this->value = value; }

  PassArgData value;
};
} // namespace cl
} // namespace llvm

namespace {

/// The name for the command line option used for parsing the textual pass
/// pipeline.
#define PASS_PIPELINE_ARG "pass-pipeline"

/// Adds command line option for each registered pass or pass pipeline, as well
/// as textual pass pipelines.
struct PassNameParser : public llvm::cl::parser<PassArgData> {
  PassNameParser(llvm::cl::Option &opt) : llvm::cl::parser<PassArgData>(opt) {}

  void initialize();
  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const override;
  size_t getOptionWidth(const llvm::cl::Option &opt) const override;
  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             PassArgData &value);

  /// If true, this parser only parses entries that correspond to a concrete
  /// pass registry entry, and does not include pipeline entries or the options
  /// for pass entries.
  bool passNamesOnly = false;
};
} // namespace

void PassNameParser::initialize() {
  llvm::cl::parser<PassArgData>::initialize();

  /// Add the pass entries.
  for (const auto &kv : *passRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
  /// Add the pass pipeline entries.
  if (!passNamesOnly) {
    for (const auto &kv : *passPipelineRegistry) {
      addLiteralOption(kv.second.getPassArgument(), &kv.second,
                       kv.second.getPassDescription());
    }
  }
}

void PassNameParser::printOptionInfo(const llvm::cl::Option &opt,
                                     size_t globalWidth) const {
  // If this parser is just parsing pass names, print a simplified option
  // string.
  if (passNamesOnly) {
    llvm::outs() << "  --" << opt.ArgStr << "=<pass-arg>";
    opt.printHelpStr(opt.HelpStr, globalWidth, opt.ArgStr.size() + 18);
    return;
  }

  // Print the information for the top-level option.
  if (opt.hasArgStr()) {
    llvm::outs() << "  --" << opt.ArgStr;
    opt.printHelpStr(opt.HelpStr, globalWidth, opt.ArgStr.size() + 7);
  } else {
    llvm::outs() << "  " << opt.HelpStr << '\n';
  }

  // Functor used to print the ordered entries of a registration map.
  auto printOrderedEntries = [&](StringRef header, auto &map) {
    llvm::SmallVector<PassRegistryEntry *, 32> orderedEntries;
    for (auto &kv : map)
      orderedEntries.push_back(&kv.second);
    llvm::array_pod_sort(
        orderedEntries.begin(), orderedEntries.end(),
        [](PassRegistryEntry *const *lhs, PassRegistryEntry *const *rhs) {
          return (*lhs)->getPassArgument().compare((*rhs)->getPassArgument());
        });

    llvm::outs().indent(4) << header << ":\n";
    for (PassRegistryEntry *entry : orderedEntries)
      entry->printHelpStr(/*indent=*/6, globalWidth);
  };

  // Print the available passes.
  printOrderedEntries("Passes", *passRegistry);

  // Print the available pass pipelines.
  if (!passPipelineRegistry->empty())
    printOrderedEntries("Pass Pipelines", *passPipelineRegistry);
}

size_t PassNameParser::getOptionWidth(const llvm::cl::Option &opt) const {
  size_t maxWidth = llvm::cl::parser<PassArgData>::getOptionWidth(opt) + 2;

  // Check for any wider pass or pipeline options.
  for (auto &entry : *passRegistry)
    maxWidth = std::max(maxWidth, entry.second.getOptionWidth() + 4);
  for (auto &entry : *passPipelineRegistry)
    maxWidth = std::max(maxWidth, entry.second.getOptionWidth() + 4);
  return maxWidth;
}

bool PassNameParser::parse(llvm::cl::Option &opt, StringRef argName,
                           StringRef arg, PassArgData &value) {
  if (llvm::cl::parser<PassArgData>::parse(opt, argName, arg, value))
    return true;
  value.options = arg;
  return false;
}

//===----------------------------------------------------------------------===//
// PassPipelineCLParser
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct PassPipelineCLParserImpl {
  PassPipelineCLParserImpl(StringRef arg, StringRef description,
                           bool passNamesOnly)
      : passList(arg, llvm::cl::desc(description)) {
    passList.getParser().passNamesOnly = passNamesOnly;
    passList.setValueExpectedFlag(llvm::cl::ValueExpected::ValueOptional);
  }

  /// Returns true if the given pass registry entry was registered at the
  /// top-level of the parser, i.e. not within an explicit textual pipeline.
  bool contains(const PassRegistryEntry *entry) const {
    return llvm::any_of(passList, [&](const PassArgData &data) {
      return data.registryEntry == entry;
    });
  }

  /// The set of passes and pass pipelines to run.
  llvm::cl::list<PassArgData, bool, PassNameParser> passList;
};
} // namespace detail
} // namespace mlir

/// Construct a pass pipeline parser with the given command line description.
PassPipelineCLParser::PassPipelineCLParser(StringRef arg, StringRef description)
    : impl(std::make_unique<detail::PassPipelineCLParserImpl>(
          arg, description, /*passNamesOnly=*/false)),
      passPipeline(
          PASS_PIPELINE_ARG,
          llvm::cl::desc("Textual description of the pass pipeline to run")) {}

PassPipelineCLParser::PassPipelineCLParser(StringRef arg, StringRef description,
                                           StringRef alias)
    : PassPipelineCLParser(arg, description) {
  passPipelineAlias.emplace(alias,
                            llvm::cl::desc("Alias for --" PASS_PIPELINE_ARG),
                            llvm::cl::aliasopt(passPipeline));
}

PassPipelineCLParser::~PassPipelineCLParser() = default;

/// Returns true if this parser contains any valid options to add.
bool PassPipelineCLParser::hasAnyOccurrences() const {
  return passPipeline.getNumOccurrences() != 0 ||
         impl->passList.getNumOccurrences() != 0;
}

/// Returns true if the given pass registry entry was registered at the
/// top-level of the parser, i.e. not within an explicit textual pipeline.
bool PassPipelineCLParser::contains(const PassRegistryEntry *entry) const {
  return impl->contains(entry);
}

/// Adds the passes defined by this parser entry to the given pass manager.
LogicalResult PassPipelineCLParser::addToPipeline(
    OpPassManager &pm,
    function_ref<LogicalResult(const Twine &)> errorHandler) const {
  if (passPipeline.getNumOccurrences()) {
    if (impl->passList.getNumOccurrences())
      return errorHandler(
          "'-" PASS_PIPELINE_ARG
          "' option can't be used with individual pass options");
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    FailureOr<OpPassManager> parsed = parsePassPipeline(passPipeline, os);
    if (failed(parsed))
      return errorHandler(errMsg);
    pm = std::move(*parsed);
    return success();
  }

  for (auto &passIt : impl->passList) {
    if (failed(passIt.registryEntry->addToPipeline(pm, passIt.options,
                                                   errorHandler)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PassNameCLParser
//===----------------------------------------------------------------------===//

/// Construct a pass pipeline parser with the given command line description.
PassNameCLParser::PassNameCLParser(StringRef arg, StringRef description)
    : impl(std::make_unique<detail::PassPipelineCLParserImpl>(
          arg, description, /*passNamesOnly=*/true)) {
  impl->passList.setMiscFlag(llvm::cl::CommaSeparated);
}
PassNameCLParser::~PassNameCLParser() = default;

/// Returns true if this parser contains any valid options to add.
bool PassNameCLParser::hasAnyOccurrences() const {
  return impl->passList.getNumOccurrences() != 0;
}

/// Returns true if the given pass registry entry was registered at the
/// top-level of the parser, i.e. not within an explicit textual pipeline.
bool PassNameCLParser::contains(const PassRegistryEntry *entry) const {
  return impl->contains(entry);
}
