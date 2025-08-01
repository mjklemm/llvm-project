//===-- DILAST.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILAST_H
#define LLDB_VALUEOBJECT_DILAST_H

#include "lldb/ValueObject/ValueObject.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>

namespace lldb_private::dil {

/// The various types DIL AST nodes (used by the DIL parser).
enum class NodeKind {
  eArraySubscriptNode,
  eBitExtractionNode,
  eErrorNode,
  eIdentifierNode,
  eMemberOfNode,
  eUnaryOpNode,
};

/// The Unary operators recognized by DIL.
enum class UnaryOpKind {
  AddrOf, // "&"
  Deref,  // "*"
};

/// Forward declaration, for use in DIL AST nodes. Definition is at the very
/// end of this file.
class Visitor;

/// The rest of the classes in this file, except for the Visitor class at the
/// very end, define all the types of AST nodes used by the DIL parser and
/// expression evaluator. The DIL parser parses the input string and creates
/// the AST parse tree from the AST nodes. The resulting AST node tree gets
/// passed to the DIL expression evaluator, which evaluates the DIL AST nodes
/// and creates/returns a ValueObjectSP containing the result.

/// Base class for AST nodes used by the Data Inspection Language (DIL) parser.
/// All of the specialized types of AST nodes inherit from this (virtual) base
/// class.
class ASTNode {
public:
  ASTNode(uint32_t location, NodeKind kind)
      : m_location(location), m_kind(kind) {}
  virtual ~ASTNode() = default;

  virtual llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const = 0;

  uint32_t GetLocation() const { return m_location; }
  NodeKind GetKind() const { return m_kind; }

private:
  uint32_t m_location;
  const NodeKind m_kind;
};

using ASTNodeUP = std::unique_ptr<ASTNode>;

class ErrorNode : public ASTNode {
public:
  ErrorNode() : ASTNode(0, NodeKind::eErrorNode) {}
  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eErrorNode;
  }
};

class IdentifierNode : public ASTNode {
public:
  IdentifierNode(uint32_t location, std::string name)
      : ASTNode(location, NodeKind::eIdentifierNode), m_name(std::move(name)) {}

  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  std::string GetName() const { return m_name; }

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eIdentifierNode;
  }

private:
  std::string m_name;
};

class MemberOfNode : public ASTNode {
public:
  MemberOfNode(uint32_t location, ASTNodeUP base, bool is_arrow,
               std::string name)
      : ASTNode(location, NodeKind::eMemberOfNode), m_base(std::move(base)),
        m_is_arrow(is_arrow), m_field_name(std::move(name)) {}

  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  ASTNode *GetBase() const { return m_base.get(); }
  bool GetIsArrow() const { return m_is_arrow; }
  llvm::StringRef GetFieldName() const { return llvm::StringRef(m_field_name); }

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eMemberOfNode;
  }

private:
  ASTNodeUP m_base;
  bool m_is_arrow;
  std::string m_field_name;
};

class UnaryOpNode : public ASTNode {
public:
  UnaryOpNode(uint32_t location, UnaryOpKind kind, ASTNodeUP operand)
      : ASTNode(location, NodeKind::eUnaryOpNode), m_kind(kind),
        m_operand(std::move(operand)) {}

  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  UnaryOpKind GetKind() const { return m_kind; }
  ASTNode *GetOperand() const { return m_operand.get(); }

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eUnaryOpNode;
  }

private:
  UnaryOpKind m_kind;
  ASTNodeUP m_operand;
};

class ArraySubscriptNode : public ASTNode {
public:
  ArraySubscriptNode(uint32_t location, ASTNodeUP base, int64_t index)
      : ASTNode(location, NodeKind::eArraySubscriptNode),
        m_base(std::move(base)), m_index(index) {}

  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  ASTNode *GetBase() const { return m_base.get(); }
  int64_t GetIndex() const { return m_index; }

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eArraySubscriptNode;
  }

private:
  ASTNodeUP m_base;
  int64_t m_index;
};

class BitFieldExtractionNode : public ASTNode {
public:
  BitFieldExtractionNode(uint32_t location, ASTNodeUP base, int64_t first_index,
                         int64_t last_index)
      : ASTNode(location, NodeKind::eBitExtractionNode),
        m_base(std::move(base)), m_first_index(first_index),
        m_last_index(last_index) {}

  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  ASTNode *GetBase() const { return m_base.get(); }
  int64_t GetFirstIndex() const { return m_first_index; }
  int64_t GetLastIndex() const { return m_last_index; }

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eBitExtractionNode;
  }

private:
  ASTNodeUP m_base;
  int64_t m_first_index;
  int64_t m_last_index;
};

/// This class contains one Visit method for each specialized type of
/// DIL AST node. The Visit methods are used to dispatch a DIL AST node to
/// the correct function in the DIL expression evaluator for evaluating that
/// type of AST node.
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual llvm::Expected<lldb::ValueObjectSP>
  Visit(const IdentifierNode *node) = 0;
  virtual llvm::Expected<lldb::ValueObjectSP>
  Visit(const MemberOfNode *node) = 0;
  virtual llvm::Expected<lldb::ValueObjectSP>
  Visit(const UnaryOpNode *node) = 0;
  virtual llvm::Expected<lldb::ValueObjectSP>
  Visit(const ArraySubscriptNode *node) = 0;
  virtual llvm::Expected<lldb::ValueObjectSP>
  Visit(const BitFieldExtractionNode *node) = 0;
};

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILAST_H
