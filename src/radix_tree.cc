// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/radix_tree.h"

#include <cstddef>
#include <utility>

namespace mini_llama {

RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;
RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

void RadixTree::Clear() {
  root_.children.clear();
  root_.terminal = false;
  root_.key.clear();
  terminal_count_ = 0;
}

bool RadixTree::empty() const { return terminal_count_ == 0; }

size_t RadixTree::size() const { return terminal_count_; }

size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b) {
  size_t i = 0;
  while (a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]) {
    ++i;
  }
  return i;
}

void RadixTree::Insert(const std::vector<int>& tokens) {
  if (tokens.empty()) {
    if (!root_.terminal) {
      root_.terminal = true;
      ++terminal_count_;
    }
    return;
  }
  InsertInto(&root_, tokens);
}

void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
  auto child_it = parent->children.find(suffix[0]);
  if (child_it == parent->children.end()) {
    auto child = std::make_unique<Node>();
    child->key = std::move(suffix);
    child->terminal = true;
    parent->children[child->key[0]] = std::move(child);
    ++terminal_count_;
    return;
  }

  std::unique_ptr<Node>& child = child_it->second;
  const size_t common = CommonPrefixLength(suffix, 0, child->key);
  if (common == child->key.size()) {
    if (common == suffix.size()) {
      if (!child->terminal) {
        child->terminal = true;
        ++terminal_count_;
      }
      return;
    }
    std::vector<int> remaining(suffix.begin() + static_cast<std::ptrdiff_t>(common),
                               suffix.end());
    InsertInto(child.get(), std::move(remaining));
    return;
  }

  auto split = std::make_unique<Node>();
  split->key.assign(child->key.begin(),
                    child->key.begin() + static_cast<std::ptrdiff_t>(common));

  child->key.erase(child->key.begin(),
                   child->key.begin() + static_cast<std::ptrdiff_t>(common));
  const int old_child_first = child->key[0];
  split->children[old_child_first] = std::move(child);

  if (common == suffix.size()) {
    split->terminal = true;
    ++terminal_count_;
  } else {
    auto new_child = std::make_unique<Node>();
    new_child->key.assign(suffix.begin() + static_cast<std::ptrdiff_t>(common),
                          suffix.end());
    new_child->terminal = true;
    split->children[new_child->key[0]] = std::move(new_child);
    ++terminal_count_;
  }

  child_it->second = std::move(split);
}

size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
  if (tokens.empty()) {
    return 0;
  }

  const Node* node = &root_;
  size_t pos = 0;
  while (pos < tokens.size()) {
    auto child_it = node->children.find(tokens[pos]);
    if (child_it == node->children.end()) {
      return pos;
    }

    const Node* child = child_it->second.get();
    const size_t common = CommonPrefixLength(tokens, pos, child->key);
    pos += common;
    if (common < child->key.size()) {
      return pos;
    }
    node = child;
  }
  return pos;
}

}  // namespace mini_llama
