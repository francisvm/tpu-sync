// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/kv_cache/logical_block_manager.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

struct BlockExtent {
  int start;
  int length;
};

// Selects `count` block IDs from ascending-sorted `sorted_candidates`,
// preferring a single best-fit contiguous extent when available and otherwise
// packing the longest contiguous extents first to minimize DMA fragmentation.
void AppendContiguousPreferredBlocks(absl::Span<const int> sorted_candidates,
                                     int count, std::vector<int>& out) {
  if (count <= 0 || sorted_candidates.empty()) {
    return;
  }

  std::vector<BlockExtent> extents;
  for (int id : sorted_candidates) {
    if (!extents.empty() &&
        id == extents.back().start + extents.back().length) {
      ++extents.back().length;
    } else {
      extents.push_back({.start = id, .length = 1});
    }
  }

  // Prefer the smallest single extent that satisfies `count` (best-fit),
  // breaking ties by lowest start block ID.
  const BlockExtent* best_fit = nullptr;
  for (const BlockExtent& ext : extents) {
    if (ext.length >= count) {
      if (best_fit == nullptr || ext.length < best_fit->length ||
          (ext.length == best_fit->length && ext.start < best_fit->start)) {
        best_fit = &ext;
      }
    }
  }
  if (best_fit != nullptr) {
    for (int i = 0; i < count; ++i) {
      out.push_back(best_fit->start + i);
    }
    return;
  }

  // No single extent fits: take the longest extents first to minimize the
  // number of discontiguous DMA runs.
  std::sort(extents.begin(), extents.end(),
            [](const BlockExtent& a, const BlockExtent& b) {
              if (a.length != b.length) {
                return a.length > b.length;
              }
              return a.start < b.start;
            });

  int remaining = count;
  for (const BlockExtent& ext : extents) {
    if (remaining <= 0) {
      break;
    }
    const int take = std::min(remaining, ext.length);
    for (int i = 0; i < take; ++i) {
      out.push_back(ext.start + i);
    }
    remaining -= take;
  }
}

}  // namespace

LogicalBlockManager::LogicalBlockManager(int num_blocks)
    : total_blocks_(num_blocks), blocks_(num_blocks), num_free_(num_blocks) {}

absl::StatusOr<std::vector<int>> LogicalBlockManager::Allocate(
    int num_blocks_to_allocate, bool lock) {
  if (num_blocks_to_allocate <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Requested blocks must be > 0, got ", num_blocks_to_allocate));
  }
  if (num_blocks_to_allocate > total_blocks_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested blocks (", num_blocks_to_allocate,
                     ") exceed total blocks (", total_blocks_, ")"));
  }

  // Find currently free blocks and unlocked allocated blocks (evictable).
  std::vector<int> free_block_ids;
  std::vector<int> evictable_block_ids;

  for (int i = 0; i < total_blocks_; ++i) {
    if (!blocks_[i].is_allocated) {
      free_block_ids.push_back(i);
    } else if (!blocks_[i].is_locked) {
      evictable_block_ids.push_back(i);
    }
  }

  if (free_block_ids.size() + evictable_block_ids.size() <
      static_cast<size_t>(num_blocks_to_allocate)) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Insufficient available blocks. Requested: ", num_blocks_to_allocate,
        ", Free: ", free_block_ids.size(),
        ", Evictable: ", evictable_block_ids.size()));
  }

  // Determine which blocks to allocate.
  std::vector<int> allocated_block_ids;
  allocated_block_ids.reserve(num_blocks_to_allocate);

  // First, use free blocks, preferring contiguous extents.
  const int free_to_use =
      std::min(static_cast<int>(free_block_ids.size()), num_blocks_to_allocate);
  AppendContiguousPreferredBlocks(free_block_ids, free_to_use,
                                  allocated_block_ids);

  // If we still need more blocks, evict from evictable_block_ids based on LRU
  // policy. Within an equal-timestamp LRU cohort, prefer contiguous extents.
  const int needed_blocks = num_blocks_to_allocate - free_to_use;
  if (needed_blocks > 0) {
    std::sort(evictable_block_ids.begin(), evictable_block_ids.end(),
              [this](int a, int b) {
                if (blocks_[a].last_access_counter ==
                    blocks_[b].last_access_counter) {
                  return a < b;
                }
                return blocks_[a].last_access_counter <
                       blocks_[b].last_access_counter;
              });

    int remaining_evict = needed_blocks;
    size_t idx = 0;
    while (idx < evictable_block_ids.size() && remaining_evict > 0) {
      const uint64_t cohort_ts =
          blocks_[evictable_block_ids[idx]].last_access_counter;
      size_t cohort_end = idx + 1;
      while (cohort_end < evictable_block_ids.size() &&
             blocks_[evictable_block_ids[cohort_end]].last_access_counter ==
                 cohort_ts) {
        ++cohort_end;
      }
      const int cohort_size = static_cast<int>(cohort_end - idx);
      if (cohort_size <= remaining_evict) {
        for (size_t k = idx; k < cohort_end; ++k) {
          const int evict_id = evictable_block_ids[k];
          blocks_[evict_id].is_allocated = false;
          allocated_block_ids.push_back(evict_id);
        }
        remaining_evict -= cohort_size;
      } else {
        std::vector<int> chosen;
        chosen.reserve(remaining_evict);
        AppendContiguousPreferredBlocks(
            absl::MakeConstSpan(evictable_block_ids)
                .subspan(idx, cohort_end - idx),
            remaining_evict, chosen);
        for (int evict_id : chosen) {
          blocks_[evict_id].is_allocated = false;
          allocated_block_ids.push_back(evict_id);
        }
        remaining_evict = 0;
      }
      idx = cohort_end;
    }
  }

  // Return allocated IDs sorted in ascending order so adjacent free + evicted
  // blocks merge into contiguous DMA runs downstream.
  std::sort(allocated_block_ids.begin(), allocated_block_ids.end());

  // Increment global counter for new allocation access timestamp.
  ++access_counter_;

  // Assign the selected blocks to the requested entity.
  for (int block_id : allocated_block_ids) {
    blocks_[block_id].is_allocated = true;
    blocks_[block_id].is_locked = lock;
    blocks_[block_id].last_access_counter = access_counter_;
  }

  num_free_ -= free_to_use;

  return allocated_block_ids;
}

absl::Status LogicalBlockManager::AllocateTarget(
    absl::Span<const int> block_ids) {
  // Validate the entire batch before mutating any state.
  std::vector<bool> seen(total_blocks_, false);
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= total_blocks_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid block ID: ", block_id));
    }
    if (blocks_[block_id].is_allocated) {
      return absl::FailedPreconditionError(
          absl::StrCat("Already allocated block ID: ", block_id));
    }
    if (seen[block_id]) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate block ID: ", block_id));
    }
    seen[block_id] = true;
  }

  ++access_counter_;
  for (int block_id : block_ids) {
    blocks_[block_id].is_allocated = true;
    blocks_[block_id].is_locked = true;
    blocks_[block_id].last_access_counter = access_counter_;
  }
  num_free_ -= static_cast<int>(block_ids.size());
  return absl::OkStatus();
}

absl::Status LogicalBlockManager::Unlock(absl::Span<const int> block_ids) {
  // First validate all blocks.
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= total_blocks_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid block ID: ", block_id));
    }
    if (!blocks_[block_id].is_allocated) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot unlock unallocated block ID: ", block_id));
    }
  }

  // Perform state update.
  for (int block_id : block_ids) {
    blocks_[block_id].is_locked = false;
  }
  return absl::OkStatus();
}

absl::Status LogicalBlockManager::Deallocate(absl::Span<const int> block_ids) {
  // First validate all blocks.
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= total_blocks_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid block ID: ", block_id));
    }
    if (!blocks_[block_id].is_allocated) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot deallocate unallocated block ID: ", block_id));
    }
  }

  // Perform state update.
  for (int block_id : block_ids) {
    if (blocks_[block_id].is_allocated) {
      blocks_[block_id].is_allocated = false;
      ++num_free_;
    }
    blocks_[block_id].is_locked = false;
  }
  return absl::OkStatus();
}

absl::Status LogicalBlockManager::AccessBlock(int block_id) {
  if (block_id < 0 || block_id >= total_blocks_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid block ID: ", block_id));
  }
  if (!blocks_[block_id].is_allocated) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot access unallocated block ID: ", block_id));
  }

  ++access_counter_;
  blocks_[block_id].last_access_counter = access_counter_;
  return absl::OkStatus();
}

bool LogicalBlockManager::IsAllocated(int block_id) const {
  if (block_id < 0 || block_id >= total_blocks_) return false;
  return blocks_[block_id].is_allocated;
}

bool LogicalBlockManager::IsLocked(int block_id) const {
  if (block_id < 0 || block_id >= total_blocks_) return false;
  return blocks_[block_id].is_locked;
}

int LogicalBlockManager::num_free_blocks() const { return num_free_; }

int LogicalBlockManager::num_allocated_blocks() const {
  int count = 0;
  for (const auto& block : blocks_) {
    if (block.is_allocated) ++count;
  }
  return count;
}

int LogicalBlockManager::num_locked_blocks() const {
  int count = 0;
  for (const auto& block : blocks_) {
    if (block.is_locked) ++count;
  }
  return count;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
