/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/transforms/collage_fuse_ops.cc
 * \brief Search for optimal fused sub-graphs and targets.
 */

#include "./collage_fuse_ops.h"

#include <math.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/transform.h>
#include <tvm/target/target.h>

#include "../collage/cost.h"
#include "../ir/dataflow_matcher_impl.h"
#include "candidate_kernel.h"
#include "cost_estimator.h"
#include "fusion_rule.h"
#include "name_supply.h"
#include "sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {
namespace {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.collage.enable_collage", Bool);

/*!
 * \brief Represents the partial placement of an expression after some number of candidate kernels
 * have been extracted.
 */
class SearchState {
 public:
  explicit SearchState(IndexSet covered) : covered_(std::move(covered)) {}

  /*!
   * \brief Order states by increasing best cost, breaking ties by lexicographic order on
   * the covering sub graph.
   */
  bool operator<(const SearchState& that) const {
    return std::tie(best_cost_, covered_) < std::tie(that.best_cost_, that.covered_);
  }

  const IndexSet& covered() const { return covered_; }

 private:
  /*! \brief Which nodes of overall expression have been placed on all paths to this state. */
  IndexSet covered_;
  /*! \brief Predecessor state for sequence of candidate kernels reaching this state with least
   * cost. Null if initial search state. */
  SearchState* pred_state_ = nullptr;
  /*!
   * \brief Cost of reaching this state using placement implied by path given by pred_state fields.
   * Includes estimated/measured cost of all candidate kernels plus any kernel launch penalty.
   * Initially invalid cost.
   */
  Cost best_cost_ = Cost::Invalid();
  /*! \brief Candidate kernel selected in transition from pred_state to this state. */
  CandidateKernel best_candidate_kernel_;

  friend class Partitioner;
};

struct CompareSearchStatePtrs {
  bool operator()(const SearchState* left, const SearchState* right) const {
    return *left < *right;
  }
};

/*! \brief Priority queue of search states, ordered by increasing cost. */
class PriorityQueue {
 public:
  PriorityQueue() = default;

  /*! \brief Pushes \p state onto the queue. */
  void Push(SearchState* state) { set_.emplace(state); }

  /*! \brief Pops the state with the least ocst off the queue. */
  SearchState* Pop() {
    ICHECK(!set_.empty());
    SearchState* state = *set_.begin();
    set_.erase(set_.begin());
    return state;
  }

  /*! \brief Updates the queue to account for \p state's best cost being lowered. */
  void Update(SearchState* state) {
    auto itr = std::find_if(set_.begin(), set_.end(), [&state](const SearchState* s) {
      return s->covered() == state->covered();
    });
    ICHECK(itr != set_.end());
    set_.erase(itr);
    set_.emplace(state);
  }

  bool empty() const { return set_.empty(); }
  size_t size() const { return set_.size(); }

 private:
  // TODO(mbs): Use an updatable priority queue.
  std::set<SearchState*, CompareSearchStatePtrs> set_;
};

/*!
 * \brief Collects and indexes all the candidate kernels for the overall expression. This index is
 * used during partitioning search to find the next valid candidate kernels to explore from the
 * current search state. We do not yet attempt to estimate the cost of each candidate kernel, and
 * when we do so during the search we may discover it to be infeasible.
 */
class CandidateKernelIndex {
 public:
  explicit CandidateKernelIndex(const DataflowGraph* dataflow_graph)
      : dataflow_graph_(dataflow_graph),
        first_inside_index_to_candidates_(dataflow_graph->size()) {}

  /*! \brief Constructs the index. */
  void Index(const Expr& expr, const std::vector<FusionSpec>& fusion_specs,
             NameSupply* name_supply) {
    for (const auto& fusion_spec : fusion_specs) {
      VLOG_CONTEXT << "spec " << fusion_spec->spec_name_;
      Array<CandidateKernel> candidates =
          fusion_spec->AllCandidateKernels(*dataflow_graph_, name_supply);
      // For each candidate...
      for (auto& candidate : candidates) {
        // We now own the candidate
        all_candidate_kernels_.push_back(candidate);
        PostDfsIndex first_inside_index = candidate->sub_graph_->first_inside_index_;
        ICHECK_LT(first_inside_index, dataflow_graph_->size());
        VLOG(1) << "Indexing candidate kernel " << candidate->ToString() << " on index "
                << first_inside_index;
        first_inside_index_to_candidates_[first_inside_index].push_back(candidate);
      }
    }
  }

  /*! \brief Returns all the candidate kernels which may begin at \p index. */
  const std::vector<CandidateKernel>& candidate_kernels_at(PostDfsIndex index) const {
    ICHECK_LT(index, dataflow_graph_->size());
    return first_inside_index_to_candidates_[index];
  }

  size_t size() const { return all_candidate_kernels_.size(); }

 private:
  /*! \brief Dataflow graph for overall expression. */
  const DataflowGraph* dataflow_graph_;
  /*!
   * \brief All available candidate kernels, across all sub-expressions, targets and fusion rules.
   */
  Array<CandidateKernel> all_candidate_kernels_;
  /*!
   * \brief Maps indexes to the all the candidate kernels which have that as their first inside
   * index.
   */
  std::vector<std::vector<CandidateKernel>> first_inside_index_to_candidates_;
};

/*!
 * \brief Finds the optimal partitioning of an expression to candidate kernels.
 * Though no candidate kernels overlap, it is possible some sub-expressions end up in
 * no candidate kernels. Those sub-expressions must be evaluated by the host executor (eg VM).
 */
class Partitioner {
 public:
  explicit Partitioner(CompilationConfig config) : config_(std::move(config)) {}

  Expr Partition(const Expr& expr) {
    VLOG_CONTEXT << "Partitioning";
    std::vector<FusionSpec> fusion_specs = GatherFusionSpecs();
    VLOG(1) << "Gathered " << fusion_specs.size() << " fusion specs";
    std::unique_ptr<DataflowGraph> dataflow_graph = CreateIndexedGraph(expr);
    VLOG(1) << "Creating dataflow graph with " << dataflow_graph->size() << " nodes";
    // Index all the candidates (which will be owned by the index).
    CandidateKernelIndex index(dataflow_graph.get());
    NameSupply name_supply("collage");
    index.Index(expr, fusion_specs, &name_supply);
    VLOG(1) << "Indexed " << index.size() << " candidate kernels";
    VLOG(1) << "Commencing search";
    pq_.Push(GetState(IndexSet(dataflow_graph->size())));
    while (!pq_.empty()) {
      SearchState* curr_state = pq_.Pop();
      VLOG(1) << "Looking at state " << curr_state->covered_.ToString();
      PostDfsIndex next_index = curr_state->covered_.FirstOutsideIndex();
      if (next_index >= dataflow_graph->size()) {
        VLOG(1) << "Finished search, recovering least-cost candidates";
        // The entire expression has been explored. Apply the best kernels walking backwards
        // from the final state.
        Expr partitioned = expr;
        while (curr_state != nullptr) {
          if (curr_state->best_candidate_kernel_.defined()) {
            VLOG(1) << "Including best candidate "
                    << curr_state->best_candidate_kernel_->ToString();
            partitioned =
                curr_state->best_candidate_kernel_->Partition(*dataflow_graph, partitioned);
          }
          curr_state = curr_state->pred_state_;
        }
        return partitioned;
      }
      size_t num_fires = 0;
      VLOG(1) << "Looking at index " << next_index;
      for (const auto& candidate_kernel : index.candidate_kernels_at(next_index)) {
        VLOG(1) << "Considering candidate kernel " << candidate_kernel->ToString();
        if (!candidate_kernel->sub_graph_->inside_.AreDisjoint(curr_state->covered_)) {
          VLOG(1) << "Kernel overlaps with already fused nodes";
          continue;
        }
        IndexSet next_covered = curr_state->covered_ | candidate_kernel->sub_graph_->inside_;
        SearchState* next_state = GetState(next_covered);
        Relax(curr_state, next_state, candidate_kernel);
        ++num_fires;
      }
      if (num_fires == 0) {
        VLOG(1) << "No candidates, skipping node";
        IndexSet next_covered =
            curr_state->covered_ | IndexSet(dataflow_graph->size(), {next_index});
        SearchState* next_state = GetState(next_covered);
        Relax(curr_state, next_state, /*candidate_kernel=*/{});
      }
    }
    ICHECK(false) << "should have reached state in which all sub-expressions are covered";
    return {};
  }

  /*!
   * \brief Returns all the fusion specifications gathered from all targets in the compilation
   * config.
   */
  std::vector<FusionSpec> GatherFusionSpecs() {
    std::vector<FusionSpec> result;
    for (const auto& target : config_->primitive_targets) {
      // We implement this on the python side only so that we can get access to the BYOC
      // pattern registry. Otherwise for the most part this implementation just bounces right
      // back into the construction helpers in fusion_rule.cc.
      static const runtime::PackedFunc* make_fusion_spec =
          runtime::Registry::Get("tvm.relay.collage.make_fusion_spec");
      ICHECK(make_fusion_spec);
      FusionSpec spec = (*make_fusion_spec)(target);
      VLOG(1) << "using spec:\n" << spec->ToString();
      result.emplace_back(std::move(spec));
    }
    return result;
  }

  /*! \brief Returns the unique state corresponding to the \p covered sub-graph. */
  SearchState* GetState(const IndexSet& covered) {
    auto itr = covered_to_state_.find(covered);
    if (itr != covered_to_state_.end()) {
      return itr->second.get();
    }
    auto state = std::make_unique<SearchState>(covered);
    SearchState* raw_ptr = state.get();
    covered_to_state_.emplace(covered, std::move(state));
    return raw_ptr;
  }

  /*!
   * \brief Record that it is possible to \p next_state by applying \p candidate_kernel
   * to \p curr_state. If the resulting cost is better than the best known
   * so far, update this state's best costs, predecessor and kernel to match.
   */
  void Relax(SearchState* curr_state, SearchState* next_state, CandidateKernel candidate) {
    Cost candidate_cost;
    if (candidate.defined()) {
      candidate_cost = candidate->EstimatedCost(&cost_estimator_);
    }
    Cost new_state_cost = candidate_cost + curr_state->best_cost_;
    const bool is_new = next_state->best_cost_.is_invalid();
    if (is_new || new_state_cost < next_state->best_cost_) {
      next_state->pred_state_ = curr_state;
      next_state->best_cost_ = new_state_cost;
      next_state->best_candidate_kernel_ = candidate;
      if (is_new) {
        pq_.Push(next_state);
      } else {
        VLOG(1) << "Found better path!";
        pq_.Update(next_state);
      }
    } else {
      VLOG(1) << "Ignoring more expensive path";
    }
  }

 private:
  /*! \brief Available targets over which to explore. */
  CompilationConfig config_;
  /*! \brief Cost estimator to use for candidate kernels. */
  CostEstimator cost_estimator_;
  /*! \brief Map from covered sub-graphs to the corresponding state. */
  std::unordered_map<IndexSet, std::unique_ptr<SearchState>, IndexSetHash, IndexSetEqual>
      covered_to_state_;
  /*! \brief Priority queue of states, ordered by increasing cost. */
  PriorityQueue pq_;
};

}  // namespace

transform::Pass CollageFuseOps(CompilationConfig compilation_config) {
  auto pass_func = [=](Function function, IRModule mod, transform::PassContext ctxt) {
    Optional<Bool> opt_enable = ctxt->GetConfig<Bool>("relay.collage.enable_collage", Bool(false));
    if (!opt_enable.value()) {
      VLOG(1) << "ignoring since collage is disabled";
      return function;
    }
    // Though nothing goes wrong it's deeply confusing when we attempt to partition when
    // invoking the compiler to estimate the cost of a candidate kernel. So skip partitioning
    // entirely if the body of the function is a call to a "Primitive" function.
    const auto* call_node = function->body.as<CallNode>();
    if (call_node) {
      const auto* call_function_node = call_node->op.as<FunctionNode>();
      if (call_function_node) {
        if (call_function_node->HasNonzeroAttr(attr::kPrimitive)) {
          return function;
        }
      }
    }
    VLOG(1) << "Partitioning from:\n" << PrettyPrint(function);
    collage::Partitioner partitioner(compilation_config);
    Function result = Downcast<Function>(partitioner.Partition(function));
    VLOG(1) << "Partitioned to:\n" << PrettyPrint(result);
    return result;
  };
  return transform::CreateFunctionPass(pass_func, /*opt_level=*/0, "CollageFuseOps", {});
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
