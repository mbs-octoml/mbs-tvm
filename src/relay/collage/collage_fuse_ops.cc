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
 * \file src/relay/collage/collage_fuse_ops.cc
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

#include "../ir/dataflow_matcher_impl.h"
#include "./candidate_kernel.h"
#include "./cost.h"
#include "./cost_estimator.h"
#include "./fusion_rule.h"
#include "./fusion_spec.h"
#include "./gather_fusion_specs.h"
#include "./name_supply.h"
#include "./priority_queue.h"
#include "./sub_graph.h"
#include "./utils.h"

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

  std::string ToString() const {
    std::ostringstream os;
    os << "State(";
    os << "covered=" << covered_.ToString();
    os << ",best_cost=" << best_cost_.ToString();
    if (best_candidate_kernel_.defined()) {
      os << ",best_candidate_kernel=" << best_candidate_kernel_->ToString();
    }
    os << ")";
    return os.str();
  }

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

struct EqualSearchStatePtrs {
  bool operator()(const SearchState* left, const SearchState* right) const {
    return left->covered() == right->covered();
  }
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
  void Index(const Array<FusionSpec>& fusion_specs) {
    for (const auto& fusion_spec : fusion_specs) {
      VLOG_CONTEXT << "spec " << fusion_spec->spec_name_;
      Array<CandidateKernel> candidates = fusion_spec->AllCandidateKernels(*dataflow_graph_);
      for (auto& candidate : candidates) {
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
  explicit Partitioner(Array<FusionSpec> fusion_specs) : fusion_specs_(std::move(fusion_specs)) {}

  Expr Partition(Expr expr) {
    VLOG_CONTEXT << "Partitioning";
    // Establish data structures.
    dataflow_graph_ = CreateIndexedGraph(expr);
    name_supply_ = std::make_unique<NameSupply>("collage");
    VLOG(1) << "Created dataflow graph with " << dataflow_graph_->size() << " nodes";
    CandidateKernelIndex index(dataflow_graph_.get());
    index.Index(fusion_specs_);
    VLOG(1) << "Indexed " << index.size() << " candidate kernels";

    // Setup initial state.
    SearchState* init_state = GetState(IndexSet(dataflow_graph_->size()));
    init_state->best_cost_ = Cost::Zero();
    pq_.Push(init_state);

    VLOG(1) << "Commencing search";
    while (!pq_.empty()) {
      SearchState* curr_state = pq_.Pop();
      VLOG(1) << "Looking at state " << curr_state->covered_.ToString();
      PostDfsIndex next_index = curr_state->covered_.FirstOutsideIndex();

      if (next_index >= dataflow_graph_->size()) {
        // The entire expression has been explored. Collect the candidates on the optimal path.
        VLOG(1) << "Finished search, recovering best candidates";
        std::vector<CandidateKernel> best_candidates;
        while (curr_state != nullptr) {
          if (curr_state->best_candidate_kernel_.defined()) {
            VLOG(1) << "Including best candidate "
                    << curr_state->best_candidate_kernel_->ToString();
            best_candidates.emplace_back(curr_state->best_candidate_kernel_);
          }
          curr_state = curr_state->pred_state_;
        }
        return CandidateKernel::ParallelPartition(std::move(dataflow_graph_), std::move(expr),
                                                  std::move(best_candidates));
      }

      size_t num_fires = 0;
      Expr sub_expr = dataflow_graph_->index_to_node(next_index)->ref();
      VLOG(1) << "Looking at index " << next_index << " for sub-expression "
              << SubExprKindAndLabel(sub_expr).second;

      // Explore all the outgoing candidates from the current state.
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

      if (MustBeLowered(sub_expr)) {
        ICHECK_GT(num_fires, 0)
            << "No candidate was found covering sub-expression at index " << next_index
            << ", suggesting the fusion rules are incomplete for the given targets.";
      } else {
        // It is (also) possible to leave this sub-expression to be evaluated by the VM.
        // We'll assume that evaluation cost is zero.
        VLOG(1) << "Allowing possibility of current sub-expression being left behind";
        IndexSet next_covered =
            curr_state->covered_ | IndexSet(dataflow_graph_->size(), {next_index});
        SearchState* next_state = GetState(next_covered);
        Relax(curr_state, next_state, /*candidate=*/{});
      }
    }
    ICHECK(false) << "should have reached end state in which all sub-expressions are covered";
    return {};
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
   * \brief Record that it is possible to reach \p next_state by choosing \p candidate_kernel
   * in \p curr_state. If the resulting cost is better than the best known so far, update
   * \p next_state's best cost, predecessor and kernel to match.
   */
  void Relax(SearchState* curr_state, SearchState* next_state, CandidateKernel candidate) {
    Cost candidate_cost = Cost::Zero();
    if (candidate.defined()) {
      candidate_cost =
          candidate->EstimatedCost(*dataflow_graph_, &cost_estimator_, name_supply_.get());
    }
    Cost new_state_cost = candidate_cost + curr_state->best_cost_;
    const bool is_new = next_state->best_cost_.is_invalid();
    if (is_new || new_state_cost < next_state->best_cost_) {
      next_state->pred_state_ = curr_state;
      Cost previously_best_cost = next_state->best_cost_;
      next_state->best_cost_ = new_state_cost;
      CandidateKernel previously_best_candidate_kernel = next_state->best_candidate_kernel_;
      next_state->best_candidate_kernel_ = std::move(candidate);
      if (is_new) {
        VLOG(1) << "transition " << curr_state->ToString() << " --> " << next_state->ToString()
                << " (New state)";
        pq_.Push(next_state);
      } else {
        VLOG(1) << "transition " << curr_state->ToString() << " --> " << next_state->ToString()
                << " (Beats "
                << (previously_best_candidate_kernel.defined()
                        ? previously_best_candidate_kernel->ToString()
                        : "null")
                << " of cost " << previously_best_cost.ToString() << ")";
        pq_.Update(next_state);
      }
    } else {
      VLOG(1) << "transition " << curr_state->ToString() << " --> " << next_state->ToString()
              << " (Unchanged)";
    }
  }

 private:
  std::unique_ptr<DataflowGraph> dataflow_graph_;
  std::unique_ptr<NameSupply> name_supply_;
  /*! \brief Available fusion specs to use during search. */
  Array<FusionSpec> fusion_specs_;
  /*! \brief Cost estimator to use for candidate kernels. */
  CostEstimator cost_estimator_;
  /*! \brief Map from covered sub-graphs to the corresponding state. */
  std::unordered_map<IndexSet, std::unique_ptr<SearchState>, IndexSetHash, IndexSetEqual>
      covered_to_state_;
  /*! \brief Priority queue of states, ordered by increasing cost. */
  PriorityQueue<SearchState, CompareSearchStatePtrs, EqualSearchStatePtrs> pq_;
};

}  // namespace

transform::Pass CollageFuseOps(const CompilationConfig& compilation_config) {
  auto pass_func = [=](IRModule mod, transform::PassContext ctxt) {
    Optional<Bool> opt_enable = ctxt->GetConfig<Bool>("relay.collage.enable_collage", Bool(false));
    if (!opt_enable.value()) {
      VLOG(1) << "ignoring since collage is disabled";
      return mod;
    }
    Array<FusionSpec> fusion_specs = GatherFusionSpecs(compilation_config);
    VLOG(1) << "Gathered " << fusion_specs.size() << " fusion specs";
    IRModule out_mod = mod->ShallowCopy();
    for (const auto& kv : mod->functions) {
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        auto function = GetRef<Function>(function_node);
        VLOG(1) << "Partitioning " << kv.first->name_hint << " from:\n" << PrettyPrint(function);
        collage::Partitioner partitioner(fusion_specs);
        Function result = Downcast<Function>(partitioner.Partition(function));
        VLOG(1) << "Partitioned " << kv.first->name_hint << " to:\n" << PrettyPrint(result);
        out_mod->Add(kv.first, result);
      }
    }
    return out_mod;
  };
  return tvm::transform::CreateModulePass(pass_func, /*opt_level=*/0, "CollageFuseOps", {});
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
