# Design Doc: Collage

This design doc and accompanying
['v2' prototype implementation](https://github.com/mbs-octoml/mbs-tvm/tree/mbs-collage-sketch)
shows how to bring tuning to TVM's operator fusion and BYOC partitioning passes. The tuning search explores the
choice of sub-graphs as well as choice of toolchain (TVM native or one of the available BYOC integrations) for
each candidate kernel so as to minimize the expected model inference latency. We call the result a 'placement'.
This new tuning layer complements the tuning traditionally done by TVM and other toolchains during lowering. It
can also complement any global tuning, for example to explore all possible global layouts. 

The approach is based on the [preprint](https://arxiv.org/pdf/2111.00655.pdf):

> *Collage: Automated Integration of Deep Learning Backends*  
> Byungsoo Jeon, Sunghyun Park, Peiyuan Liao, Sheng Xu, Tianqi Chen, Zhihao Jia

(We like the 'Collage' name so much we'd like to continue to use it here, subject to author approval.)

This tuning approach contrasts with TVM's existing "greedy" and "manual" approaches to fusion and BYOC:

- Greedy: Currently only the largest possible supported sub-graphs are used for kernels, irrespective of their execution
  time. With Collage many more candidate sub-graphs are explored, and it is possible for two smaller sub-graphs to yield
  better overall latency than one large sub-graph if they mix toolchains.
- Manual: Currently the TVM user must commit to a BYOC toolchain and invoke the corresponding partitioning function
  before the main TVM compilation flow proceeds. With Collage the choice of toolchain can be automated based on measured
  latency. Collage will also explore mixing and matching between multiple BYOC toolchains as well as TVM's native
  backend.

The design replaces TVM's fixed `FuseOps` and BYOC-provided `partition_for_<toolchain>` operations (built using
the `MergeComposite`/`AnnotateTarget`/`MergeCompilerRegions`/`PartitionGraph` passes) with a single new
`CollageFuseOps` pass. The pass is carefully engineered to build directly on the existing `"TOpPattern"` attributes
(provided for every Relay operator and used by `FuseOps`), BYOC `"target.<toolchain>"`
operator predicates (provided for some operator/toolchain pairs by 'operator-based' BYOC integrations) and BYOC operator
pattern/predicates (registered in the pattern table by 'pattern-based' BYOC integrations). In this way only the more
boilerplate aspects of existing BYOC integrations need to be adjusted to support Collage.

Collage offers four advantages:

- **Latency**: Overall model latency may be reduced compared to TVM native, TVM with a specific BYOC toolchain, or a
  non-TVM compiler such as TensorRT.
- **Automation**: The choice of which BYOC toolchains to enable can be automated.
- **Economy of implementation**: Five standalone passes using three separate mechanisms for expressing fusion
  rules/algorithms and implementing partitioning can be replaced with one, which itself is built from compositional
  primitives.
- **Decoupling**: It is ok for a candidate kernel found during search to actually not be valid for a toolchain (even
  TVMs). Such candidates can simply be given 'infinite' cost and thus ignored. This can help reduce the tight coupling
  between backends and fusion rules.

The results of the preprint were derived in a [branch](https://github.com/cmu-catalyst/collage) from
[TVM](https://github.com/apache/tvm) at `461d06eb5cfc7954f1983779acd05c47cea269f1`. We have since rebased that code onto
main, and refer to it as the
['v1' prototype implementation](https://github.com/mbs-octoml/mbs-tvm/tree/mbs-collage-port). In comparison to the
'v1' prototype, this design:

- Avoids the need to add any new 'Collage specific' fusion patterns and predicates. We want to make sure Collage can
  work even for out-of-tree BYOC toolchains (modulo some of the BYOC API changes we discuss below).
- Builds on the existing support for heterogeneous `Target`s to represent the menu of available toolchains to use during
  search. In particular, we want to allow users to blend `on_device` annotations (to express preferences for which
  devices should execute which sub-graphs) with Collage (to find the best kernels and toolchains respecting those device
  preferences).
- Uses the existing convention for `"Primitive"`, `"Composite"` and `"Compiler"` attributes on Relay `Function`s to
  express the assignment of sub-graph to toolchain.
- Implements support for 3rd party libraries (eg cudnn) so as to allow an N-to-1 mapping from Relay operators to library
  call (this is not yet implemented in the 'v2' prototype, see below for the sketch).
- Is implemented mostly in C++.

However:

- The 'v2' prototype only implements the 'op-level' dynamic-programming based search strategy from the paper. Though the
  paper reports encouraging results with the 'graph-level' evolutionary-search strategy we leave that to future work.

This design supersedes the
earlier [draft](https://www.notion.so/octoml/Design-Doc-Collage-in-Main-f2306fb1ae7a4245ac0b3752bd244a1e) based on what
we've learned so far in the 'v2' prototype.

## Known Limitations

- **Some BYOC boilerplate changes required**: TVM's current BYOC integration API only requires the 'lowering/codegen'
  function to be registered to a well-known global function name. Everything else is up to the BYOC author.
    - Collage requires pattern-based BYOC integrations to register their patterns in the global pattern table.
    - Collage requires the BYOC lowering function to yield a valid `runtime::Module` without requiring any additional
      BYOC-specific passes to be run.
    - Collage requires the BYOC integration to either correctly test for which operators are supported in the
      pattern/operator predicate, or gracefully propagate failure rather than CHECK-fail if an unsupported operator is
      included in a candidate kernel. Thus a BYOC integration will need to be 'robustified' to become 'Collage
      compatible'.
- **Higher tuning cost**: Obviously Collage needs to estimate the latency of many more candidate kernels, and each
  candidate may itself trigger tuning during lowering. For TVM this can require `O(thousands)` of  trials and take
  `O(hours)`, so we'll be very dependent on cached tuning logs to amortize this cost between models for the same
  target.
- **Task extraction vs Tuning**: Traditionally TVM has had three phases: i) Task extraction (find the fused sub-graphs
  to tune), ii) Tuning (find a good schedule for those sub-graphs), and iii) Compilation (re-compile the model, now 
  retrieving schedules for all the anticipated sub-graphs from the cache.) However the Collage 'v2' prototype collapses
  all these phases. This lets us lazily explore the implied search graph (nodes = partially rewritten models, edges =
  selected of sub-graph and toolchain as a candidate kernel, cost = estimated sum of kernel costs plus launch
  penalties), and thus only pay the cost of tuning candidate kernels which could possibly influence the final
  placement.
- **No per-candidate rewriting**: Though Collage can explore the choice of sub-graph and toolchain, it cannot explore
  any additional rewrites to apply to the arguments or result of that sub-graph. So, for example, Collage cannot be used
  to search over the choice of layout for a kernel since any choice other than the model's default must be
  'corrected' for by the inserted layout transformations. To support this efficiently we'd need to abandon the
  simple-minded but fast `SubGraph` representation we describe below in favor of something like an EGraph
  representation, which seems like a very large change for TVM.
- **Dependency management**: Currently BYOC integrations tend to assume they are the only non-TVM toolchain in use. So
  it's possible two toolchains introduce runtime dependencies which can't be satisfied. Collage has no notion of
  dependencies or incompatibilities and may attemt to mix candidate kernels we can't support in prod.
- **Additive kernel cost assumption**: Collage as per this design assumes the cost of running candidate kernels is
  additive, plus a small launch penalty. However cache effects can dominate measured latency, particularly for 'light'
  kernels. Thus there may be a **additive error** in the final result:

  > additive_error = measured_latency(collage_placement) - sum_{kernel} (estimated_latency(kernel) + penalty)

- **Limited search space**: Naively exploring all sub-graphs is `O(n!)`, so we need to constrain the search. The easiest
  approach is just to limit candidate kernels to sub-graphs of just a few operators. This can mean significatly faster
  candidates are not explored, yielding a placement with high **optimality loss**:

  > optimality_loss = measured_latency(collage_placement) - measured_latency(true_optimal_placement)

  Though the 'true' optimal placement may be infeasible to find, the user may easily discover a high
  **apparent loss**, eg by comparing the Collage result with a traditional BYOC partitioning result:

  > apparent_loss = measured_latency(collage_placement) - measured_latency(users_own_placement)

- **Fragile toolchains**: Some BYOC toolchains are intended to be stand-alone compilers in their own right, and have
  been tuned against common models and include global flags to guide optimizations such as reducing precision. However
  Collage will only feed these toolchains smaller sub-graphs, thus making the limited search space problem more severe.
- **High variance in lightweight kernels**: Small kernels can have high variance, thus the choice of which toolchain to
  use can be arbitrary. We probably want to i) validate our variance estimator is accurate, ii) choose a percentile
  slightly above 50% for the estimated candidate kernel latency, and iii) fall back to hard-coded priorities when the
  measured variance is too high.
- **Non-compositional BYOC toolchains**: BYOC partitioning functions often run global passes to get the Relay graph into
  a state better aligned with the toolchain on the assumption they are the exclusive partitioning pass. Most obvious is
  the choice of layout, and if two BYOC integrations have a different choice of layout then there's currently no way for
  them to be used concurrently. All of those passes must either be i) pushed up to global configuration (which could be
  explored by a search layer outside of TVM), ii) pushed into the BYOC lowering/codegen function (to prepare the
  sub-graph for further compilation) or iii) moved into the standard Relay optimization passes run before
  `CollageFuseOps`.
- **Repeated FuseOps**: Some passes (eg `ManifestAlloc`) introduce new calls to primitive function which must be fused
  and lowered, even though the main work of fusion and lowering has already occured. We'll need to either
  retain `FuseOps`, or support `CollageFuseOps` in 'lite' mode to handle those.
- **Explainability**: It's easy to show the user the final placement and estimated times for each kernel, but harder to
  show why that placement won out over all others during search.

## TODO in the 'v2' prototype

- Implement extern-for-TVM support and bring in `cudnn` and `cublas`.
- Finish `SubGraph` support for all of Relay.
- Cross-check against one of the 'v1' models.
- Bring up on `GPT2`.
- Explore `float16` performance mixing `CUTLASS` and `TensorRT`.
- Implement TVM tuning-on-the-fly.
- Connect estimator to production cache.
- Estimator in units of `IRModule` not `Function`. Resolve `params` binding question.
- Find model+target combination that shows compelling speedup from mixing w.r.t. all other options, including
  stand alone `TensorRT`.

