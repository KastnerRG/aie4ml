# AIE4ML Paper Implementation Audit

This document maps the claims and architecture in `aie4ml.pdf` to the code in this repository.

It has two goals:

1. Explain which parts of the repo implement which parts of the paper.
2. Identify where the repo only partially matches the paper, or diverges from it.

## Quick verdict

The repository does implement the core flow described in the paper:

- `hls4ml` backend plugin registration
- A dedicated AIE IR
- a pass pipeline for lowering, quantization, resolve, packing, placement, routing, and emission
- dense-layer kernels based on `aie::mmul`
- fused bias and ReLU in the generated kernel
- cascade-based layer scaling
- memory-tile style graph planning through `adf::shared_buffer` descriptors
- x86 and AIE simulation support
- generated Vitis projects with templates and packed weights

The strongest coverage is for the paper's compiler/toolflow sections and for the dense kernel implementation.

The main gaps or mismatches are:

- only dense layers with optional bias and ReLU are implemented in this repo
- benchmark/evaluation scripts and reproduced numbers from Section V are not present
- the implemented placement cost function does not exactly match Equation (2) in the paper
- the pass order in code differs from the paper's narrative
- the paper mentions `.analyze()`, but the repo currently exposes report parsing via `read_aie_report()` rather than a backend method with that name

## Coverage matrix

| Paper topic | Repo status | Main implementation points | Notes |
| --- | --- | --- | --- |
| Backend integration with `hls4ml` | Implemented | `src/aie4ml/plugin.py`, `src/aie4ml/aie_backend.py` | Backend and writer are registered as `AIE`. |
| Dedicated AIE IR | Implemented | `src/aie4ml/ir/graph.py`, `src/aie4ml/ir/context.py` | Logical, kernel, and physical IR are all present. |
| Lowering from `hls4ml` IR | Implemented | `src/aie4ml/passes/lower.py` | Builds AIE IR, copies metadata, normalizes transpose traits. |
| Quantized integer flow / bit-exact handling | Implemented | `src/aie4ml/passes/quant.py`, `src/aie4ml/passes/resolve_registry.py`, `src/aie4ml/simulation.py` | Quantizes tensors, resolves AIE storage types, quantizes IO for prediction. |
| Dense + ReLU fusion | Implemented | `src/aie4ml/passes/fuse_activation.py` | Fusion exists, but as a separate pass after quantization rather than inside lowering. |
| Transpose/view folding | Implemented | `src/aie4ml/passes/fold_transpose.py` | Turns transpose into IO-view metadata instead of a runtime operator. |
| Resolve numeric types, tiling, parallelism, placement hints | Implemented | `src/aie4ml/passes/resolve.py`, `src/aie4ml/passes/resolve_registry.py` | This is the main policy-driven attribute resolver. |
| Packing tiled weights and bias | Implemented | `src/aie4ml/passes/pack.py`, `src/aie4ml/kernel_registry.py` | Packs stationary tensors into slice/tile-major layouts expected by the kernel. |
| Dense kernel using `aie::mmul` | Implemented | `src/aie4ml/templates/nnet_utils/dense_bias_relu/*` | This is the paper's main kernel implementation. |
| 2x2 blocked schedule with four accumulators | Implemented | `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.cpp` | `C00/C01/C10/C11` are computed in parallel exactly as described. |
| Bias and ReLU fused in kernel epilogue | Implemented | `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.cpp` | Single-tile and cascade-final kernels do fused bias and optional ReLU. |
| Layer scaling with cascade rows/columns | Implemented | `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu_graph.h`, `src/aie4ml/passes/resolve_registry.py` | `CAS_LENGTH` and `CAS_NUM` map to horizontal and vertical scaling. |
| Memory-tile routing / retiling / padding | Implemented | `src/aie4ml/passes/memory_plan.py`, `src/aie4ml/passes/memtile_legalize.py`, `src/aie4ml/templates/firmware/graph_plan.h.jinja` | Uses shared buffers plus explicit tiling descriptors, offsets, traversal, and boundary dimensions. |
| Fanout handling | Implemented | `src/aie4ml/passes/fanout_legalize.py` | Splits multi-consumer/output entries before materialization. |
| Memtile port-limit legalization | Implemented | `src/aie4ml/passes/memtile_legalize.py` | Shards edges across multiple memtile units when port counts exceed device limits. |
| Automatic graph placement (B&B) | Implemented with caveat | `src/aie4ml/passes/placement.py` | Branch-and-bound exists, but vertical cost differs from the paper formula. |
| Project emission via Jinja | Implemented | `src/aie4ml/writer.py`, `src/aie4ml/templates/firmware/*` | Generates a full Vitis project. |
| x86 and AIE simulation | Implemented | `src/aie4ml/aie_backend.py`, `src/aie4ml/simulation.py` | `predict(..., simulator='x86'|'aie')` is implemented. |
| AIE report parsing / profiling extraction | Partially implemented | `src/aie4ml/simulation.py` | Report parsing exists, but not as a backend `.analyze()` method. |
| Support for AIE-ML and AIE-MLv2 | Implemented, limited to current operator set | `src/aie4ml/aie_devices.json`, `src/aie4ml/kernel_registry.py`, `src/aie4ml/passes/resolve_registry.py` | Device catalog and tiling options include both generations. |
| Multiple operator/model classes beyond dense | Not implemented | repo-wide | Current scope is dense/linear with optional bias and ReLU. |
| Reproduced evaluation pipeline from Section V | Not present | repo-wide | No benchmark harness or data reproducing paper tables/figures. |

## Guide 1: Read the repo in the same order as the paper's compiler flow

### 1. Frontend and user API

Paper mapping:

- Section IV-B toolflow and simulation
- Figure 2 user API

Repo files:

- `src/aie4ml/plugin.py`
- `src/aie4ml/aie_backend.py`

What they do:

- `plugin.py` registers the backend name `AIE` and its writer with `hls4ml`.
- `AIEBackend` defines the backend-specific flow and the user-visible methods:
  - `create_initial_config()`
  - `write()`
  - `compile()`
  - `predict()`
  - `build()`

How this matches the paper:

- The paper says users enter through `hls4ml` and select backend `AIE`. That is exactly how this repo is wired.
- The paper says the flow supports x86 and AIE simulation. `predict()` supports `simulator='x86'` and `simulator='aie'`.
- The paper says users can pass floating-point NumPy inputs and optionally quantize/dequantize at the boundary. `predict(..., quantize_io=True)` does that.

### 2. Lowering into a dedicated AIE IR

Paper mapping:

- Section IV-A, "Intermediate Representation (IR)"
- Section IV-A, pass step 1

Repo files:

- `src/aie4ml/passes/lower.py`
- `src/aie4ml/ir/graph.py`
- `src/aie4ml/ir/context.py`

What they do:

- `LowerToAieIr` builds a separate AIE-specific IR from the `hls4ml` model.
- The IR is split into:
  - `LogicalIR`: semantic graph of tensors and ops
  - `KernelIR`: chosen kernel variants plus resolved attributes
  - `PhysicalIR`: placement and routing plan
- Traits such as `fused_activation` and `io_view` are attached to nodes to carry AIE-specific semantics.

How this matches the paper:

- This is a direct implementation of the paper's "dedicated IR" idea.
- The repo really does use the AIE IR as the central structure passed through all later stages.

Important difference:

- The paper says lowering applies simple fusions such as Dense+ReLU.
- In the repo, lowering only builds IR and attaches baseline metadata; Dense+ReLU fusion is done later by `FuseActivationCasts`.

### 3. Quantization and resolved AIE attributes

Paper mapping:

- Section IV-A, pass steps 2 and 3
- Abstract claim on preserving bit-exactness

Repo files:

- `src/aie4ml/passes/quant.py`
- `src/aie4ml/passes/resolve.py`
- `src/aie4ml/passes/resolve_registry.py`
- `src/aie4ml/aie_types.py`

What they do:

- `IntegerQuantizer` converts frontend tensors into integer arrays used by the backend:
  - quantized weights
  - quantized bias
  - quantization metadata for input/output/weight/bias
- `resolve_registry.py` then derives:
  - storage dtypes for input/output/weight/bias/accumulator
  - required output shift
  - tiling (`tile_m`, `tile_k`, `tile_n`)
  - parallelism (`cas_num`, `cas_length`)
  - padded feature sizes and padded independent extent
  - IO routing defaults
  - placement hints
  - staging metadata

How this matches the paper:

- This is the heart of the paper's "resolve all deterministic AIE attributes" step.
- The code honors user overrides for tiling, parallelism, and placement if valid.
- The numeric flow preserves explicit rounding/saturation metadata and carries it into kernel configuration.

How bit-exactness is approached:

- The backend quantizes weights and biases into integer arrays using the frontend precision intent.
- The kernel shift is derived from input/weight/output fractional bits.
- `predict()` quantizes float inputs and dequantizes integer outputs consistently with resolved precisions.

Practical limitation:

- The repo currently supports the dense kernel precision combinations listed in `DenseKernelVariant.supported_precisions`.
- Operator coverage is much narrower than the paper's broad framing of "complete neural networks".

### 4. Packing stationary tensors

Paper mapping:

- Section IV-A, pass step 4
- Section III-A kernel data layout assumptions

Repo files:

- `src/aie4ml/passes/pack.py`
- `src/aie4ml/kernel_registry.py`

What they do:

- `pack_mmul_rhs_matrix()` tiles and flattens weights according to:
  - input slice
  - output slice
  - kernel tile sizes
  - cascade layout
- `pack_vector_by_n_slice()` packs bias per output slice.
- The selected kernel variant (`DenseKernelVariant`) owns the pack behavior and publishes packed artifacts to code generation.

How this matches the paper:

- This directly implements the paper's statement that stationary tensors are reorganized into tiled layouts compatible with AIE intrinsics.

## Guide 2: Understand the hardware implementation from the generated kernel code

### 1. The `aie::mmul` dense kernel

Paper mapping:

- Section III-A
- Algorithm 1
- Figure 1 left

Repo files:

- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h`
- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.cpp`
- `src/aie4ml/templates/firmware/variants/dense_bias_relu/parameters.h.jinja`

What is implemented:

- The kernel family is split into:
  - `dense_single`
  - `dense_first`
  - `dense_middle`
  - `dense_last`
- `dense_single` handles the single-tile case.
- `dense_first/middle/last` implement horizontal cascades for partial-sum reduction.

Why this matches the paper closely:

- The kernel is explicitly based on `aie::mmul<M, K, N, ...>`.
- The 2x2 blocked schedule is present:
  - two input tiles
  - two weight tiles
  - four accumulators: `C00`, `C01`, `C10`, `C11`
- Bias is fused into the accumulator domain.
- ReLU is optionally fused before store.
- Output conversion applies the resolved `SHIFT` when converting accumulators to output vectors.

Concrete code signals:

- `MMUL C00, C01, C10, C11;`
- repeated `mac()` updates in the K loop
- optional bias load/replication
- optional `aie::max(..., 0)` before store

Notes:

- The paper discusses `VST.SRS` as a fused quantized store conceptually. The repo expresses this through `to_vector<result_t>(SHIFT)` plus AIE rounding/saturation mode setup rather than exposing a raw `VST.SRS` primitive directly.
- Input transpose support is implemented with `aie::transpose(...)` when the folded view requires it.

### 2. Layer-level scaling across the 2D array

Paper mapping:

- Section III-B
- Figure 1 middle

Repo files:

- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu_graph.h`
- `src/aie4ml/passes/resolve_registry.py`

What is implemented:

- `cas_length` controls horizontal cascade length.
- `cas_num` controls the number of vertical chains producing distinct output slices.
- `dense_bias_relu_graph` instantiates a `CAS_NUM x CAS_LENGTH` kernel grid.
- Cascades are connected west-to-east with `connect<cascade>(...)`.
- One input port per column is connected to all rows in that column.

How this matches the paper:

- The code uses cascades for horizontal partial-sum reduction.
- It replicates rows vertically for output slicing.
- It broadcasts the same column input to all kernels in that column.

Important implementation detail:

- The repo achieves the "input broadcast upward from memory tiles" behavior by combining the top-level graph plan with per-layer graph fanout. At the layer graph level, `in1[col]` is connected to every row kernel for that column.

### 3. Weights and bias resident on AIE tiles

Paper mapping:

- Section III introduction to kernel/memory design

Repo files:

- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu_graph.h`
- `src/aie4ml/templates/firmware/app.cpp.jinja`

What is implemented:

- Weights and biases are exposed as ADF parameter ports.
- The generated app updates them once before running the graph.
- Kernel graphs connect these parameter ports asynchronously to the kernels.

How this matches the paper:

- This implements the paper's "weights and biases loaded once from PS through RTP ports and stored directly in local AIE memories."

## Guide 3: Understand memory tiles, inter-layer routing, and graph emission

### 1. Memory planning and memtile materialization

Paper mapping:

- Section III-B data partitioning through memory tiles
- Section III-C scaling across layers and graph interconnect
- Section IV-A, pass step 5

Repo files:

- `src/aie4ml/passes/memory_plan.py`
- `src/aie4ml/passes/fanout_legalize.py`
- `src/aie4ml/passes/memtile_legalize.py`
- `src/aie4ml/templates/firmware/graph_plan.h.jinja`

What they do:

- `memory_plan.py` collects producer/consumer connections and groups them into edge entries.
- It decides whether an edge can be direct or must go through a memtile/shared buffer.
- `fanout_legalize.py` rewrites multi-consumer edges into legal single-consumer entries.
- `memtile_legalize.py` shards entries when memtile in/out port limits would be exceeded.
- `graph_plan.h.jinja` emits `adf::shared_buffer` instances plus detailed tiling/read/write descriptors.

How this matches the paper:

- Inter-layer dataflow is implemented through shared buffers with explicit dimensions, offsets, tiling, and traversal descriptors.
- The emitted descriptors support:
  - retiling between producer and consumer
  - padding via `boundary_dimension`
  - double buffering via `num_buffers = 2`

Why this is the strongest evidence for the paper's memory-tile claims:

- The code is not doing a vague "buffer between layers"; it is computing the exact DMA-style access descriptors that the paper describes.
- The planner distinguishes graph inputs, graph outputs, kernel-to-kernel edges, and legalizes them before materialization.

Useful mental model:

- `memory_plan.py` is where the repo becomes a graph compiler rather than only a kernel generator.

### 2. Direct edges versus memtile transport

Paper mapping:

- Section III-C graph interconnect

Repo files:

- `src/aie4ml/passes/memory_plan.py`

What it does:

- If a connection is simple enough, the planner can emit a direct kernel-to-kernel edge.
- Otherwise it emits a shared-buffer-backed path.

Direct transport is allowed only when:

- one producer
- one consumer
- one producer port
- one consumer port
- no graph output split
- no transpose-based view complication
- user routing policy does not force memtile

Why this matters:

- The paper emphasizes memory-tile interconnect, but the repo also contains an optimization path for direct transport when legal.
- That is an implementation improvement/detail that is not highlighted in the paper text.

### 3. Project emission

Paper mapping:

- Section IV-A, pass step 7

Repo files:

- `src/aie4ml/writer.py`
- `src/aie4ml/templates/firmware/top_graph.h.jinja`
- `src/aie4ml/templates/firmware/graph_plan.h.jinja`
- `src/aie4ml/templates/firmware/app.cpp.jinja`
- `src/aie4ml/templates/firmware/Makefile.jinja`
- `src/aie4ml/templates/firmware/aie.cfg.jinja`

What they do:

- Export pipeline IR to `aie_pipeline.json`
- write packed weight and bias headers
- copy reusable kernel sources
- render the top graph, graph plan, parameters, app, and build files

How this matches the paper:

- This is exactly the paper's "templated code generation system" and "project emission" stage.
- The result is a buildable Vitis project, not only generated kernels.

## Guide 4: Understand automatic placement and where it diverges from the paper

### 1. What the repo implements

Paper mapping:

- Section IV-C
- Equation (2)
- Figure 3

Repo file:

- `src/aie4ml/passes/placement.py`

What is implemented:

- Each layer is modeled as a rectangle:
  - width = cascade length
  - height = cascade count
- Placement is solved for a linear chain with branch-and-bound.
- Fixed user coordinates are honored as hard anchors.
- A greedy seed solution is used to improve pruning.
- A placement conflict rule includes a one-column left margin to avoid bank-conflict-style adjacency issues.

This is a real branch-and-bound implementation, not just a greedy scan.

### 2. Important mismatch with the paper

The paper's cost function is:

- horizontal distance between successive graphs
- plus `lambda * |r_out - r_in|`
- plus `mu * top_row`

The repo currently computes:

- horizontal distance between successive graphs
- plus `lambda * abs(r_in - 0)`
- plus `mu * y`

The code even marks this with a TODO:

- `# TODO for direct connections we must use both vertical costs`

Interpretation:

- The branch-and-bound framework matches the paper.
- The exact vertical term in Equation (2) does not.
- So the repo implements the placement idea, but not the exact published objective.

This is the clearest paper-to-code divergence in the repository.

### 3. Pass ordering mismatch

The paper describes:

1. graph planning
2. placement

The repo flow is:

1. resolve
2. pack
3. placement
4. memory entry collection
5. fanout legalization
6. memtile legalization
7. memory plan materialization

Interpretation:

- Placement is done before graph planning in code.
- That is workable because placement only needs kernel footprints and optional hints.
- But it is not the same pipeline order described in the paper.

## Guide 5: Simulation, runtime API, and testing

### 1. Prediction and simulation

Paper mapping:

- Section IV-B toolflow and simulation

Repo files:

- `src/aie4ml/aie_backend.py`
- `src/aie4ml/simulation.py`

What is implemented:

- `compile()` invokes `make x86com`
- `predict(..., simulator='x86')` invokes `make x86sim`
- `predict(..., simulator='aie')` invokes `make aiesim`
- input files are written in PLIO format
- outputs are collected and optionally dequantized back to float

How this matches the paper:

- This is a direct implementation of the x86/AIE simulation story from the paper.

Difference from the paper figure/API wording:

- The paper mentions `.analyze()`.
- The repo provides `read_aie_report()` in `simulation.py`, but not a backend method named `.analyze()`.

### 2. What the tests actually verify

Repo files:

- `tests/test_hls4ml_dense.py`
- `tests/test_hls4ml_fanout.py`

What is covered:

- conversion from QKeras models into the AIE backend
- x86 simulation against QKeras reference outputs
- fanout cases
- N-D input handling

What is not covered:

- reproduced paper benchmarks
- hardware AIE performance numbers from Section V
- automated validation of placement quality versus published Figure 3
- systematic mixed-precision cross-layer cases

## Missing or partial relative to the paper

### Clearly missing from the repo

- Benchmark harness reproducing Section V tables and figures
- Cross-device comparison scripts and baselines
- Additional operator coverage beyond dense/linear plus optional bias and ReLU

### Partially represented

- PyTorch/TensorFlow support exists only indirectly through `hls4ml` compatibility; the repo itself does not contain dedicated frontend adapters for those frameworks.
- AIE report analysis exists as a helper function rather than the paper's `.analyze()` API.
- Mixed precision is supported at the per-layer resolution/configuration level, but the repo does not contain strong explicit tests or dedicated cast operators demonstrating arbitrary cross-layer precision transitions.

### Divergences worth calling out

- Fusion is not part of the lowering pass itself; it is a separate pass.
- Placement happens before graph planning in code.
- The placement cost function differs from Equation (2), specifically in the vertical-distance term.

## Best file reading order for future work

If you want to understand or extend the implementation efficiently, read in this order:

1. `src/aie4ml/aie_backend.py`
2. `src/aie4ml/passes/lower.py`
3. `src/aie4ml/passes/quant.py`
4. `src/aie4ml/passes/resolve.py`
5. `src/aie4ml/passes/resolve_registry.py`
6. `src/aie4ml/kernel_registry.py`
7. `src/aie4ml/passes/pack.py`
8. `src/aie4ml/passes/placement.py`
9. `src/aie4ml/passes/memory_plan.py`
10. `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.cpp`
11. `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu_graph.h`
12. `src/aie4ml/writer.py`
13. `src/aie4ml/templates/firmware/graph_plan.h.jinja`
14. `src/aie4ml/simulation.py`

## Bottom line

This repo is a credible implementation of the paper's main compiler architecture and dense-kernel strategy.

The core paper claims that are clearly implemented are:

- end-to-end `hls4ml` backend flow
- AIE-specific IR and passes
- integer quantized dense kernel generation
- fused bias/ReLU
- cascade-based layer scaling
- memory-tile-aware inter-layer routing
- branch-and-bound style placement
- Vitis project emission and simulation

The main places where the paper overstates what this repo alone demonstrates are:

- the breadth of model/operator coverage
- the reproduced evaluation story
- the exact placement objective as published

