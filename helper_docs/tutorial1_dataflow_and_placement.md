# Tutorial 1 Dataflow And Placement Notes

Scope: this note uses `tutorials/tutorial_1.ipynb` as the concrete example, and cross-checks three things:

- the paper claim in Section III.C,
- the aie4ml compiler code that generates the AIE project,
- the generated project files already present in `tutorials/proj_aie/` and `tutorials/proj_aie_tuned/`.

## Short answer

For `tutorial_1`, aie4ml does not wire the network as pure kernel-to-kernel streams. The generated project uses:

- `PLIO` stream ports at the graph boundary,
- `adf::shared_buffer` objects for graph input/output staging and inter-layer movement,
- `read_access` / `write_access` tiling descriptors to describe how those buffers are filled and drained,
- `cascade` links only inside a multi-column dense layer,
- RTP/parameter ports for weights and bias.

For placement, the repo does implement a real placement pass that matches the paper at a high level: it models each layer as a rectangle and runs a branch-and-bound search. But the solver places each layer graph by its top-left anchor, not every kernel tile independently. The per-kernel tile coordinates are then derived deterministically inside the generated layer graph template. There is also one important mismatch with the paper: the current vertical-cost term in code is not exactly the same as Equation (2).

## Q1. Dataflow walkthrough

### 1. Start from the neural network in `tutorial_1`

The notebook builds a very small quantized MLP:

- input features = 128
- hidden layer = 256
- output features = 64
- batch size = 8

The relevant notebook cells are:

- `tutorials/tutorial_1.ipynb:161-163` for `IN_FEATURES = 128`, `HIDDEN = 256`, `OUT_FEATURES = 64`
- `tutorials/tutorial_1.ipynb:167-178` for the QKeras model
- `tutorials/tutorial_1.ipynb:433-434` for `BATCH = 8` and `ITERS = 10`

The model structure is:

`QActivation(8-bit) -> QDense(128 -> 256) -> ReLU -> QDense(256 -> 64) -> QActivation(8-bit)`

After lowering into the AIE flow, the two real compute layers are:

- `qfc1_aie`: 128 inputs -> 256 outputs, with fused ReLU
- `qfc2_aie`: 256 inputs -> 64 outputs

You can see that directly in the generated IR:

- `tutorials/proj_aie_tuned/aie_pipeline.json:18-26` for `qfc1_aie`
- `tutorials/proj_aie_tuned/aie_pipeline.json:97-105` for `qfc2_aie`

For the rest of this walkthrough, I will use `tutorials/proj_aie_tuned/` as the main example, because it shows the full mechanism the paper talks about:

- feature slicing on the input,
- multiple input PLIOs,
- a 2x2 placement for the first layer,
- vertical broadcast,
- horizontal cascade.

The untuned `proj_aie/` follows the same pattern, just with a smaller first layer graph and no cascade because `CAS_LENGTH = 1`.

### 2. What hardware shape aie4ml chose for the tuned project

The tuned notebook overrides the first dense layer with:

- `parallelism = {cas_num: 2, cas_length: 2}`
- `tiling = {tile_m: 4, tile_k: 8, tile_n: 8}`
- `placement = {row: 0, col: 10}`

That comes directly from `tutorials/tutorial_1.ipynb:1576-1597`.

The generated tuned config in `tutorials/proj_aie_tuned/src/parameters.h` makes the final architecture explicit:

- `L1Cfg` in `tutorials/proj_aie_tuned/src/parameters.h:8-35`
- `L2Cfg` in `tutorials/proj_aie_tuned/src/parameters.h:36-64`

For `qfc1_aie`:

- `CAS_LENGTH = 2`, `CAS_NUM = 2` in `tutorials/proj_aie_tuned/src/parameters.h:14`
- `IN_FEAT_SLICE = 64`, `OUT_FEAT_SLICE = 128` in `tutorials/proj_aie_tuned/src/parameters.h:25-28`
- `M = 4`, `K = 8`, `N = 8` in `tutorials/proj_aie_tuned/src/parameters.h:19`
- placement anchor = `(10, 0)` in `tutorials/proj_aie_tuned/src/parameters.h:20-21`

This means:

- the first layer is implemented as a 2 columns x 2 rows compute rectangle,
- each tile sees 64 input features and contributes to or produces 128 output features,
- left-to-right is the input-feature split,
- top-to-bottom is the output-feature split.

For `qfc2_aie`:

- `CAS_LENGTH = 1`, `CAS_NUM = 1` in `tutorials/proj_aie_tuned/src/parameters.h:42`
- `IN_FEAT_SLICE = 256`, `OUT_FEAT_SLICE = 64` in `tutorials/proj_aie_tuned/src/parameters.h:53-56`
- placement anchor = `(13, 0)` in `tutorials/proj_aie_tuned/src/parameters.h:48-49`

So the second layer is a single compute tile that consumes the full 256-feature hidden activation and produces the full 64-feature output.

### 3. Big architectural picture for `proj_aie_tuned`

At a high level, the tuned project uses:

- 5 compute tiles total:
  - 4 tiles for `qfc1_aie`
  - 1 tile for `qfc2_aie`
- 3 PLIOs total:
  - 2 input PLIOs
  - 1 output PLIO
- 3 graph-level shared buffers:
  - `buffer_layer2_out`
  - `buffer_layer5_out`
  - `buffer_layer6_out`
- double buffering on every graph-level shared buffer (`num_buffers = 2`)

The graph-level shared buffers are declared in:

- `tutorials/proj_aie_tuned/src/top_graph.h:36-38`

and instantiated as double-buffered in:

- `tutorials/proj_aie_tuned/src/graph_plan.h:14-19`
- `tutorials/proj_aie_tuned/src/graph_plan.h:55-60`
- `tutorials/proj_aie_tuned/src/graph_plan.h:93-98`

So from the framework's point of view, there are three MEM-tile-style staging points in the graph. The paper describes this as chaining layer graphs through memory tiles with ping-pong buffering and programmable tilers, and the generated project matches that model.

One important nuance: the repo proves the existence of three `shared_buffer` stages and Vitis-generated DMA-backed adaptors for them. It does not give a single simple line that says "exactly N hardware MEM tiles reserved." The final physical packing of those shared buffers is done by Vitis. So the safe statement is:

- 3 graph-level MEM-tile-mediated handoff buffers are used,
- and Vitis materializes them with DMA-backed memory/stream adaptors.

### 4. Where the compute tiles and PLIOs are physically placed

The tuned compute placement is explicit.

The layer anchors are applied in:

- `tutorials/proj_aie_tuned/src/top_graph.h:81-88`

The per-kernel tile coordinates are then derived in:

- `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:32-45`

For `qfc1_aie`, the placement anchor is `(10,0)`, so the four kernels land at:

- `kk[0]` -> `(10,0)`
- `kk[1]` -> `(11,0)`
- `kk[2]` -> `(10,1)`
- `kk[3]` -> `(11,1)`

For `qfc2_aie`, the single kernel lands at:

- `kk[0]` -> `(13,0)`

The compiled Vitis map report confirms those placements:

- `tutorials/proj_aie_tuned/Map_Report.csv:55-60`

The same report also shows the PLIO locations:

- `PLIO_ifm_0` at column 11, channel 0
- `PLIO_ifm_1` at column 11, channel 4
- `PLIO_ofm_0` at column 13, channel 0

Proof:

- `tutorials/proj_aie_tuned/Map_Report.csv:49-52`

This already looks very similar to the paper's picture:

- inputs are injected once per cascade column,
- rows are replicated vertically,
- partial sums move left-to-right across a row,
- graph-to-graph handoff happens through shared memory staging.

There is one more useful physical detail in the layer template: each kernel's main activation buffer is placed on the tile immediately to the west:

- `adf::location<adf::buffer>(kk[idx].in[0]) = bank(tileCol-1, tileRow, ...)` in `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:42-45`

The tuned map report reflects that pattern. For example:

- `qfc1_aie.kk[0]` is at `(10,0)`, while its `buffer_layer2_out -> kk[0].pi0` buffer is mapped at `(9,0)` in `tutorials/proj_aie_tuned/Map_Report.csv:39-44`
- `qfc1_aie.kk[2]` is at `(10,1)`, while the corresponding input-side buffer is at `(9,1)` in `tutorials/proj_aie_tuned/Map_Report.csv:45`
- `qfc2_aie.kk[0]` is at `(13,0)`, while `buffer_layer5_out -> kk[0].pi0` is mapped at `(12,0)` in `tutorials/proj_aie_tuned/Map_Report.csv:43`

### 5. Step-by-step dataflow through the tuned project

What follows is the clearest way to read the project: follow one batch of activations from the notebook input to the final output file.

#### Step 1: the input batch enters through two 128-bit PLIO streams

The tuned application creates:

- two input PLIOs in `tutorials/proj_aie_tuned/app.cpp:17`
- one output PLIO in `tutorials/proj_aie_tuned/app.cpp:24`

The input PLIOs are created from:

- `data/ifm_c0.txt`
- `data/ifm_c1.txt`

in `tutorials/proj_aie_tuned/app.cpp:33-40`.

The output PLIO writes to:

- `data/y_p0.txt`

in `tutorials/proj_aie_tuned/app.cpp:60-67`.

These PLIOs are stream ports, not windows:

- `tutorials/proj_aie_tuned/Work/arch/cfgraph.xml:3-5`

Why are there two input PLIOs? Because tuned `qfc1_aie` splits the 128 input features into two 64-feature slices:

- `IN_FEAT_SLICE = 64` in `tutorials/proj_aie_tuned/src/parameters.h:25`
- two input ports in `tutorials/proj_aie_tuned/src/top_graph.h:17`

For a batch of 8 and 8-bit activations:

- full layer input = `128 x 8 = 1024` bytes
- each PLIO carries half of that = `64 x 8 = 512` bytes
- with a 128-bit PLIO width, each input PLIO sends `512 / 16 = 32` beats

So the tuned project injects the input batch as two 512-byte activation streams.

#### Step 2: those two streams are written into the first shared buffer

The first graph-level buffer is:

- `buffer_layer2_out` in `tutorials/proj_aie_tuned/src/graph_plan.h:14-19`

Its logical size is:

- `{128, 8}` elements of `int8_t`

which is exactly the whole first-layer input tensor for one batch:

- 128 features
- 8 batch elements

The buffer is double-buffered:

- `num_buffers(self.buffer_layer2_out) = 2` in `tutorials/proj_aie_tuned/src/graph_plan.h:19`

That is the ping-pong behavior the paper describes: one buffer can drain while the other refills.

The two PLIO streams write into two disjoint feature ranges of that same buffer:

- `ifm[0] -> buffer_layer2_out.in[0]` with offset `{0,0}` in `tutorials/proj_aie_tuned/src/graph_plan.h:21-26`
- `ifm[1] -> buffer_layer2_out.in[1]` with offset `{64,0}` in `tutorials/proj_aie_tuned/src/graph_plan.h:27-32`

So the first shared buffer reconstructs the full `128 x 8` activation tensor from two incoming 64-feature halves.

This is also where the "DMA" shows up in compiled form. The Vitis interposer report labels these buffer ports as DMA-backed memory/stream adaptors:

- `buffer_layer2_out.in[0]` and `.in[1]`: `adaptor <stream,memory> : Lock Buff DMA`
- `buffer_layer2_out.out[0]` and `.out[1]`: `adaptor <memory,stream> : Lock Buff DMA`

Proof:

- `tutorials/proj_aie_tuned/Work/reports/Interposer_analysis_report.txt:20-23`

So in this framework, DMA is not handwritten code. It is the Vitis-generated implementation of the `shared_buffer` plus `read_access` / `write_access` tiling descriptors.

#### Step 3: the first shared buffer feeds the four compute tiles of `qfc1_aie`

The same `buffer_layer2_out` is read back out in two slices:

- `buffer_layer2_out.out[0] -> qfc1_aie.in1[0]` in `tutorials/proj_aie_tuned/src/graph_plan.h:33-43`
- `buffer_layer2_out.out[1] -> qfc1_aie.in1[1]` in `tutorials/proj_aie_tuned/src/graph_plan.h:44-54`

Each read port uses:

- `tiling_dimension = {8,4}`

which is the transfer shape the memory planner chose for feeding the dense kernels.

Inside the dense layer graph, each column input is broadcast vertically to both rows:

- `connect<>(in1[col], kk[idx].in[0])` in `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:94-99`

That means:

- `in1[0]` feeds `kk[0]` and `kk[2]`
- `in1[1]` feeds `kk[1]` and `kk[3]`

This is the concrete implementation of the paper's statement that inputs are injected once per cascade column and broadcast vertically.

The kernel-side activation interface is buffer-based, not stream-based:

- `input_buffer<data_t>& ifm` in `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h:32`
- `input_buffer<data_t>& ifm` in `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h:50`
- `input_buffer<data_t>& ifm` in `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h:65`
- `input_buffer<data_t>& ifm` in `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h:82`

So the dataflow at this point is:

`PLIO stream -> shared buffer -> DMA-backed read tiler -> kernel input_buffer`

#### Step 4: what each of the four `qfc1_aie` tiles actually computes

The four tiles are arranged as:

- row 0:
  - `kk[0]` at `(10,0)`
  - `kk[1]` at `(11,0)`
- row 1:
  - `kk[2]` at `(10,1)`
  - `kk[3]` at `(11,1)`

Each row is responsible for one 128-feature output slice:

- row 0 produces output features `0..127`
- row 1 produces output features `128..255`

Each column is responsible for one 64-feature input slice:

- left column consumes input features `0..63`
- right column consumes input features `64..127`

So the work split is:

- `kk[0]`: first 64 inputs -> first 128 outputs, partial sums only
- `kk[1]`: second 64 inputs -> first 128 outputs, adds bias/ReLU, emits final output
- `kk[2]`: first 64 inputs -> second 128 outputs, partial sums only
- `kk[3]`: second 64 inputs -> second 128 outputs, adds bias/ReLU, emits final output

That is exactly why the first layer uses both vertical replication and horizontal cascade.

#### Step 5: inside `qfc1_aie`, partial sums move on cascade, not through the shared buffer

The first layer graph builds a horizontal cascade in each row:

- `kk[0] -> kk[1]`
- `kk[2] -> kk[3]`

Proof:

- `connect<cascade>(...)` in `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:108-117`

The kernel types are chosen accordingly:

- left tile in a row = `dense_first`
- right tile in a row = `dense_last`

Proof:

- `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:60-72`

The kernel signatures confirm the channel types:

- `dense_first` writes `output_cascade`
- `dense_last` reads `input_cascade` and writes `output_buffer`

Proof:

- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h:50-52`
- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.h:82-86`

So the first layer uses:

- buffer-based activation input,
- cascade for horizontal partial sums,
- RTP parameters for weights and bias.

The weights and bias do not flow through PLIO. They are runtime parameters connected directly to the kernels:

- `connect<parameter>(wts[idx], async(kk[idx].in[1]))` in `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:80-81`
- `connect<parameter>(bias[row], async(...))` in `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:82-89`

This also matches the paper's claim that weights and bias are RTP-loaded and remain resident on-chip.

#### Step 6: the completed `qfc1_aie` outputs go back into shared memory

Only the rightmost tile of each row emits a real output stream:

- `kk[1] -> out1[0]`
- `kk[3] -> out1[1]`

Proof:

- `tutorials/proj_aie_tuned/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:102-105`

Those two outputs are written into the second graph-level shared buffer:

- `buffer_layer5_out` in `tutorials/proj_aie_tuned/src/graph_plan.h:55-60`

This buffer has logical shape:

- `{256, 8}` elements of `uint8_t`

That is exactly the hidden activation tensor after the first layer:

- 256 hidden features
- 8 batch elements

The two row outputs are written into disjoint halves:

- `qfc1_aie.out1[0] -> buffer_layer5_out.in[0]` with offset `{0,0}` in `tutorials/proj_aie_tuned/src/graph_plan.h:62-71`
- `qfc1_aie.out1[1] -> buffer_layer5_out.in[1]` with offset `{128,0}` in `tutorials/proj_aie_tuned/src/graph_plan.h:72-81`

So the first layer returns its results to shared memory, where the two 128-feature slices are merged into one `256 x 8` activation buffer.

The compiled mapper report shows exactly these edges:

- `buffer_layer5_out_pi0` fed by `qfc1_aie.kk[1]`
- `buffer_layer5_out_pi1` fed by `qfc1_aie.kk[3]`

Proof:

- `tutorials/proj_aie_tuned/Work/reports/AIEMapper_Report.txt:26-27`

#### Step 7: `qfc2_aie` reads the full hidden tensor from shared memory

The second layer reads the entire `256 x 8` hidden buffer back through a single input port:

- `buffer_layer5_out.out[0] -> qfc2_aie.in1[0]` in `tutorials/proj_aie_tuned/src/graph_plan.h:82-92`

`qfc2_aie` is a single tile, so there is:

- no vertical replication,
- no horizontal cascade,
- no feature split.

It simply consumes all 256 hidden features and produces all 64 output features on tile `(13,0)`.

Its activation interface is still buffer-based, and its weights/bias are still RTP parameters, just like the first layer.

#### Step 8: the second layer writes the final output to the third shared buffer, then back to PLIO

The final output path is:

- `qfc2_aie.out1[0] -> buffer_layer6_out.in[0]` in `tutorials/proj_aie_tuned/src/graph_plan.h:100-109`
- `buffer_layer6_out.out[0] -> ofm[0]` in `tutorials/proj_aie_tuned/src/graph_plan.h:110-116`
- `ofm[0] -> PLIO_ofm_0` in `tutorials/proj_aie_tuned/app.cpp:60-67`

`buffer_layer6_out` has logical shape:

- `{64, 8}` elements of `int8_t`

which is the full final output tensor for batch 8.

That is:

- `64 x 8 = 512` bytes total
- at 128 bits per PLIO beat, `512 / 16 = 32` output beats

So the very last step is:

`qfc2 output_buffer -> shared buffer -> DMA-backed read tiler -> output PLIO stream`

### 6. What types of transport are actually used

This is the cleanest final summary for the tuned project:

- graph boundary:
  - PLIO streams
- graph-to-graph handoff:
  - `adf::shared_buffer` with `read_access` / `write_access` tiling
  - double buffering (`num_buffers = 2`)
  - Vitis-generated DMA-backed memory/stream adaptors
- kernel activation ports:
  - `input_buffer<>` / `output_buffer<>`
- within a wide dense layer:
  - `cascade` for horizontal partial sums
- weights and bias:
  - RTP / `connect<parameter>(...)`

So the answer to your original channel question is:

- they do use stream at the PLIO boundary,
- they do use cascade inside a multi-column dense layer,
- they do not use old `input_window` / `output_window` channels in the kernels,
- and the graph-level movement is centered around shared buffers plus tiling descriptors, which Vitis turns into the actual DMA-managed movement.

## Q2. Is the paper's placement algorithm implemented, and does the repo say which tile goes where?

## 1. Yes, there is a real placement pass in the repo

The paper's Section III.C says each graph is modeled as a rectangle whose:

- width = cascade length,
- height = cascade count,

and then a branch-and-bound search minimizes a weighted cost over horizontal hops, vertical hops, and a top-row bias.

That exact overall structure is present in the repo:

- `src/aie4ml/passes/placement.py:21-30` defines `Rect`
- `src/aie4ml/passes/placement.py:108-133` defines the chain placement cost
- `src/aie4ml/passes/placement.py:141-237` implements `_bnb_place_chain(...)`
- `src/aie4ml/passes/placement.py:281-376` implements the optimizer pass `PlaceKernels`
- `src/aie4ml/kernel_registry.py:524-529` defines the footprint as `width = cas_length`, `height = cas_num`

So the answer to "is Section III.C implemented at all?" is yes.

## 2. What the solver actually places

The solver does not place every kernel tile as an unrelated object. It places each layer graph as one rectangular block.

The evidence is:

- `src/aie4ml/passes/placement.py:323-330` creates one `Rect` per node with one input side and one output side
- `src/aie4ml/passes/placement.py:344` adds one `NodeAdapter` per logical node
- `src/aie4ml/passes/placement.py:367-374` writes one `{col,row}` placement per node into `ctx.ir.physical.placements`

Then the generated config stores that per-layer anchor:

- `src/aie4ml/templates/firmware/variants/dense_bias_relu/parameters.h.jinja:14-15` emits `col_placement` and `row_placement`
- `tutorials/proj_aie/src/parameters.h:20-21` stores `L1Cfg` at `(7,0)`
- `tutorials/proj_aie/src/parameters.h:48-49` stores `L2Cfg` at `(9,0)`
- `tutorials/proj_aie_tuned/src/parameters.h:20-21` stores tuned `L1Cfg` at `(10,0)`
- `tutorials/proj_aie_tuned/src/parameters.h:48-49` stores tuned `L2Cfg` at `(13,0)`

Then `top_graph` applies those anchors:

- `tutorials/proj_aie/src/top_graph.h:81-88`
- template source: `src/aie4ml/templates/firmware/top_graph.h.jinja:99-104`

## 3. Where the repo explicitly says which tile each compute kernel goes to

Inside the dense layer graph template, kernel tile coordinates are assigned explicitly:

- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu_graph.h:37-45`
- generated form in `tutorials/proj_aie/src/kernels/dense_bias_relu/dense_bias_relu_graph.h:37-45`

The logic is:

- `tileRow = ROW_START + (idx / CAS_LENGTH)`
- `tileCol = COL_START + (idx % CAS_LENGTH)`
- `adf::location<adf::kernel>(kk[idx]) = adf::tile(tileCol, tileRow);`

So yes, the generated project does explicitly say which compute tile each kernel goes to.

But it does so in two stages:

1. the placement pass chooses the layer anchor `(col,row)`,
2. the layer template expands that anchor into per-kernel tile coordinates.

That is an important nuance. The solver does not output `"kk[3] -> tile (11,1)"` directly. It outputs `"this dense layer starts at (10,0)"`, and the graph template derives the internal rectangle layout.

## 4. What about PLIO and memory buffers?

The repo is much more explicit about compute-kernel placement than PLIO or shared-buffer placement.

For compute kernels:

- yes, aie4ml emits explicit `adf::location<adf::kernel>(...)`
- it also emits explicit bank placements for kernel-local buffers and stacks in `dense_bias_relu_graph.h:42-52`

For PLIO and graph-level shared buffers:

- the generated source expresses connectivity and tiling,
- but it does not directly hard-code a tile location for `PLIO_ifm_0` or `buffer_layer5_out` in the same way.

Actual Vitis output then shows where those resources landed. In the tuned build:

- `tutorials/proj_aie_tuned/Map_Report.csv:49-52` places `PLIO_ifm_0`, `PLIO_ifm_1`, and `PLIO_ofm_0`
- `tutorials/proj_aie_tuned/Map_Report.csv:55-60` shows the compute kernels at `(10,0)`, `(11,0)`, `(10,1)`, `(11,1)`, and `(13,0)`

So the framework explicitly fixes compute tiles, while Vitis finalizes PLIO and shared-buffer physical mapping.

## 5. Important mismatch with the paper

The code does not exactly implement the paper's Equation (2) as written.

Paper Section III.C describes the cost term using both:

- horizontal distance between successive graph endpoints,
- vertical distance between successive graph endpoints,
- and a bias toward smaller top rows.

But the code currently does:

- `horiz += abs(c_out - c_in)`
- `vert += abs(r_in - 0)`

in `src/aie4ml/passes/placement.py:122-129`, and it even leaves a TODO saying:

- `# TODO for direct connections we must use both vertical costs`

So the placement pass is real and matches the paper structurally, but the current vertical-cost term is an approximation, not an exact implementation of Equation (2).

## Q3. How `tutorial_1` measures throughput and latency

### 1. What the notebook actually does

The notebook's performance path is very short:

1. build the AIE project,
2. run prediction with `simulator='aie'`,
3. call `read_aie_report(...)`.

The relevant notebook cells are:

- `tutorials/tutorial_1.ipynb:1401` for `compare_bit_exact(hls_model, aie_model, sim_mode = 'aie')`
- `tutorials/tutorial_1.ipynb:1541-1543` for `report = read_aie_report(aie_model)`
- `tutorials/tutorial_1.ipynb:10370-10372` for the same sequence on `aie_model_tuned`

The important point is that the notebook does **not** call a trace target or a profiling target. Its measurement path is the helper function `read_aie_report(...)`.

### 2. What `read_aie_report(...)` really computes

The implementation is in `src/aie4ml/simulation.py`.

It does two things:

- reads timing from simulator output files,
- reads static mapping text from the compiler reports.

The relevant code path is:

- `read_aie_report(...)` in `src/aie4ml/simulation.py:20-47`
- `_analyze_aie_out_interval(...)` in `src/aie4ml/simulation.py:64-95`
- `_parse_timing(...)` in `src/aie4ml/simulation.py:98-123`
- `compute_ops(...)` in `src/aie4ml/simulation.py:140-148`
- `_infer_batch_size(...)` in `src/aie4ml/simulation.py:151-159`

The logic is:

1. open `aiesimulator_output/data/y_p*.txt`,
2. parse time stamps of the form `T ...`,
3. look for `TLAST`,
4. compute the time difference between consecutive `TLAST` events,
5. call that the output interval,
6. compute throughput as `ops_per_batch / output_interval_ns`.

So the reported throughput is **derived** from simulator output cadence. It is not read from a hardware counter, and it is not extracted from a trace graph.

### 3. What counts as "throughput" in this helper

The numerator comes from dense-layer arithmetic only:

- `compute_ops(...)` adds `2 * n_in * n_out` for every dense layer in `src/aie4ml/simulation.py:143-148`

For `tutorial_1`, that is:

- `qfc1`: `2 * 128 * 256 = 65536` ops
- `qfc2`: `2 * 256 * 64 = 32768` ops
- total per inference = `98304` ops

With `BATCH = 8`, the helper uses:

- `98304 * 8 = 786432` ops per output batch

Then `read_aie_report(...)` computes:

- `Avg_GOPs = ops_per_batch / avg_output_interval_ns`
- `Min_GOPs = ops_per_batch / max_output_interval_ns`
- `Max_GOPs = ops_per_batch / min_output_interval_ns`

Proof:

- `src/aie4ml/simulation.py:33-42`

This means the denominator includes the actual simulated graph execution cadence, including on-chip data movement, while the numerator counts only dense arithmetic. So the reported GOP/s is a dense-op throughput metric, not a complete cycle-accounted hardware utilization metric.

### 4. What counts as "latency" here, and what does not

This is the subtle but important point.

What the helper explicitly measures is:

- **steady-state output interval**, meaning the time between one completed output packet and the next

That is what `_parse_timing(...)` returns:

- `"""Return TLAST-to-TLAST intervals (in nanoseconds)."""` in `src/aie4ml/simulation.py:99`

What it does **not** explicitly measure is:

- end-to-end first-inference latency,
- input-to-first-output latency,
- per-layer latency,
- PL/PS overhead.

You can see this directly in the parser logic:

- `last_tlast_time` starts as `None`
- the first `TLAST` only initializes that state
- only the **second and later** `TLAST` events produce a measured interval

Proof:

- `src/aie4ml/simulation.py:102-123`

So if we are being precise:

- the notebook **does capture throughput-related timing**
- the notebook **does not report true first-result latency**
- the field named `output_interval` is closer to an initiation interval / steady-state batch period than to full end-to-end latency

### 5. Concrete evidence from the generated simulator files

The current tuned project already contains the simulator output file that `read_aie_report(...)` uses:

- `tutorials/proj_aie_tuned/aiesimulator_output/data/y_p0.txt`

That file contains:

- timestamp lines such as `T 2646400 ps`
- data words
- `TLAST` markers

Proof:

- `tutorials/proj_aie_tuned/aiesimulator_output/data/y_p0.txt:1`
- `tutorials/proj_aie_tuned/aiesimulator_output/data/y_p0.txt:64`
- `tutorials/proj_aie_tuned/aiesimulator_output/data/y_p0.txt:129`

Because `ITERS = 10`, there are 10 output packets and therefore 9 TLAST-to-TLAST intervals:

- `tutorials/tutorial_1.ipynb:434` for `ITERS = 10`

Recomputing the helper's metric on the tuned project file gives:

- 9 intervals
- min output interval = `486.4 ns`
- max output interval = `489.6 ns`
- average output interval = `487.467 ns`

Using the helper's own operation count formula, that becomes:

- average throughput = about `1.613 GOP/s`

For comparison, the untuned project currently in tree gives:

- average output interval = `557.511 ns`
- average throughput = about `1.411 GOP/s`

So the tuned tutorial does show a measurable throughput improvement, but again this is based on output cadence from simulation output files, not on trace instrumentation.

### 6. What the helper returns besides timing

`read_aie_report(...)` also returns `AIE_info`, but that comes from:

- `Work/reports/app_mapping_analysis_report.txt`

Proof:

- `src/aie4ml/simulation.py:50-61`

That report is mainly static placement/mapping information, for example:

- block mapping and utilization in `tutorials/proj_aie_tuned/Work/reports/app_mapping_analysis_report.txt:14-23`
- port mapping in `tutorials/proj_aie_tuned/Work/reports/app_mapping_analysis_report.txt:30-63`
- shared-buffer mapping in `tutorials/proj_aie_tuned/Work/reports/app_mapping_analysis_report.txt:115-119`

So the notebook's `report` object is a mixture of:

- dynamic timing derived from `aiesimulator_output/data/y_p*.txt`
- static mapping information from `app_mapping_analysis_report.txt`

### 7. Is trace used here?

Not in the current notebook flow.

The generated Makefile does support optional trace and profile modes:

- `make profile` runs `aiesimulator --online -wdb -text --profile`
- `make trace` runs `aiesimulator --online -wdb -text`

Proof:

- `tutorials/proj_aie_tuned/Makefile:91-98`
- template source: `src/aie4ml/templates/firmware/Makefile.jinja:91-98`

But the notebook never calls those targets. More importantly, the compiled project shows that the trace graph is empty:

- `event_trace_graph.trace_ports` is `{}` in `tutorials/proj_aie_tuned/Work/reports/compiler_report.json:2466-2467`
- `Nodes in trace graph = 0` in `tutorials/proj_aie_tuned/Work/logs/aie_hw.log:167`

There is also explicit evidence that the router found no trace nets:

- `Found 9 nets and 0 trace nets.` in `tutorials/proj_aie_tuned/log:464`

So your suspicion is correct: the tutorial is **not** using trace to measure throughput or latency.

One nuance: the build still sets `--num-trace-streams=4`, but that only reserves trace capacity. It does not mean a trace graph was actually instantiated. The real evidence is the empty trace graph and zero trace nets.

### 8. Bottom line on measurement

For `tutorial_1`, the measurement story is:

- correctness is checked by `compare_bit_exact(...)`
- throughput is derived from AIE simulator output cadence
- the timing metric actually exposed is `TLAST-to-TLAST` output interval
- the helper does not expose true first-inference latency
- no event trace graph is used in the notebook's current flow

## Final takeaway

For `tutorial_1`, the dataflow is best understood as:

- PLIO streams at the graph boundary,
- shared-buffer / memtile-style staging between graph stages,
- buffer-based compute ports on the kernels,
- cascade only inside a horizontally chained dense layer,
- RTP/parameter ports for weights and bias.

For placement, the repo does implement the paper's branch-and-bound style layer placement, and it does explicitly assign compute kernels to tiles in the generated ADF graph. The main caveat is that the solver chooses per-layer anchors, not per-kernel placements directly, and the exact cost function in code is not fully identical to the paper's Equation (2).
