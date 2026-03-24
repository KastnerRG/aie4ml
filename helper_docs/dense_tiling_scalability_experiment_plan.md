# Dense Tiling Scalability Experiment Plan

## Goal

Plan a set of dense-layer experiments that expose when multi-tile tiling helps, when it stops helping, and what layer sizes are large enough to make a single AIE-ML tile meaningfully busy in the current `aie4ml` backend.

This note answers two concrete questions:

1. Should the batch size baseline be `4` or `8` for your use case?
2. What `(in, out)` sizes make sense as the starting point for a "single-tile reasonably utilized" dense-layer study?

## Short Answers

### 1. Batch size: use `8` as the main baseline, keep `4` as a deliberate underutilization control

For the current backend and the default AIE-ML int8 dense tile shape, `batch=4` is structurally underutilized.

Why:

- The current default int8 AIE-ML kernel tiling order starts with `tile_m=4, tile_k=8, tile_n=8`.
- The resolver pads the independent extent to `2 * tile_m`.
- Therefore with `tile_m=4`, the effective batch granularity is `8`.
- So `batch=4` is padded to `8`, which means 50% independent-dimension utilization before you even consider other overheads.

In this codebase, `batch=8` is the correct small-batch baseline for the default int8 kernel.

Important nuance:

- This is not a universal AIE-ML statement.
- It is a statement about **this backend and its current kernel schedule**.
- If you intentionally switch to `tile_m=8` later, then `batch=8` would itself pad to `16`, so the best baseline for that tile shape would change.

### 2. Single-tile starting point: start with `batch=8`, `in=128`, `out=128`

For the current backend, `128 x 128` is the strongest default starting point for a one-tile int8 dense-layer study.

Why:

- It exactly fills the current one-tile int8 weight-memory budget: `128 * 128 * 1 byte = 16384 bytes`.
- It matches the default alignment structure well: with `tile_k=8` and `tile_n=8`, the effective aligned slice quanta are `16` features on both input and output sides.
- It gives enough work to amortize launch/padding/scheduling overhead much better than very small layers.
- It is square, so it avoids conflating poor tile utilization with an extreme input/output aspect ratio.

If you want one additional full-one-tile control, use:

- `64 x 256`
- `256 x 64`

These also fill the one-tile int8 weight budget, but they are intentionally skewed and are better used as aspect-ratio controls, not as the primary starting point.

## What "fully utilize one tile" should mean here

There are **two different meanings** of single-tile utilization, and they should not be confused.

### A. Compute-side utilization

AMD states that an AIE-ML tile can sustain `256 INT8 MACs/cycle`.

The AMD AI Engine API also defines `aie::mmul<M,K,N>` as a blocked matrix multiply with:

- `size_A = M * K`
- `size_B = K * N`
- `size_C = M * N`

So for the default int8 AIE-ML tile shape used first by this backend, `4 x 8 x 8` corresponds to:

- `4 * 8 * 8 = 256` MACs per blocked multiply

Inference from those two facts:

- A single `mmul<4,8,8>` block matches the published `256 INT8 MACs/cycle` peak at the tile level.
- But a whole dense layer only approaches that peak if it provides enough repeated `mmul` work to amortize control, buffering, padding, IO, and graph overhead.

So a tiny layer can be **theoretically compatible** with the tile primitive while still being a bad utilization experiment.

### B. Memory-side utilization

In this backend, one of the strongest practical limits is the per-tile weight memory budget.

For VEK280 in the local device catalog:

- `WeightMemBytes = 16384`

For int8 weights, that means the one-tile dense-layer weight matrix budget is approximately:

- `in_slice * out_slice <= 16384`

For one tile (`cas_num=1`, `cas_length=1`) this gives the useful boundary:

- `128 x 128 = 16384` bytes: exactly full
- `64 x 256 = 16384` bytes: exactly full
- `256 x 64 = 16384` bytes: exactly full
- `128 x 256 = 32768` bytes: no longer one-tile feasible at int8

That memory boundary is the cleanest practical definition of a "full" one-tile int8 dense layer in this backend.

## Verified Backend Facts From This Repo

### Dense-kernel schedule and alignment

The current dense kernel enforces:

- per-tile weight bytes `<= 16384`
- `IN_FEAT_SLICE % (2 * K) == 0`
- `OUT_FEAT_SLICE % (2 * N) == 0`
- `padded_independent_extent % (2 * M) == 0`

In the default int8 AIE-ML tiling path, the first supported tile is:

- `(tile_m, tile_k, tile_n) = (4, 8, 8)`

So the default aligned quanta are effectively:

- batch / independent extent multiple of `8`
- input slice multiple of `16`
- output slice multiple of `16`

### Batch padding rule

The resolver explicitly uses:

- `_aligned_batch_size(batch, tile_m) = align_up(batch, 2 * tile_m)`

So with the default `tile_m=4`:

- `batch=4 -> padded to 8`
- `batch=8 -> stays 8`

This is the concrete reason `batch=4` is not a good baseline for the current kernel.

### Parallelism search preference

When the resolver searches `(cas_num, cas_length)` pairs, it explicitly prefers candidates that:

- use more of the per-tile weight memory
- avoid output slices that are much larger than input slices
- avoid alignment padding waste

That is consistent with using square or near-square layers as the main baseline and using highly skewed shapes as targeted controls.

## Practical Single-Tile Layer Regimes

Assume int8 weights, default tiling `(4,8,8)`, and `batch=8`.

| Shape `(in,out)` | Weight bytes | One-tile fill ratio | Ideal tile MAC cycles at batch 8* | Use in study |
|---|---:|---:|---:|---|
| `16 x 16` | 256 | 1.6% | 8 | Too small for main study; keep only as a sanity check |
| `32 x 32` | 1024 | 6.25% | 32 | Still overhead-dominated |
| `64 x 64` | 4096 | 25% | 128 | Small-layer regime |
| `64 x 128` | 8192 | 50% | 256 | Mid-size regime |
| `128 x 64` | 8192 | 50% | 256 | Mid-size regime |
| `128 x 128` | 16384 | 100% | 512 | Best default single-tile starting point |
| `64 x 256` | 16384 | 100% | 512 | Full-tile but output-heavy control |
| `256 x 64` | 16384 | 100% | 512 | Full-tile but input-heavy control |
| `128 x 256` | 32768 | 200% | 1024 | Crosses into multi-tile-required regime |

\* Ideal tile MAC cycles here means `batch * in * out / 256` and assumes perfect peak compute utilization. Real FIFO latency will be higher because the measured graph latency includes IO, buffering, and scheduling overhead.

## Recommended Experiment Matrix

Keep the first pass simple:

- precision: int8 x int8
- tiling: default `(4,8,8)` unless an experiment explicitly varies it
- batch baseline: `8`
- compare one tile against explicit multi-tile overrides using `cas_num` and/or `cas_length`

### Experiment 0: Batch calibration

Purpose:

- Verify that `batch=4` is a bad primary baseline in this backend.

Run these shapes:

- `64 x 64`
- `128 x 128`
- `128 x 256`

Run each at:

- `batch=4`
- `batch=8`

Recommended tile settings:

- one tile: `(cas_num, cas_length) = (1,1)` where feasible
- output split: `(2,1)`
- 2D split: `(2,2)` for the larger layer

Expected outcome:

- `batch=4` should show clearly worse per-tile efficiency because it is padded to `8` for the default `tile_m=4`.

### Experiment 1: Fix input, sweep output

Purpose:

- This is your stated starting direction.
- Measure when output expansion begins to justify output-side tiling.

Recommended baseline:

- fix `in = 128`
- sweep `out = {32, 64, 128, 256, 512}`
- use `batch = 8`

Tile comparisons:

- `1x1`
- `2x1`
- `4x1`

Interpretation:

- `out <= 128` remains one-tile feasible and gives a clean one-vs-many comparison.
- `out > 128` moves past the one-tile weight-memory boundary and becomes the multi-tile-required regime.

Expected outcome:

- Small outputs (`32`, `64`) should show weak speedup from extra tiles.
- `128` is the most important one-tile-vs-multi-tile crossover point.
- `256` and above should show tiling becoming necessary rather than optional.

### Experiment 2: Fix output, sweep input

Purpose:

- Symmetric check for input-side splitting (`cas_length`).

Recommended baseline:

- fix `out = 128`
- sweep `in = {32, 64, 128, 256, 512}`
- use `batch = 8`

Tile comparisons:

- `1x1`
- `1x2`
- `1x4`

Expected outcome:

- Similar crossover behavior to Experiment 1, but this isolates the effect of input-side partitioning.

### Experiment 3: Constant one-tile memory footprint, vary aspect ratio

Purpose:

- Separate "more work" from "different shape".

Recommended shapes:

- `64 x 256`
- `128 x 128`
- `256 x 64`

All three have the same int8 one-tile weight footprint: `16384` bytes.

Tile comparisons:

- `1x1`
- `2x1`
- `1x2`
- `2x2`

Expected outcome:

- If one shape scales worse, that is evidence of shape-driven inefficiency, not just size.

### Experiment 4: Near-boundary and beyond-boundary scaling

Purpose:

- Measure diminishing returns around the transition where one tile stops being enough.

Recommended shapes:

- near boundary: `96 x 128`, `112 x 128`, `128 x 128`
- beyond boundary: `128 x 192`, `128 x 256`, `256 x 256`

Tile comparisons:

- near boundary: `1x1`, `2x1`, `1x2`, `2x2`
- beyond boundary: whichever multi-tile settings fit cleanly, starting with `2x1` and `1x2`

Expected outcome:

- Diminishing returns should be easiest to see near `128 x 128`, where one tile is already near fully used.
- Above the boundary, extra tiles help more because they are serving required capacity, not just optional parallelism.

### Experiment 5: Optional tile-shape sensitivity

Purpose:

- Only do this after the first four experiments are stable.
- Test whether the conclusions are backend-default-specific or robust to tile-shape changes.

Suggested cases:

- default: `(4,8,8)` with `batch=8`
- alternate: `(8,8,8)` with `batch=16`

Reason for matching batch to tile shape:

- the backend pads the independent extent to `2 * tile_m`
- so if `tile_m=8`, a fair batch baseline is `16`, not `8`

## Metrics To Record

For each run, record at least:

- FIFO latency cycles from `fifo_latency.json`
- wall-clock latency in ns/us if reported
- tile count = `cas_num * cas_length`
- effective MACs/cycle:
  - `batch * in * out / fifo_latency_cycles`
- speedup vs 1 tile:
  - `latency_1tile / latency_ntiles`
- scaling efficiency:
  - `speedup / tile_count`
- per-tile efficiency relative to peak:
  - `effective_MACs_per_cycle / (256 * tile_count)`
- tile-memory fill ratio:
  - `weight_tile_bytes / 16384`
- padding efficiency for batch:
  - `actual_batch / padded_batch`

Important caution:

- The reported FIFO latency is an end-to-end graph measure, not a pure compute-kernel cycle count.
- So these utilization metrics are best treated as **effective backend utilization**, not raw silicon utilization.
- That is exactly what you want for tiling-scalability experiments.

## Recommended Starting Campaign

If you want the smallest high-value first campaign, do this first:

1. Fix `batch=8`, default `(4,8,8)`.
2. Run `in=128`, `out={32,64,128,256}`.
3. For each, compare:
   - `1x1`
   - `2x1`
   - `4x1`
4. Add one constant-footprint control set:
   - `64x256`, `128x128`, `256x64`
5. Add one batch check:
   - `128x128` at `batch=4` vs `batch=8`

If those results are clean, then the next step is the full input/output/aspect-ratio grid.

## Final Recommendations

- Use `batch=8` as the primary small-batch baseline for the current backend.
- Keep `batch=4` only as a control that intentionally demonstrates underutilization.
- Use `128x128` as the default "single-tile reasonably full" starting point.
- Use `64x64` as the small-layer regime and `64x256` / `256x64` as aspect-ratio controls.
- Use `128x256` as the first clear multi-tile-required transition point.
- Keep default tile shape `(4,8,8)` fixed until you have the first scaling curves; otherwise you will mix tiling effects with kernel-shape effects.

## References

### AMD primary sources

- AMD AI Engine Technology page: https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html
  - states `256 INT8 MACs/cycle per tile in AIE-ML`
  - states `64 kB` local data memory per AIE-ML tile
  - states `512 kB` memory tiles
- AMD AI Engine API User Guide, Matrix Multiplication: https://download.amd.com/docnav/aiengine/xilinx2025_2/aiengine_api/aie_api/doc/group__group__mmul.html
  - defines `aie::mmul<M,K,N>` and `size_A=M*K`, `size_B=K*N`, `size_C=M*N`
  - lists supported AIE-ML int8 mmul shapes including `2x8x8`, `4x8x8`, and `8x8x8`

### Local repo evidence

- `src/aie4ml/kernel_registry.py`
  - default AIE-ML int8 tiling options begin with `(4, 8, 8)`
- `src/aie4ml/passes/resolve_registry.py`
  - `_aligned_batch_size(batch, tile_m)` pads to `2 * tile_m`
  - parallelism selection prefers high tile-memory fill and low padding waste
- `src/aie4ml/templates/nnet_utils/dense_bias_relu/dense_bias_relu.cpp`
  - enforces `weight bytes <= 16384`
  - enforces divisibility by `2*K`, `2*N`, and `2*M`
- `src/aie4ml/aie_devices.json`
  - VEK280 catalog entry exposes `WeightMemBytes = 16384`, `MaxMemTileInPorts = 8`, `MaxMemTileOutPorts = 8`
