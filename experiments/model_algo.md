# Model Ranking Algorithm

## Pseudocode

```text
INPUT: model layers, API=(4,8,8), usable_cols=31

for each layer:
  candidates = []
  for PK in 1..8:
    for PN in 1..8:
      raw_k = ceil(in_features / PK)
      raw_n = ceil(out_features / PN)
      if raw_k is not 32-bit aligned: reject
      k = align_up(raw_k, backend_K_alignment)
      n = align_up(raw_n, backend_N_alignment)
      if k > 128 or n > 128: reject
      if k * n * weight_bytes > weight_mem: reject
      add candidate (PK, PN, tile_workload=k*n, padding=k*n*PK*PN - raw_work)
  prune dominated candidates

beam = {empty_state}
for each layer:
  extend every beam state with every legal candidate
  used_width = sum(PK) + gaps_between_layers
  reject if used_width + minimum_future_width > 31
  bottleneck = max(layer tile_workloads)
  workload_imbalance = sum(bottleneck - layer tile_workload)
  score state lexicographically by:
    1. smaller bottleneck tile_workload
    2. smaller workload imbalance
    3. smaller width above soft cap
    4. smaller total padding
    5. smaller width gap below soft cap
    6. smaller max_rows
    7. larger used_width (up to soft cap)
    8. larger sum(PK)
    9. larger sum(PK*PN)
  keep best B states

rank complete states by the same score
execute a few diverse ranked candidates
```

Notes:

```text
workload_imbalance = how uneven the layer workloads are
                  = sum(max_workload - layer_workload)
                  = 0 means all layers are perfectly balanced

the 9 ranking terms are not added with weights into one scalar
score = (
  bottleneck,
  workload_imbalance,
  width_above_soft_cap,
  total_padding,
  width_gap_below_soft_cap,
  max_rows,
  -used_width_clipped_to_soft_cap,
  -sum(PK),
  -sum(PK*PN)
)

states are compared left-to-right:
first minimize term 1; if tied, minimize term 2; if tied, term 3; etc.
```

## Example: `vae_lhc_large`

Model:

```text
64x128 | 128x64 | 64x12 | 12x64 | 64x128 | 128x64
soft cap = 31
```

Important legal low-workload candidates:

```text
L0 64x128: 4x4 (512), 2x8 (512)
L1 128x64: 8x2 (512)
L2 64x12:  2x1 (512)   ; 1x1 exists but is worse (1024)
L3 12x64:  1x2 (512)   ; 1x1 exists but is worse (1024)
L4 64x128: 2x8 (512), 4x4 (512)
L5 128x64: 8x2 (512)
```

Step-by-step:

```text
1. Pick the smallest legal per-layer workloads that also match each other.
2. Layers 2 and 3 force padding, so their best balanced choices are 2x1 and 1x2.
3. Combine candidates and keep only width <= 31.
4. Score by bottleneck first, then imbalance, then routing/width penalties.
```

Best balanced combinations found:

```text
rank 0: 4x4 | 8x2 | 2x1 | 1x2 | 2x8 | 8x2
  width = 4+8+2+1+2+8 + 5 gaps = 30
  bottleneck = 512
  imbalance = 0
  padding = 512

rank 1: 2x8 | 8x2 | 2x1 | 1x2 | 4x4 | 8x2
  width = 30
  bottleneck = 512
  imbalance = 0
  padding = 512

rank 2: 4x4 | 8x2 | 2x1 | 1x2 | 4x4 | 4x4
  width = 28
  bottleneck = 512
  imbalance = 0
  padding = 512
```

Interpretation:

```text
the algorithm is trying to make every layer run at about the same tile workload,
while still using as many columns as possible without hurting routing too much.
```
