import argparse
import os
from pathlib import Path

import numpy as np

from sweep_utils import TILE_API_TILINGS, api_tilings, build_dense_aie_model, default_output_dtype, dtype_bits, measure_latency_ns, needs_execution, plot_bar, save_bar_csv, seed_everything, source_vitis

BATCH = 8
IN_FEATURES = 64
OUT_FEATURES = 64
ALLOWED_TILE_MS = {2, 4}
CAS_LENGTH = 1
CAS_NUM = 1
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DTYPE_PAIRS = [dtype_pair for dtype_pair in TILE_API_TILINGS if dtype_pair != ("i8", "i16")]
SKIP_TILINGS = {("i16", "i16"): {(4, 2, 8)}}

seed_everything()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    dtype_pairs = DTYPE_PAIRS
    all_output_dirs = []
    experiments = []
    for input_dtype, weight_dtype in dtype_pairs:
        dtype_tag = f"dt_x{dtype_bits(input_dtype)}_k{dtype_bits(weight_dtype)}"
        exp_name = f"tile_k_n__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}__{dtype_tag}"
        output_root = args.runs_root / exp_name
        output_dtype = default_output_dtype(input_dtype, weight_dtype)
        metadata_rows = [
            {
                "tile_m": tile_m,
                "tile_k": tile_k,
                "tile_n": tile_n,
                "cas_length": CAS_LENGTH,
                "cas_num": CAS_NUM,
                "output_dtype": output_dtype,
            }
            for tile_m, tile_k, tile_n in api_tilings((input_dtype, weight_dtype))
            if tile_m in ALLOWED_TILE_MS and (tile_m, tile_k, tile_n) not in SKIP_TILINGS.get((input_dtype, weight_dtype), set())
        ]
        x_values = [f"({meta['tile_m']},{meta['tile_k']},{meta['tile_n']})" for meta in metadata_rows]
        output_dirs = [
            output_root / f"tile_m_{meta['tile_m']}_tile_k_{meta['tile_k']}_tile_n_{meta['tile_n']}_out_{dtype_bits(meta['output_dtype'])}"
            for meta in metadata_rows
        ]
        experiments.append((input_dtype, weight_dtype, exp_name, output_root, metadata_rows, x_values, output_dirs))
        all_output_dirs.extend(output_dirs)

    if needs_execution(all_output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    for input_dtype, weight_dtype, exp_name, output_root, metadata_rows, x_values, output_dirs in experiments:
        print(f"{exp_name}")
        output_root.mkdir(parents=True, exist_ok=True)
        latencies_ns = np.full(len(metadata_rows), np.nan)
        for index, meta in enumerate(metadata_rows):
            output_dir = output_dirs[index]
            latency_ns, status = measure_latency_ns(
                output_dir,
                lambda path, meta=meta, input_dtype=input_dtype, weight_dtype=weight_dtype: build_dense_aie_model(
                    IN_FEATURES,
                    OUT_FEATURES,
                    BATCH,
                    ITERS,
                    PLATFORM,
                    path,
                    "tile_k_n",
                    parallelism={"cas_num": CAS_NUM, "cas_length": CAS_LENGTH},
                    tiling={"tile_m": meta["tile_m"], "tile_k": meta["tile_k"], "tile_n": meta["tile_n"]},
                    input_dtype=input_dtype,
                    weight_dtype=weight_dtype,
                    output_dtype=meta["output_dtype"],
                ),
                IN_FEATURES,
                BATCH,
                rerun_failed=args.rerun_failed,
            )
            latencies_ns[index] = latency_ns
            summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
            print(f"[{status}] dt=({input_dtype},{weight_dtype})->{meta['output_dtype']} cas_num={CAS_NUM} cas_length={CAS_LENGTH} tile_api={x_values[index]} latency_ns={summary}")
            save_bar_csv(output_root, x_values, latencies_ns, metadata_rows)

        title = f"Latency Per Inference, layer=({BATCH},{IN_FEATURES},{OUT_FEATURES}), dt=({input_dtype},{weight_dtype})->{metadata_rows[0]['output_dtype']}"
        output_png = RESULTS_ROOT / f"lat_hm__{exp_name}.png"
        plot_bar(output_png, x_values, latencies_ns, "api tile (M,K,N)", "latency (us)", title)


if __name__ == "__main__":
    main()
