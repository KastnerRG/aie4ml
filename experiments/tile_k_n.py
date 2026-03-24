import argparse
import os
from pathlib import Path

import numpy as np

from sweep_utils import api_tilings, build_dense_aie_model, measure_latency_ns, needs_execution, plot_bar, save_bar_csv, seed_everything, source_vitis

BATCH = 8
IN_FEATURES = 128
OUT_FEATURES = 128
INPUT_DTYPE = "i8"
WEIGHT_DTYPE = "i8"
ALLOWED_TILE_MS = {2, 4}
CAS_LENGTH = 1
CAS_NUM = 1
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / f"tile_k_n__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}"

seed_everything()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    metadata_rows = [
        {"tile_m": tile_m, "tile_k": tile_k, "tile_n": tile_n, "cas_length": CAS_LENGTH, "cas_num": CAS_NUM}
        for tile_m, tile_k, tile_n in api_tilings((INPUT_DTYPE, WEIGHT_DTYPE))
        if tile_m in ALLOWED_TILE_MS
    ]
    x_values = [f"({meta['tile_m']},{meta['tile_k']},{meta['tile_n']})" for meta in metadata_rows]
    output_dirs = [
        args.output_root / f"tile_m_{meta['tile_m']}_tile_k_{meta['tile_k']}_tile_n_{meta['tile_n']}"
        for meta in metadata_rows
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    latencies_ns = np.full(len(metadata_rows), np.nan)
    for index, meta in enumerate(metadata_rows):
        output_dir = args.output_root / f"tile_m_{meta['tile_m']}_tile_k_{meta['tile_k']}_tile_n_{meta['tile_n']}"
        latency_ns, status = measure_latency_ns(
            output_dir,
            lambda path, meta=meta: build_dense_aie_model(
                IN_FEATURES,
                OUT_FEATURES,
                BATCH,
                ITERS,
                PLATFORM,
                path,
                "tile_k_n",
                parallelism={"cas_num": CAS_NUM, "cas_length": CAS_LENGTH},
                tiling={"tile_m": meta["tile_m"], "tile_k": meta["tile_k"], "tile_n": meta["tile_n"]},
            ),
            IN_FEATURES,
            BATCH,
            rerun_failed=args.rerun_failed,
        )
        latencies_ns[index] = latency_ns
        summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
        print(f"[{status}] cas_num={CAS_NUM} cas_length={CAS_LENGTH} tile_api={x_values[index]} latency_ns={summary}")
        save_bar_csv(args.output_root, x_values, latencies_ns, metadata_rows)

    title = f"Latency Per Inference, layer=({BATCH},{IN_FEATURES},{OUT_FEATURES})"
    output_png = RESULTS_ROOT / f"lat_hm__tile_k_n__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}.png"
    plot_bar(output_png, x_values, latencies_ns, "api tile (M,K,N)", "latency (us)", title)


if __name__ == "__main__":
    main()
