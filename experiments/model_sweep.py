import argparse
import os
from pathlib import Path

import numpy as np

from sweep_utils import (
    build_dense_chain_aie_model,
    measure_latency_ns,
    needs_execution,
    plot_grouped_bars,
    save_grouped_bar_csv,
    seed_everything,
    source_vitis,
)

BATCH = 8
IN_FEATURES = 128
OUT_FEATURES = 128
NUM_LAYERS = 6
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
IO_ROUTE_MODES = ["auto", "direct", "memtile"]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / f"model_sweep__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}__layers_{NUM_LAYERS}"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

seed_everything()


def measure_point(io_mode, use_relu, output_root, rerun_failed):
    output_dir = output_root / f"io_route_{io_mode}__relu_{int(use_relu)}"
    return measure_latency_ns(
        output_dir,
        lambda path: build_dense_chain_aie_model(
            IN_FEATURES,
            OUT_FEATURES,
            NUM_LAYERS,
            BATCH,
            ITERS,
            PLATFORM,
            path,
            "model_sweep",
            io_mode=io_mode,
            use_relu=use_relu,
        ),
        IN_FEATURES,
        BATCH,
        rerun_failed=rerun_failed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    output_dirs = [
        args.output_root / f"io_route_{io_mode}__relu_{int(use_relu)}"
        for io_mode in IO_ROUTE_MODES
        for use_relu in (False, True)
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    latencies = {False: [], True: []}
    for io_mode in IO_ROUTE_MODES:
        for use_relu in (False, True):
            latency_ns, status = measure_point(io_mode, use_relu, args.output_root, args.rerun_failed)
            latencies[use_relu].append(latency_ns)
            summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
            print(f"[{status}] io_route={io_mode:>7} relu={'on' if use_relu else 'off':>3} latency_ns={summary}")

    series = [
        ("without relu", np.array(latencies[False], dtype=float)),
        ("with relu", np.array(latencies[True], dtype=float)),
    ]
    save_grouped_bar_csv(args.output_root, IO_ROUTE_MODES, series)
    output_png = RESULTS_ROOT / f"lat_bar__model_sweep__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}__layers_{NUM_LAYERS}.png"
    plot_grouped_bars(
        output_png,
        IO_ROUTE_MODES,
        series,
        "io_route",
        "latency (us)",
        f"6-layer dense chain, size=({BATCH},{IN_FEATURES},{OUT_FEATURES}), dtype=i8",
    )


if __name__ == "__main__":
    main()
