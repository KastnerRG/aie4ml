import argparse
import os
from pathlib import Path

from sweep_utils import (
    build_dense_aie_model,
    measure_latency_ns,
    needs_execution,
    plot_heatmap,
    run_heatmap_sweep,
    save_heatmap_csv,
    seed_everything,
    source_vitis,
    sweep_factors,
    tagged_output_dir,
)

IN_FEATURES = 128
OUT_FEATURES = 128
BATCH = 8
ITERS = 1
TILE_M = 4
TILE_K_MAX = 32
TILE_N_MAX = 32
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tmp/tools/Xilinx/2025.2/Vitis/.settings64-Vitis.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "tile_k_n"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

seed_everything()


def measure_point(tile_n, tile_k, output_root, rerun_failed):
    output_dir = output_root / f"tile_k_{tile_k}_tile_n_{tile_n}"
    return measure_latency_ns(
        output_dir,
        lambda path: build_dense_aie_model(
            IN_FEATURES,
            OUT_FEATURES,
            BATCH,
            ITERS,
            PLATFORM,
            path,
            "tile_k_n",
            tiling={"tile_m": TILE_M, "tile_k": tile_k, "tile_n": tile_n},
        ),
        IN_FEATURES,
        BATCH,
        rerun_failed=rerun_failed,
    )


def describe_point(tile_n, tile_k, summary, status):
    return (
        f"[{status}] tile_m={TILE_M:>2} tile_k={tile_k:>2} tile_n={tile_n:>2} "
        f"latency_ns={summary}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    tile_ks = sweep_factors(TILE_K_MAX)
    tile_ns = sweep_factors(TILE_N_MAX)
    x_labels = [str(tile_n) for tile_n in tile_ns]
    y_labels = [str(tile_k) for tile_k in tile_ks]
    output_dirs = [
        args.output_root / f"tile_k_{tile_k}_tile_n_{tile_n}"
        for tile_k in tile_ks
        for tile_n in tile_ns
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    latencies_ns = run_heatmap_sweep(
        tile_ns,
        tile_ks,
        args.output_root,
        args.rerun_failed,
        measure_point,
        describe_point,
        lambda latencies: save_heatmap_csv(args.output_root, "tile_k\\tile_n", x_labels, y_labels, latencies),
    )
    output_png = RESULTS_ROOT / f"lat_hm__tile_k_n__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}.png"
    plot_heatmap(output_png, x_labels, y_labels, "tile_n", "tile_k", latencies_ns)


if __name__ == "__main__":
    main()
