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
    tuple_labels,
)

IN_FEATURES = 128
OUT_FEATURES = 128
BATCH = 8
ITERS = 1
CAS_LENGTH_MAX = 32
CAS_NUM_MAX = 32
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / f"cas_num_len__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

seed_everything()


def measure_point(cas_num, cas_length, output_root, rerun_failed):
    output_dir = output_root / f"cas_num_{cas_num}_cas_length_{cas_length}"
    return measure_latency_ns(
        output_dir,
        lambda path: build_dense_aie_model(
            IN_FEATURES,
            OUT_FEATURES,
            BATCH,
            ITERS,
            PLATFORM,
            path,
            "cas_num_len",
            parallelism={"cas_num": cas_num, "cas_length": cas_length},
        ),
        IN_FEATURES,
        BATCH,
        rerun_failed=rerun_failed,
    )


def describe_point(cas_num, cas_length, summary, status):
    return (
        f"[{status}] cas_num={cas_num:>2} cas_length={cas_length:>2} "
        f"tile_out={OUT_FEATURES // cas_num:>3} tile_in={IN_FEATURES // cas_length:>3} "
        f"latency_ns={summary}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    cas_lengths = sweep_factors(CAS_LENGTH_MAX)
    cas_nums = sweep_factors(CAS_NUM_MAX)
    x_labels = tuple_labels(cas_nums, [OUT_FEATURES // cas_num for cas_num in cas_nums])
    y_labels = tuple_labels(cas_lengths, [IN_FEATURES // cas_length for cas_length in cas_lengths])
    output_dirs = [
        args.output_root / f"cas_num_{cas_num}_cas_length_{cas_length}"
        for cas_length in cas_lengths
        for cas_num in cas_nums
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    latencies_ns = run_heatmap_sweep(
        cas_nums,
        cas_lengths,
        args.output_root,
        args.rerun_failed,
        measure_point,
        describe_point,
        lambda latencies: save_heatmap_csv(args.output_root, "cas_len\\cas_num", x_labels, y_labels, latencies),
    )
    output_png = RESULTS_ROOT / f"lat_hm__cas_num_len__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}.png"
    plot_heatmap(output_png, x_labels, y_labels, "(cas_num, tile_out_size)", "(cas_len, tile_in_size)", latencies_ns)


if __name__ == "__main__":
    main()
