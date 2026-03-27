import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sweep_utils import (
    build_dense_aie_model,
    measure_latency_ns,
    needs_execution,
    seed_everything,
    source_vitis,
    sweep_factors,
    tuple_labels,
)

IN_FEATURES = 128
OUT_FEATURES = 128
BATCH = 8
ITERS = 1
CAS_LENGTH_MAX = 8
CAS_NUM_MAX = 8
FIXED_TILING = {"tile_m": 4, "tile_k": 8, "tile_n": 8}
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / f"cas_num_len__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

seed_everything()


def save_heatmap_csv(output_root, header, x_labels, y_labels, latencies_ns):
    np.save(output_root / "latencies_ns.npy", latencies_ns)
    rows = [",".join([header, *x_labels])]
    for label, row in zip(y_labels, latencies_ns / 1000.0):
        values = ["nan" if np.isnan(value) else f"{value:.6f}" for value in row]
        rows.append(",".join([label, *values]))
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def plot_heatmap(output_png, x_labels, y_labels, x_title, y_title, latencies_ns):
    latencies_us = latencies_ns / 1000.0
    masked = np.ma.masked_invalid(latencies_us)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(masked, origin="lower", cmap=cmap)
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title("Latency Per Inference")
    for row in range(latencies_us.shape[0]):
        for col in range(latencies_us.shape[1]):
            if not np.isnan(latencies_us[row, col]):
                ax.text(col, row, f"{latencies_us[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="latency (us)")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def run_heatmap_sweep(x_values, y_values, output_root, rerun_failed, measure_point, describe_point, save_progress):
    latencies = np.full((len(y_values), len(x_values)), np.nan)
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            latency_ns, status = measure_point(x_value, y_value, output_root, rerun_failed)
            latencies[row, col] = latency_ns
            summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
            print(describe_point(x_value, y_value, summary, status))
            save_progress(latencies)
    return latencies


def measure_point(cas_length, cas_num, output_root, rerun_failed):
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
            tiling=FIXED_TILING,
        ),
        IN_FEATURES,
        BATCH,
        rerun_failed=rerun_failed,
    )


def describe_point(cas_length, cas_num, summary, status):
    return (
        f"[{status}] cas_length={cas_length:>2} cas_num={cas_num:>2} "
        f"tile_in={IN_FEATURES // cas_length:>3} tile_out={OUT_FEATURES // cas_num:>3} "
        f"api=(4,8,8) latency_ns={summary}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    cas_lengths = sweep_factors(CAS_LENGTH_MAX)
    cas_nums = sweep_factors(CAS_NUM_MAX)
    x_labels = tuple_labels(cas_lengths, [IN_FEATURES // cas_length for cas_length in cas_lengths])
    y_labels = tuple_labels(cas_nums, [OUT_FEATURES // cas_num for cas_num in cas_nums])
    output_dirs = [
        args.output_root / f"cas_num_{cas_num}_cas_length_{cas_length}"
        for cas_num in cas_nums
        for cas_length in cas_lengths
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    latencies_ns = run_heatmap_sweep(
        cas_lengths,
        cas_nums,
        args.output_root,
        args.rerun_failed,
        measure_point,
        describe_point,
        lambda latencies: save_heatmap_csv(args.output_root, "cas_num\\cas_len", x_labels, y_labels, latencies),
    )
    output_png = RESULTS_ROOT / f"lat_hm__cas_num_len__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}.png"
    plot_heatmap(output_png, x_labels, y_labels, "(cas_len, tile_in_size)", "(cas_num, tile_out_size)", latencies_ns)


if __name__ == "__main__":
    main()
