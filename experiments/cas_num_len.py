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
)

DEFAULT_IN_FEATURES = 128
DEFAULT_OUT_FEATURES = 128
DEFAULT_R_M = 2
DEFAULT_ITERS = 1
DEFAULT_RATIO_VALUES = (2, 4, 8, 16)
DEFAULT_TILING = {"tile_m": 4, "tile_k": 8, "tile_n": 8}
DEFAULT_BATCH = DEFAULT_R_M * DEFAULT_TILING["tile_m"]
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

seed_everything()


def api_tag(tiling):
    return f"api_{tiling['tile_m']}_{tiling['tile_k']}_{tiling['tile_n']}"


def default_output_root(batch, in_features, out_features, tiling, separate_api_dir=False):
    if separate_api_dir:
        return RUNS_ROOT / f"cas_num_len__{api_tag(tiling)}__size_{batch}_{in_features}_{out_features}"
    return RUNS_ROOT / f"cas_num_len__size_{batch}_{in_features}_{out_features}"


def default_output_png(batch, in_features, out_features, tiling, separate_api_dir=False):
    if separate_api_dir:
        return RESULTS_ROOT / f"cas_num_len__{api_tag(tiling)}" / f"lat_hm__cas_num_len__size_{batch}_{in_features}_{out_features}.png"
    return RESULTS_ROOT / f"lat_hm__cas_num_len__size_{batch}_{in_features}_{out_features}.png"


def save_heatmap_csv(output_root, header, x_labels, y_labels, latencies_ns):
    np.save(output_root / "latencies_ns.npy", latencies_ns)
    rows = [",".join([header, *x_labels])]
    for label, row in zip(y_labels, latencies_ns / 1000.0):
        values = ["nan" if np.isnan(value) else f"{value:.6f}" for value in row]
        rows.append(",".join([label, *values]))
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def plot_heatmap(output_png, x_labels, y_labels, x_title, y_title, latencies_ns, title):
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
    ax.set_title(title)
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


def format_api(tiling):
    return f"({tiling['tile_m']},{tiling['tile_k']},{tiling['tile_n']})"


def actual_tile_sizes(tiling, r_m, r_k, r_n):
    return {
        "tile_m": r_m * tiling["tile_m"],
        "tile_k": r_k * tiling["tile_k"],
        "tile_n": r_n * tiling["tile_n"],
    }


def ratio_point_metadata(in_features, out_features, batch, tiling, r_m, r_k, r_n, output_root):
    tile_sizes = actual_tile_sizes(tiling, r_m, r_k, r_n)
    if batch != tile_sizes["tile_m"]:
        raise ValueError(
            f"batch={batch} must equal r_m * M_api = {r_m} * {tiling['tile_m']} = {tile_sizes['tile_m']}"
        )
    if in_features % tile_sizes["tile_k"] != 0 or out_features % tile_sizes["tile_n"] != 0:
        return None
    cas_length = in_features // tile_sizes["tile_k"]
    cas_num = out_features // tile_sizes["tile_n"]
    return {
        "r_k": r_k,
        "r_n": r_n,
        "cas_length": cas_length,
        "cas_num": cas_num,
        "tile_m": tile_sizes["tile_m"],
        "tile_k": tile_sizes["tile_k"],
        "tile_n": tile_sizes["tile_n"],
        "output_dir": output_root / f"rk_{r_k}_rn_{r_n}",
    }


def ratio_output_dirs(output_root, in_features, out_features, batch, tiling, r_m, ratio_values):
    output_dirs = []
    for r_n in ratio_values:
        for r_k in ratio_values:
            metadata = ratio_point_metadata(in_features, out_features, batch, tiling, r_m, r_k, r_n, output_root)
            if metadata is not None:
                output_dirs.append(metadata["output_dir"])
    return output_dirs


def run_cas_num_len_heatmap(
    output_root,
    output_png,
    in_features=DEFAULT_IN_FEATURES,
    out_features=DEFAULT_OUT_FEATURES,
    batch=DEFAULT_BATCH,
    iters=DEFAULT_ITERS,
    ratio_values=DEFAULT_RATIO_VALUES,
    r_m=DEFAULT_R_M,
    tiling=None,
    rerun_failed=False,
    source_environment=True,
    platform=PLATFORM,
    vitis_settings=VITIS_SETTINGS,
):
    tiling = DEFAULT_TILING.copy() if tiling is None else {key: int(value) for key, value in tiling.items()}
    ratio_values = [int(value) for value in ratio_values]
    x_labels = [str(r_k) for r_k in ratio_values]
    y_labels = [str(r_n) for r_n in ratio_values]
    output_dirs = ratio_output_dirs(output_root, in_features, out_features, batch, tiling, r_m, ratio_values)

    output_root.mkdir(parents=True, exist_ok=True)
    if source_environment and needs_execution(output_dirs, rerun_failed):
        source_vitis(vitis_settings)

    def measure_point(r_k, r_n, point_output_root, point_rerun_failed):
        metadata = ratio_point_metadata(in_features, out_features, batch, tiling, r_m, r_k, r_n, point_output_root)
        if metadata is None:
            return np.nan, "invalid"
        return measure_latency_ns(
            metadata["output_dir"],
            lambda path: build_dense_aie_model(
                in_features,
                out_features,
                batch,
                iters,
                platform,
                path,
                "cas_num_len",
                parallelism={"cas_num": metadata["cas_num"], "cas_length": metadata["cas_length"]},
                tiling=tiling,
            ),
            in_features,
            batch,
            rerun_failed=point_rerun_failed,
        )

    def describe_point(r_k, r_n, summary, status):
        metadata = ratio_point_metadata(in_features, out_features, batch, tiling, r_m, r_k, r_n, output_root)
        requested_tiles = actual_tile_sizes(tiling, r_m, r_k, r_n)
        if metadata is None:
            return (
                f"[{status}] rK={r_k:>2} rN={r_n:>2} "
                f"tile=({requested_tiles['tile_m']:>2},{requested_tiles['tile_k']:>3},{requested_tiles['tile_n']:>3}) "
                f"size=({batch},{in_features},{out_features}) api={format_api(tiling)} latency_ns={summary}"
            )
        return (
            f"[{status}] rK={r_k:>2} rN={r_n:>2} "
            f"tile=({metadata['tile_m']:>2},{metadata['tile_k']:>3},{metadata['tile_n']:>3}) "
            f"cas_length={metadata['cas_length']:>2} cas_num={metadata['cas_num']:>2} "
            f"api={format_api(tiling)} latency_ns={summary}"
        )

    latencies_ns = run_heatmap_sweep(
        ratio_values,
        ratio_values,
        output_root,
        rerun_failed,
        measure_point,
        describe_point,
        lambda latencies: save_heatmap_csv(output_root, "rN\\rK", x_labels, y_labels, latencies),
    )
    plot_heatmap(
        output_png,
        x_labels,
        y_labels,
        "rK = K_tile / K_api",
        "rN = N_tile / N_api",
        latencies_ns,
        f"Latency Per Inference, size=({batch},{in_features},{out_features}), api={format_api(tiling)}, rM={r_m}",
    )
    return latencies_ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=DEFAULT_IN_FEATURES)
    parser.add_argument("--out-features", type=int, default=DEFAULT_OUT_FEATURES)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--r-m", type=int, default=DEFAULT_R_M)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--ratio-values", nargs="+", type=int, default=list(DEFAULT_RATIO_VALUES))
    parser.add_argument("--tile-m", type=int, default=DEFAULT_TILING["tile_m"])
    parser.add_argument("--tile-k", type=int, default=DEFAULT_TILING["tile_k"])
    parser.add_argument("--tile-n", type=int, default=DEFAULT_TILING["tile_n"])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--separate-api-dir", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()
    tiling = {"tile_m": args.tile_m, "tile_k": args.tile_k, "tile_n": args.tile_n}
    batch = args.r_m * tiling["tile_m"] if args.batch is None else args.batch
    output_root = args.output_root or default_output_root(
        batch,
        args.in_features,
        args.out_features,
        tiling,
        separate_api_dir=args.separate_api_dir,
    )
    output_png = args.output_png or default_output_png(
        batch,
        args.in_features,
        args.out_features,
        tiling,
        separate_api_dir=args.separate_api_dir,
    )
    run_cas_num_len_heatmap(
        output_root=output_root,
        output_png=output_png,
        in_features=args.in_features,
        out_features=args.out_features,
        batch=batch,
        iters=args.iters,
        ratio_values=args.ratio_values,
        r_m=args.r_m,
        tiling=tiling,
        rerun_failed=args.rerun_failed,
    )


if __name__ == "__main__":
    main()
