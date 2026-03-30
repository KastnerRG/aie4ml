import argparse

import matplotlib.pyplot as plt
import numpy as np

from cas_num_len import (
    DEFAULT_ITERS,
    DEFAULT_R_M,
    PLATFORM,
    VITIS_SETTINGS,
    actual_tile_sizes,
    api_tag,
    format_api,
    needs_execution,
    RESULTS_ROOT,
    RUNS_ROOT,
    source_vitis,
)
from sweep_utils import build_dense_aie_model, measure_latency_ns

DEFAULT_APIS = [(4, 8, 8), (4, 16, 8)]
DEFAULT_API_BLOCKS = (8, 16, 32)
DEFAULT_RATIO_POINTS = ((2, 2), (2, 4))


def parse_pair(value):
    cleaned = value.replace("x", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {value!r}")
    return int(parts[0]), int(parts[1])


def parse_triple(value):
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated values, got {value!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def api_output_root(tiling):
    return RUNS_ROOT / f"cas_num_len_fixed_tiles__{api_tag(tiling)}"


def api_output_png(tiling):
    return RESULTS_ROOT / f"cas_num_len_fixed_tiles__{api_tag(tiling)}" / f"lat_line__cas_num_len_fixed_tiles__{api_tag(tiling)}.png"


def point_output_dir(output_root, api_blocks, r_k, r_n):
    return output_root / f"rk_{r_k}_rn_{r_n}__L_{api_blocks}"


def point_metadata(tiling, r_m, r_k, r_n, api_blocks, output_root):
    batch = r_m * tiling["tile_m"]
    in_features = api_blocks * tiling["tile_k"]
    out_features = api_blocks * tiling["tile_n"]
    tile_sizes = actual_tile_sizes(tiling, r_m, r_k, r_n)
    if api_blocks % r_k != 0 or api_blocks % r_n != 0:
        return None
    return {
        "batch": batch,
        "in_features": in_features,
        "out_features": out_features,
        "cas_length": api_blocks // r_k,
        "cas_num": api_blocks // r_n,
        "tile_m": tile_sizes["tile_m"],
        "tile_k": tile_sizes["tile_k"],
        "tile_n": tile_sizes["tile_n"],
        "output_dir": point_output_dir(output_root, api_blocks, r_k, r_n),
    }


def collect_output_dirs(tiling, r_m, api_blocks_values, ratio_points, output_root):
    output_dirs = []
    for api_blocks in api_blocks_values:
        for r_k, r_n in ratio_points:
            metadata = point_metadata(tiling, r_m, r_k, r_n, api_blocks, output_root)
            if metadata is not None:
                output_dirs.append(metadata["output_dir"])
    return output_dirs


def safe_label(label):
    return "".join(char if char.isalnum() else "_" for char in label).strip("_")


def save_series_csv(output_root, x_labels, series):
    np.savez(output_root / "latencies_ns.npz", **{safe_label(label): values for label, values in series})
    header = ["x_value", *[safe_label(label) + "_us" for label, _ in series]]
    rows = [",".join(header)]
    for index, x_label in enumerate(x_labels):
        values = [f'"{x_label}"']
        for _, latencies_ns in series:
            value = latencies_ns[index]
            values.append("nan" if np.isnan(value) else f"{value / 1000.0:.6f}")
        rows.append(",".join(values))
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def plot_series(output_png, x_labels, series, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap("tab10")
    ymax = 0.0
    for index, (label, latencies_ns) in enumerate(series):
        latencies_us = latencies_ns / 1000.0
        ax.plot(
            range(len(x_labels)),
            latencies_us,
            marker="o",
            linewidth=2.0,
            color=colors(index % colors.N),
            label=label,
        )
        for x_pos, value in enumerate(latencies_us):
            if np.isnan(value):
                ax.text(x_pos, 0.0, "fail", ha="center", va="bottom", fontsize=8)
                continue
            ymax = max(ymax, float(value))
            ax.text(x_pos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_xlabel("global layer size in API blocks")
    ax.set_ylabel("latency (us)")
    ax.set_title(title)
    ax.legend()
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apis", nargs="+", type=parse_triple, default=DEFAULT_APIS)
    parser.add_argument("--api-blocks", nargs="+", type=int, default=list(DEFAULT_API_BLOCKS))
    parser.add_argument("--ratio-points", nargs="+", type=parse_pair, default=list(DEFAULT_RATIO_POINTS))
    parser.add_argument("--r-m", type=int, default=DEFAULT_R_M)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    jobs = []
    all_output_dirs = []
    for tile_m, tile_k, tile_n in args.apis:
        tiling = {"tile_m": tile_m, "tile_k": tile_k, "tile_n": tile_n}
        output_root = api_output_root(tiling)
        output_png = api_output_png(tiling)
        jobs.append((tiling, output_root, output_png))
        all_output_dirs.extend(collect_output_dirs(tiling, args.r_m, args.api_blocks, args.ratio_points, output_root))

    if needs_execution(all_output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    for tiling, output_root, output_png in jobs:
        print(f"Generating fixed-tile comparison for api={format_api(tiling)} rM={args.r_m}")
        output_root.mkdir(parents=True, exist_ok=True)
        x_labels = []
        series = []
        for r_k, r_n in args.ratio_points:
            latencies_ns = np.full(len(args.api_blocks), np.nan)
            for index, api_blocks in enumerate(args.api_blocks):
                metadata = point_metadata(tiling, args.r_m, r_k, r_n, api_blocks, output_root)
                batch = args.r_m * tiling["tile_m"]
                in_features = api_blocks * tiling["tile_k"]
                out_features = api_blocks * tiling["tile_n"]
                if index >= len(x_labels):
                    x_labels.append(f"L={api_blocks}\n({batch},{in_features},{out_features})")
                if metadata is None:
                    print(
                        f"[invalid] api={format_api(tiling)} rK={r_k:>2} rN={r_n:>2} "
                        f"L={api_blocks:>2} size=({batch},{in_features},{out_features})"
                    )
                    continue
                latency_ns, status = measure_latency_ns(
                    metadata["output_dir"],
                    lambda path, metadata=metadata, in_features=in_features, out_features=out_features, batch=batch: build_dense_aie_model(
                        in_features,
                        out_features,
                        batch,
                        args.iters,
                        PLATFORM,
                        path,
                        "cas_num_len_fixed_tiles",
                        parallelism={"cas_num": metadata["cas_num"], "cas_length": metadata["cas_length"]},
                        tiling=tiling,
                    ),
                    in_features,
                    batch,
                    rerun_failed=args.rerun_failed,
                )
                latencies_ns[index] = latency_ns
                summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
                print(
                    f"[{status}] api={format_api(tiling)} rK={r_k:>2} rN={r_n:>2} "
                    f"L={api_blocks:>2} size=({batch},{in_features},{out_features}) "
                    f"tile=({metadata['tile_m']},{metadata['tile_k']},{metadata['tile_n']}) "
                    f"cas_length={metadata['cas_length']:>2} cas_num={metadata['cas_num']:>2} latency_ns={summary}"
                )
            tile_sizes = actual_tile_sizes(tiling, args.r_m, r_k, r_n)
            series.append((f"(rK,rN)=({r_k},{r_n}) tile=({tile_sizes['tile_m']},{tile_sizes['tile_k']},{tile_sizes['tile_n']})", latencies_ns))
        save_series_csv(output_root, x_labels, series)
        plot_series(
            output_png,
            x_labels,
            series,
            f"Fixed Tile Latency vs Layer Size, api={format_api(tiling)}, rM={args.r_m}",
        )


if __name__ == "__main__":
    main()
