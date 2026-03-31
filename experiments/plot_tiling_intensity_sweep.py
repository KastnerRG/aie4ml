import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sweep_utils import load_rows_csv

EXPERIMENT_ROOT = Path(__file__).resolve().parent / "runs" / "tiling_intensity_sweep"
CSV_PATH = EXPERIMENT_ROOT / "tiling_intensity_sweep.csv"
ONE_LAYER_CSV_PATH = EXPERIMENT_ROOT / "tiling_intensity_sweep_one_layer.csv"
OUTPUT_PNG = Path(__file__).resolve().parent / "results" / "line__tiling_intensity_sweep.png"


def latency_us(row):
    value = row.get("latency_us", "")
    if value in ("", "nan"):
        return np.nan
    return float(value)


def plot_tiling_intensity_sweep(output_png, rows, single_layer_rows):
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
        }
    )
    ordered = sorted(rows, key=lambda row: (int(row["cas_length"]), int(row["cas_num"])))
    single_lookup = {
        (int(row["cas_length"]), int(row["cas_num"])): row
        for row in single_layer_rows
    }
    x = list(range(len(ordered)))
    x_labels = [f"({row['cas_length']},{row['cas_num']})" for row in ordered]
    latencies = [latency_us(row) for row in ordered]
    single_layer_latencies = [
        latency_us(single_lookup.get((int(row["cas_length"]), int(row["cas_num"])), {}))
        for row in ordered
    ]
    total_tiles_per_layer = [float(row["cas_product"]) for row in ordered]
    cols_per_layer = [float(row["cas_length"]) for row in ordered]
    rows_per_layer = [float(row["cas_num"]) for row in ordered]

    first = ordered[0]
    title = (
        f"Cost of exhausting AIE columns"
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(5.0, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.0], "hspace": 0.05},
    )

    for ax in (ax_top, ax_bottom):
        ax.axvspan(1.5, len(x) - 0.5, color="#e6e6e6", alpha=0.9, zorder=0)
        ax.axvline(1.5, color="black", linestyle=":", linewidth=1.5, zorder=1)
        ax.set_xlim(-0.5, len(x) - 0.5)

    ax_top.plot(
        x,
        latencies,
        color="#d62728",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label=f"{first['num_layers']} layers",
        zorder=3,
    )
    ax_top.plot(
        x,
        single_layer_latencies,
        color="#ff9896",
        marker="o",
        linewidth=2.0,
        markersize=6,
        linestyle="--",
        label="single layer",
        zorder=4,
    )
    ax_top.set_ylabel("latency (us)")
    ax_top.set_title(title, pad=2)

    finite_latencies = [value for value in latencies + single_layer_latencies if np.isfinite(value)]
    if finite_latencies:
        ymax = max(finite_latencies)
        ax_top.set_ylim(0.0, ymax * 1.15)

    ymin, ymax = ax_top.get_ylim()
    label_y = min(max(2.0, ymin + 0.05 * (ymax - ymin)), ymax - 0.3)
    ax_top.text(
        0.5,
        label_y,
        "placement\nbands = 1",
        ha="center",
        va="bottom",
        fontsize=15,
        color="#1f1f1f",
    )
    ax_top.text(
        2.5,
        label_y,
        "placement\nbands = 2",
        ha="center",
        va="bottom",
        fontsize=15,
        color="#1f1f1f",
    )

    ax_top.grid(axis="y", alpha=0.25)
    ax_top.legend(loc="upper left")

    bar_width = 0.30
    ax_bottom.bar(
        [pos - bar_width for pos in x],
        total_tiles_per_layer,
        width=bar_width,
        color="#7f7f7f",
        label="total tiles/layer",
        zorder=3,
    )
    ax_bottom.bar(
        x,
        cols_per_layer,
        width=bar_width,
        color="#1f77b4",
        label="cols/layer",
        zorder=3,
    )
    ax_bottom.bar(
        [pos + bar_width for pos in x],
        rows_per_layer,
        width=bar_width,
        color="#9ecae1",
        label="rows/layer",
        zorder=3,
    )
    ax_bottom.set_ylabel("tiles per layer")
    ax_bottom.set_yticks([0, 2, 3, 4, 6, 12])
    ax_bottom.tick_params(axis="y", labelsize=12)
    ax_bottom.set_xticks(x, labels=x_labels)
    ax_bottom.set_xlabel(r"$(P_K, Q_K)$ compute tile cols, rows per layer")
    ax_bottom.grid(axis="y", alpha=0.25)
    ax_bottom.legend(loc="upper right")

    fig.tight_layout(h_pad=0.1)
    fig.subplots_adjust(top=0.95)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--single-layer-csv-path", type=Path, default=ONE_LAYER_CSV_PATH)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    args = parser.parse_args()

    rows = load_rows_csv(args.csv_path)
    single_layer_rows = load_rows_csv(args.single_layer_csv_path)
    plot_tiling_intensity_sweep(args.output_png, rows, single_layer_rows)


if __name__ == "__main__":
    main()
