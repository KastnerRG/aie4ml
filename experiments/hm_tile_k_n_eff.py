import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sweep_utils import load_rows_csv

EXPERIMENT_ROOT = Path(__file__).resolve().parent / "runs" / "exp_tile_k_n_eff__dt_x8_k8__batch_8"
CSV_PATH = EXPERIMENT_ROOT / "tile_k_n_eff.csv"
OUTPUT_PNG = Path(__file__).resolve().parent / "results" / "hm__tile_k_n_eff__dt_x8_k8__batch_8.png"


def plot_tile_k_n_eff_heatmap(output_png, rows, title):
    ordered_rows = sorted(rows, key=lambda row: (int(row["macs"]), 0 if row["orientation"] == "in>out" else 1, row["shape"]))
    x_labels = []
    seen_shapes = set()
    for row in ordered_rows:
        if row["shape"] not in seen_shapes:
            x_labels.append(row["shape"])
            seen_shapes.add(row["shape"])
    y_labels = ["(4,8,8)", "(2,8,8)", "(2,16,8)", "(4,8,4)", "(4,16,4)", "(4,16,8)"]
    values = np.full((len(y_labels), len(x_labels)), np.nan)
    x_index = {label: index for index, label in enumerate(x_labels)}
    y_index = {label: index for index, label in enumerate(y_labels)}
    for row in rows:
        if row["status"] != "success":
            continue
        values[y_index[row["api"]], x_index[row["shape"]]] = float(row["gops_per_s"])
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(16, 4.8))
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_xlabel("layer shape (batch,in,out)")
    ax.set_ylabel("api tile (M,K,N)")
    ax.set_title(title)
    finite_values = values[np.isfinite(values)]
    vmax = float(finite_values.max()) if finite_values.size else 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if not np.isnan(values[row, col]):
                color = "black" if vmax and values[row, col] >= 0.6 * vmax else "white"
                ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color=color, fontsize=8)
    box_linewidth = 3
    y0 = -0.5
    y1 = len(y_labels) - 0.5
    boundaries = {-0.5, len(x_labels) - 0.5}
    for start_col in range(0, len(x_labels), 2):
        width = min(2, len(x_labels) - start_col)
        x0 = start_col - 0.5
        x1 = start_col + width - 0.5
        ax.plot([x0, x1], [y0, y0], color="black", linewidth=box_linewidth)
        ax.plot([x0, x1], [y1, y1], color="black", linewidth=box_linewidth)
        boundaries.update([x0, x1])
    for x in sorted(boundaries):
        ax.plot([x, x], [y0, y1], color="black", linewidth=box_linewidth)
    fig.colorbar(image, ax=ax, label="efficiency (GOps/s)")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    args = parser.parse_args()

    rows = load_rows_csv(args.csv_path)
    plot_tile_k_n_eff_heatmap(args.output_png, rows, "Tile API Efficiency Heatmap, dt=i8xi8, batch=8")


if __name__ == "__main__":
    main()
