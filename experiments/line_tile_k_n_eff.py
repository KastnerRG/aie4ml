import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from sweep_utils import load_rows_csv

EXPERIMENT_ROOT = Path(__file__).resolve().parent / "runs" / "exp_tile_k_n_eff__dt_x8_k8__batch_8"
CSV_PATH = EXPERIMENT_ROOT / "tile_k_n_eff.csv"
OUTPUT_PNG = Path(__file__).resolve().parent / "results" / "line__tile_k_n_eff__dt_x8_k8__batch_8.png"


def plot_tile_k_n_eff_lines(output_png, rows, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    api_order = ["(4,8,8)", "(2,8,8)", "(2,16,8)", "(4,8,4)", "(4,16,4)", "(4,16,8)"]
    colors = {label: plt.get_cmap("tab10")(index) for index, label in enumerate(api_order)}
    styles = [("in>out", ":", "in > out"), ("in<out", "-", "in < out")]
    x_rows = sorted(rows, key=lambda row: int(row["macs"]))
    x_labels = []
    x_positions = {}
    for row in x_rows:
        macs = int(row["macs"])
        if macs in x_positions:
            continue
        left, right = sorted((int(row["in_features"]), int(row["out_features"])))
        x_positions[macs] = len(x_labels)
        x_labels.append(f"{macs / 1000.0:.1f}k\n({row['batch']},{left}*{right})")
    plotted = set()
    for api in api_order:
        for orientation, linestyle, suffix in styles:
            series = [row for row in rows if row["api"] == api and row["orientation"] == orientation and row["status"] == "success"]
            if not series:
                continue
            series.sort(key=lambda row: x_positions[int(row["macs"])])
            ax.plot(
                [x_positions[int(row["macs"])] for row in series],
                [float(row["gops_per_s"]) for row in series],
                color=colors[api],
                linestyle=linestyle,
                linewidth=3.0 if api == "(4,8,8)" else 1.8,
                marker="o",
                label=f"{api} {suffix}" if (api, orientation) not in plotted else None,
            )
            plotted.add((api, orientation))
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_xlabel("MACs = (batch,in*out)")
    ax.set_ylabel("efficiency (GOps/s)")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
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
    plot_tile_k_n_eff_lines(args.output_png, rows, "Tile API Efficiency, dt=i8xi8, batch=8")


if __name__ == "__main__":
    main()
