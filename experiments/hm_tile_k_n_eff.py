import argparse
from pathlib import Path

from sweep_utils import load_rows_csv, plot_tile_k_n_eff_heatmap

EXPERIMENT_ROOT = Path(__file__).resolve().parent / "runs" / "exp_tile_k_n_eff__dt_x8_k8__batch_8"
CSV_PATH = EXPERIMENT_ROOT / "tile_k_n_eff.csv"
OUTPUT_PNG = Path(__file__).resolve().parent / "results" / "hm__tile_k_n_eff__dt_x8_k8__batch_8.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    args = parser.parse_args()

    rows = load_rows_csv(args.csv_path)
    plot_tile_k_n_eff_heatmap(args.output_png, rows, "Tile API Efficiency Heatmap, dt=i8xi8, batch=8")


if __name__ == "__main__":
    main()
