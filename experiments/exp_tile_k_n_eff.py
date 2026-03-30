import argparse
import os
from pathlib import Path

import numpy as np

from sweep_utils import (
    build_dense_aie_model,
    measure_latency_ns,
    needs_execution,
    save_rows_csv,
    seed_everything,
    source_vitis,
)

BATCH = 8
INPUT_DTYPE = "i8"
WEIGHT_DTYPE = "i8"
OUTPUT_DTYPE = "i8"
CAS_NUM = 1
CAS_LENGTH = 1
ITERS = 1
T_VALUES = range(3, 9)
BASE_SHAPES = [(16 * t, 32) for t in T_VALUES]
EXTENDED_SHAPES = [(dim, 128) for dim in range(48, 129, 16)]
API_TILINGS = [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8)]
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "exp_tile_k_n_eff__dt_x8_k8__batch_8"
CSV_NAME = "tile_k_n_eff.csv"

seed_everything()


def shape_pairs():
    pairs = []
    for in_features, out_features in [*BASE_SHAPES, *EXTENDED_SHAPES]:
        pairs.append((in_features, out_features))
        if in_features != out_features:
            pairs.append((out_features, in_features))
    return pairs


def orientation(in_features, out_features):
    return "in>out" if in_features > out_features else "in<out"


def point_output_dir(output_root, in_features, out_features, tiling):
    tile_m, tile_k, tile_n = tiling
    return output_root / f"b{BATCH}_i{in_features}_o{out_features}_m{tile_m}_k{tile_k}_n{tile_n}"


def api_label(tiling):
    return f"({tiling[0]},{tiling[1]},{tiling[2]})"


def shape_label(batch, in_features, out_features):
    return f"({batch},{in_features},{out_features})"


def macs_per_inference(batch, in_features, out_features):
    return int(batch * in_features * out_features)


def gops_per_second(macs, latency_ns):
    if latency_ns is None or np.isnan(latency_ns) or latency_ns <= 0:
        return np.nan
    return (2.0 * macs) / latency_ns


def throughput_fps(latency_ns):
    if latency_ns is None or np.isnan(latency_ns) or latency_ns <= 0:
        return np.nan
    return 1e9 / latency_ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    experiments = [
        (in_features, out_features, tiling)
        for in_features, out_features in shape_pairs()
        for tiling in API_TILINGS
    ]
    output_dirs = [point_output_dir(args.output_root, in_features, out_features, tiling) for in_features, out_features, tiling in experiments]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    rows = []
    for in_features, out_features, tiling in experiments:
        tile_m, tile_k, tile_n = tiling
        output_dir = point_output_dir(args.output_root, in_features, out_features, tiling)
        latency_ns, status = measure_latency_ns(
            output_dir,
            lambda path, in_features=in_features, out_features=out_features, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n: build_dense_aie_model(
                in_features,
                out_features,
                BATCH,
                ITERS,
                PLATFORM,
                path,
                "exp_tile_k_n_eff",
                parallelism={"cas_num": CAS_NUM, "cas_length": CAS_LENGTH},
                tiling={"tile_m": tile_m, "tile_k": tile_k, "tile_n": tile_n},
                input_dtype=INPUT_DTYPE,
                weight_dtype=WEIGHT_DTYPE,
                output_dtype=OUTPUT_DTYPE,
            ),
            in_features,
            BATCH,
            rerun_failed=args.rerun_failed,
        )
        macs = macs_per_inference(BATCH, in_features, out_features)
        ops = 2 * macs
        row = {
            "batch": BATCH,
            "in_features": in_features,
            "out_features": out_features,
            "shape": shape_label(BATCH, in_features, out_features),
            "orientation": orientation(in_features, out_features),
            "api": api_label(tiling),
            "tile_m": tile_m,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "cas_num": CAS_NUM,
            "cas_length": CAS_LENGTH,
            "dtype": f"{INPUT_DTYPE}x{WEIGHT_DTYPE}",
            "macs": macs,
            "ops": ops,
            "latency_ns": latency_ns,
            "latency_us": np.nan if np.isnan(latency_ns) else latency_ns / 1000.0,
            "gops_per_s": gops_per_second(macs, latency_ns),
            "gmacs_per_s": np.nan if np.isnan(latency_ns) or latency_ns <= 0 else macs / latency_ns,
            "throughput_fps": throughput_fps(latency_ns),
            "status": "success" if not np.isnan(latency_ns) else "failed",
            "output_dir": str(output_dir),
        }
        rows.append(row)
        save_rows_csv(
            args.output_root / CSV_NAME,
            rows,
            [
                "batch",
                "in_features",
                "out_features",
                "shape",
                "orientation",
                "api",
                "tile_m",
                "tile_k",
                "tile_n",
                "cas_num",
                "cas_length",
                "dtype",
                "macs",
                "ops",
                "latency_ns",
                "latency_us",
                "gops_per_s",
                "gmacs_per_s",
                "throughput_fps",
                "status",
                "output_dir",
            ],
        )
        summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
        print(
            f"[{status}] shape={row['shape']} api={row['api']} "
            f"macs={macs} latency_ns={summary} gops_per_s={row['gops_per_s'] if not np.isnan(row['gops_per_s']) else np.nan}"
        )


if __name__ == "__main__":
    main()
