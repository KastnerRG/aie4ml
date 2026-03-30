import argparse
import json
import os
from pathlib import Path

import hls4ml
import numpy as np
import tensorflow as tf
from aie4ml.simulation import read_aie_report
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras.models import Sequential

from sweep_utils import default_output_dtype, dtype_bits, dtype_frac, save_rows_csv, seed_everything, source_vitis

NUM_LAYERS = 8
BATCH = 8
IN_FEATURES = 192
OUT_FEATURES = 192
INPUT_DTYPE = "i8"
WEIGHT_DTYPE = "i8"
OUTPUT_DTYPE = "i8"
FIXED_TILING = {"tile_m": 4, "tile_k": 8, "tile_n": 8}
CAS_CONFIGS = [(2, 6), (3, 4), (4, 3), (6, 2)]
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "tiling_intensity_sweep"
CSV_PATH = OUTPUT_ROOT / "tiling_intensity_sweep.csv"
PROJECT_NAME = "tiling_intensity_sweep"
RESULT_ATOL = 1e-3
BUILD_THREADS = 32

seed_everything()


def output_dir_for(cas_length, cas_num):
    # Keep the run directory shell-safe: Vitis emits unquoted shell redirections
    # against paths inside the build tree, so parentheses in the path break build.
    return OUTPUT_ROOT / f"LMKN_{NUM_LAYERS}_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}__PK_{cas_length}__PN_{cas_num}"


def result_path(output_dir):
    return output_dir / "result.json"


def load_result(output_dir):
    path = result_path(output_dir)
    return json.loads(path.read_text()) if path.exists() else None


def save_result(output_dir, payload):
    result_path(output_dir).write_text(json.dumps(payload, indent=2))


def failure_summary(output_dir, fallback=None):
    for name in ["AIECompiler.log", "log"]:
        path = output_dir / name
        if not path.exists():
            continue
        lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        for line in reversed(lines):
            if "ERROR" in line or "Failed" in line or "Compilation Failed" in line:
                return line
    return fallback


def needs_execution(output_dirs, rerun_failed):
    for output_dir in output_dirs:
        result = load_result(output_dir)
        if result and result.get("status") == "success":
            continue
        if result and result.get("status") == "failed" and not rerun_failed:
            continue
        if result is None:
            return True
        if rerun_failed:
            return True
    return False


def throughput_fps(latency_ns):
    if latency_ns is None or np.isnan(latency_ns) or latency_ns <= 0:
        return np.nan
    return 1e9 / latency_ns


def gops_per_second(num_layers, batch, in_features, out_features, latency_ns):
    if latency_ns is None or np.isnan(latency_ns) or latency_ns <= 0:
        return np.nan
    total_ops = 2.0 * num_layers * batch * in_features * out_features
    return total_ops / latency_ns


def build_dense_chain_model(
    in_features,
    out_features,
    num_layers,
    input_dtype=INPUT_DTYPE,
    weight_dtype=WEIGHT_DTYPE,
    output_dtype=OUTPUT_DTYPE,
):
    tf.keras.backend.clear_session()
    output_dtype = default_output_dtype(input_dtype, weight_dtype) if output_dtype is None else output_dtype
    input_bits = dtype_bits(input_dtype)
    weight_bits = dtype_bits(weight_dtype)
    input_frac = dtype_frac(input_dtype)
    output_bits = dtype_bits(output_dtype)
    output_frac = dtype_frac(output_dtype)
    layers = [QActivation(quantized_bits(input_bits, input_frac), name="input_quant", input_shape=(in_features,))]
    for index in range(num_layers):
        layers.append(
            QDense(
                out_features,
                name=f"dense_{index}",
                kernel_quantizer=quantized_bits(weight_bits, 0, alpha=1),
                bias_quantizer=quantized_bits(output_bits, output_frac, alpha=1),
            )
        )
        layers.append(QActivation(quantized_bits(output_bits, output_frac), name=f"quant_{index}"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="mse")
    return model


def build_dense_chain_aie_model(model, batch, iterations, platform, output_dir, project_name, parallelism, tiling):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    layer_cfg = cfg.setdefault("LayerName", {})
    for layer_name, config in layer_cfg.items():
        if "dense" not in layer_name.lower():
            continue
        config["parallelism"] = {key: int(value) for key, value in parallelism.items()}
        config["tiling"] = {key: int(value) for key, value in tiling.items()}
    aie_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(output_dir),
        backend="aie",
        project_name=project_name,
        batch_size=batch,
        iterations=iterations,
        part=platform,
    )
    aie_model.compile()
    return aie_model


def fixed_input(batch, in_features):
    return np.random.default_rng(42).random((batch, in_features), dtype=np.float32)


def force_make_jobs(output_dir, jobs=BUILD_THREADS):
    makefile_path = output_dir / "Makefile"
    if not makefile_path.exists():
        return
    content = makefile_path.read_text()
    old = "JOBS := $(shell n=$$(nproc); echo $$(( n > 32 ? 32 : n )))"
    new = f"JOBS := {int(jobs)}"
    if old in content:
        makefile_path.write_text(content.replace(old, new))


def active_cores_path(output_dir):
    return output_dir / "Work" / "aie" / "active_cores.json"


def extract_layout_metrics(output_dir):
    path = active_cores_path(output_dir)
    if not path.exists():
        return {
            "active_core_count": np.nan,
            "active_column_count": np.nan,
            "active_row_count": np.nan,
            "min_active_col": np.nan,
            "max_active_col": np.nan,
            "min_active_row": np.nan,
            "max_active_row": np.nan,
            "active_col_span": np.nan,
            "active_row_span": np.nan,
            "column_band_count": np.nan,
            "row_band_count": np.nan,
            "uses_multiple_rows": False,
            "uses_multiple_column_bands": False,
        }

    payload = json.loads(path.read_text())
    core_names = []
    for entry in payload.get("ActiveCores", []):
        core_names.extend(entry.keys())

    coords = []
    for name in core_names:
        try:
            col_text, row_text = name.split("_", 1)
            coords.append((int(col_text), int(row_text)))
        except ValueError:
            continue

    if not coords:
        return {
            "active_core_count": 0,
            "active_column_count": 0,
            "active_row_count": 0,
            "min_active_col": np.nan,
            "max_active_col": np.nan,
            "min_active_row": np.nan,
            "max_active_row": np.nan,
            "active_col_span": np.nan,
            "active_row_span": np.nan,
            "column_band_count": 0,
            "row_band_count": 0,
            "uses_multiple_rows": False,
            "uses_multiple_column_bands": False,
        }

    cols = sorted({col for col, _ in coords})
    rows = sorted({row for _, row in coords})

    def band_count(values):
        if not values:
            return 0
        count = 1
        for prev, curr in zip(values, values[1:]):
            if curr != prev + 1:
                count += 1
        return count

    row_to_cols = {}
    for col, row in coords:
        row_to_cols.setdefault(row, set()).add(col)

    column_band_count = sum(band_count(sorted(cols_in_row)) for cols_in_row in row_to_cols.values())
    row_band_count = band_count(rows)

    return {
        "active_core_count": len(coords),
        "active_column_count": len(cols),
        "active_row_count": len(rows),
        "min_active_col": min(cols),
        "max_active_col": max(cols),
        "min_active_row": min(rows),
        "max_active_row": max(rows),
        "active_col_span": max(cols) - min(cols) + 1,
        "active_row_span": max(rows) - min(rows) + 1,
        "column_band_count": column_band_count,
        "row_band_count": row_band_count,
        "uses_multiple_rows": len(rows) > 1,
        "uses_multiple_column_bands": column_band_count > row_band_count,
    }


def measure_configuration(output_dir, cas_length, cas_num, rerun_failed):
    output_dir.mkdir(parents=True, exist_ok=True)

    cached = load_result(output_dir)
    if cached and cached.get("status") == "success":
        return cached, "cached"
    if cached and cached.get("status") == "failed" and not rerun_failed:
        return cached, "failed"

    try:
        seed_everything()
        model = build_dense_chain_model(IN_FEATURES, OUT_FEATURES, NUM_LAYERS)
        x = fixed_input(BATCH, IN_FEATURES)
        reference = np.asarray(model(x, training=False))
        aie_model = build_dense_chain_aie_model(
            model,
            BATCH,
            ITERS,
            PLATFORM,
            output_dir,
            PROJECT_NAME,
            parallelism={"cas_length": cas_length, "cas_num": cas_num},
            tiling=FIXED_TILING,
        )
        force_make_jobs(output_dir)
        aie_model.build()
        prediction = np.asarray(aie_model.predict(x, simulator="aie"))
        latency_ns = float(read_aie_report(aie_model)["fifo_latency"]["ns"])
        if prediction.shape != reference.shape:
            output_ok = False
            max_abs_error = np.nan
            error = f"output shape mismatch: expected {reference.shape}, got {prediction.shape}"
        else:
            max_abs_error = float(np.max(np.abs(reference - prediction)))
            output_ok = bool(np.allclose(reference, prediction, atol=RESULT_ATOL, rtol=0.0))
            error = None if output_ok else f"output mismatch: max_abs_error={max_abs_error:.6f}"
        payload = {
            "status": "success" if output_ok else "failed",
            "latency_ns": latency_ns,
            "throughput_fps": throughput_fps(latency_ns),
            "gops_per_s": gops_per_second(NUM_LAYERS, BATCH, IN_FEATURES, OUT_FEATURES, latency_ns),
            "output_ok": output_ok,
            "max_abs_error": max_abs_error,
            "error": error,
        }
        payload.update(extract_layout_metrics(output_dir))
        save_result(output_dir, payload)
        return payload, "new"
    except Exception as exc:
        payload = {
            "status": "failed",
            "latency_ns": np.nan,
            "throughput_fps": np.nan,
            "gops_per_s": np.nan,
            "output_ok": False,
            "max_abs_error": np.nan,
            "error": failure_summary(output_dir, str(exc)),
        }
        payload.update(extract_layout_metrics(output_dir))
        save_result(output_dir, payload)
        return payload, "failed"


def csv_row(output_dir, cas_length, cas_num, result):
    layout = extract_layout_metrics(output_dir)
    return {
        "num_layers": NUM_LAYERS,
        "batch": BATCH,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "input_dtype": INPUT_DTYPE,
        "weight_dtype": WEIGHT_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "tile_m": FIXED_TILING["tile_m"],
        "tile_k": FIXED_TILING["tile_k"],
        "tile_n": FIXED_TILING["tile_n"],
        "build_threads": BUILD_THREADS,
        "cas_length": cas_length,
        "cas_num": cas_num,
        "cas_product": cas_length * cas_num,
        "latency_ns": result["latency_ns"],
        "latency_us": np.nan if np.isnan(result["latency_ns"]) else result["latency_ns"] / 1000.0,
        "throughput_fps": result["throughput_fps"],
        "gops_per_s": result["gops_per_s"],
        "output_ok": result["output_ok"],
        "max_abs_error": result["max_abs_error"],
        "status": result["status"],
        "error": result["error"],
        "active_core_count": layout["active_core_count"],
        "active_column_count": layout["active_column_count"],
        "active_row_count": layout["active_row_count"],
        "min_active_col": layout["min_active_col"],
        "max_active_col": layout["max_active_col"],
        "min_active_row": layout["min_active_row"],
        "max_active_row": layout["max_active_row"],
        "active_col_span": layout["active_col_span"],
        "active_row_span": layout["active_row_span"],
        "column_band_count": layout["column_band_count"],
        "row_band_count": layout["row_band_count"],
        "uses_multiple_rows": layout["uses_multiple_rows"],
        "uses_multiple_column_bands": layout["uses_multiple_column_bands"],
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    output_dirs = [args.output_root / output_dir_for(cas_length, cas_num).name for cas_length, cas_num in CAS_CONFIGS]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    rows = []
    for cas_length, cas_num in CAS_CONFIGS:
        output_dir = args.output_root / output_dir_for(cas_length, cas_num).name
        result, execution_status = measure_configuration(output_dir, cas_length, cas_num, args.rerun_failed)
        rows.append(csv_row(output_dir, cas_length, cas_num, result))
        save_rows_csv(
            CSV_PATH if args.output_root == OUTPUT_ROOT else args.output_root / CSV_PATH.name,
            rows,
            [
                "num_layers",
                "batch",
                "in_features",
                "out_features",
                "input_dtype",
                "weight_dtype",
                "output_dtype",
                "tile_m",
                "tile_k",
                "tile_n",
                "build_threads",
                "cas_length",
                "cas_num",
                "cas_product",
                "latency_ns",
                "latency_us",
                "throughput_fps",
                "gops_per_s",
                "output_ok",
                "max_abs_error",
                "status",
                "error",
                "active_core_count",
                "active_column_count",
                "active_row_count",
                "min_active_col",
                "max_active_col",
                "min_active_row",
                "max_active_row",
                "active_col_span",
                "active_row_span",
                "column_band_count",
                "row_band_count",
                "uses_multiple_rows",
                "uses_multiple_column_bands",
                "output_dir",
            ],
        )
        latency_summary = "nan" if np.isnan(result["latency_ns"]) else f"{result['latency_ns']:.3f}"
        throughput_summary = "nan" if np.isnan(result["throughput_fps"]) else f"{result['throughput_fps']:.6f}"
        gops_summary = "nan" if np.isnan(result["gops_per_s"]) else f"{result['gops_per_s']:.6f}"
        layout = extract_layout_metrics(output_dir)
        error_suffix = f" error={result['error']}" if result["error"] else ""
        print(
            f"[{execution_status}] PK={cas_length:>2} PN={cas_num:>2} "
            f"latency_ns={latency_summary} throughput_fps={throughput_summary} "
            f"gops_per_s={gops_summary} active_cols={layout['active_column_count']} "
            f"active_rows={layout['active_row_count']} bands={layout['column_band_count']} "
            f"output_ok={result['output_ok']} status={result['status']}{error_suffix}"
        )


if __name__ == "__main__":
    main()
