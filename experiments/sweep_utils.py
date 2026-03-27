import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import hls4ml
import numpy as np
import tensorflow as tf
from aie4ml.simulation import read_aie_report
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras.models import Sequential

TILE_API_TILINGS = {
    ("i8", "i8"): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
    ("i16", "i8"): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
    ("i8", "i16"): [(4, 4, 4), (4, 4, 8)],
    ("i16", "i16"): [(4, 4, 4), (2, 4, 8), (4, 2, 8), (4, 4, 8), (8, 1, 8), (8, 2, 8)],
}


def seed_everything(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def source_vitis(vitis_settings):
    proc = subprocess.run(
        ["bash", "-lc", f"source {vitis_settings} >/dev/null 2>&1 && env -0"],
        capture_output=True,
        check=True,
    )
    for entry in proc.stdout.split(b"\x00"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            os.environ[key.decode()] = value.decode()
    env_prefix = Path(sys.prefix)
    os.environ["PATH"] = f"{env_prefix / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{env_prefix / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")


def dtype_bits(dtype_name):
    return int(dtype_name[1:])


def dtype_frac(dtype_name):
    return 2 if dtype_bits(dtype_name) <= 8 else 4


def default_output_dtype(input_dtype, weight_dtype):
    return {
        ("i8", "i8"): "i8",
        ("i16", "i8"): "i8",
        ("i8", "i16"): "i8",
        ("i16", "i16"): "i16",
    }[(input_dtype, weight_dtype)]


def build_dense_model(in_features, out_features, input_dtype="i8", weight_dtype="i8", output_dtype=None):
    output_dtype = default_output_dtype(input_dtype, weight_dtype) if output_dtype is None else output_dtype
    input_bits = dtype_bits(input_dtype)
    weight_bits = dtype_bits(weight_dtype)
    input_frac = dtype_frac(input_dtype)
    output_bits = dtype_bits(output_dtype)
    output_frac = dtype_frac(output_dtype)
    model = Sequential(
        [
            QActivation(quantized_bits(input_bits, input_frac), name="input_quant", input_shape=(in_features,)),
            QDense(
                out_features,
                name="dense",
                kernel_quantizer=quantized_bits(weight_bits, 0, alpha=1),
                bias_quantizer=quantized_bits(output_bits, output_frac, alpha=1),
            ),
            QActivation(quantized_bits(output_bits, output_frac), name="output_quant"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_dense_aie_model(
    in_features,
    out_features,
    batch,
    iterations,
    platform,
    output_dir,
    project_name,
    parallelism=None,
    tiling=None,
    input_dtype="i8",
    weight_dtype="i8",
    output_dtype=None,
):
    model = build_dense_model(
        in_features,
        out_features,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
    )
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    dense_name = next(name for name in cfg["LayerName"] if "dense" in name.lower() or "fc" in name.lower())
    if parallelism:
        cfg["LayerName"][dense_name]["parallelism"] = {k: int(v) for k, v in parallelism.items()}
    if tiling:
        cfg["LayerName"][dense_name]["tiling"] = {k: int(v) for k, v in tiling.items()}
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


def result_path(output_dir):
    return output_dir / "result.json"


def load_result(output_dir):
    path = result_path(output_dir)
    return json.loads(path.read_text()) if path.exists() else None


def save_result(output_dir, status, latency_ns=None, error=None):
    payload = {"status": status, "latency_ns": latency_ns, "error": error}
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


def load_cached_latency_ns(output_dir):
    result = load_result(output_dir)
    if result and result["status"] == "success":
        return float(result["latency_ns"])
    report = read_aie_report(output_dir)
    latency = report.get("fifo_latency", {}).get("ns")
    if latency is not None:
        save_result(output_dir, "success", latency_ns=float(latency))
        return float(latency)
    return None


def load_failed_result(output_dir):
    result = load_result(output_dir)
    if result and result["status"] == "failed":
        return result
    error = failure_summary(output_dir)
    if error is not None:
        save_result(output_dir, "failed", error=error)
        return load_result(output_dir)
    return None


def measure_latency_ns(output_dir, build_aie_model, in_features, batch, rerun_failed=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_ns = load_cached_latency_ns(output_dir)
    if latency_ns is not None:
        return latency_ns, "cached"

    if load_failed_result(output_dir) and not rerun_failed:
        return np.nan, "failed"

    try:
        aie_model = build_aie_model(output_dir)
        aie_model.build()
        x = np.random.random((batch, in_features)).astype(np.float32)
        aie_model.predict(x, simulator="aie")
        latency_ns = read_aie_report(aie_model)["fifo_latency"]["ns"]
        save_result(output_dir, "success", latency_ns=float(latency_ns))
        return float(latency_ns), "new"
    except Exception as exc:
        save_result(output_dir, "failed", error=failure_summary(output_dir, str(exc)))
        return np.nan, "failed"


def sweep_factors(max_factor):
    factors = []
    factor = 1
    while factor <= max_factor:
        factors.append(factor)
        factor *= 2
    return factors


def api_tilings(dtype_pair, tile_m=None):
    tilings = TILE_API_TILINGS[dtype_pair]
    if tile_m is None:
        return tilings
    return [tiling for tiling in tilings if tiling[0] == tile_m]


def valid_tile_sizes(feature_size, dtype_pair, axis, tile_m=None):
    dim = {"k": 1, "n": 2}[axis]
    api_sizes = {tiling[dim] for tiling in api_tilings(dtype_pair, tile_m=tile_m)}
    return [
        size
        for size in range(1, feature_size + 1)
        if feature_size % size == 0 and any(size % api_size == 0 for api_size in api_sizes)
    ]


def select_api_tiling(tile_k_size, tile_n_size, dtype_pair, tile_m=None):
    matches = [
        tiling
        for tiling in api_tilings(dtype_pair, tile_m=tile_m)
        if tile_k_size % tiling[1] == 0 and tile_n_size % tiling[2] == 0
    ]
    if not matches:
        raise ValueError(
            f"No API tiling matches tile_k_size={tile_k_size}, tile_n_size={tile_n_size}, dtype_pair={dtype_pair}, tile_m={tile_m}"
        )
    return max(matches, key=lambda tiling: (tiling[1] * tiling[2], tiling[1], tiling[2]))


def tuple_labels(first_values, second_values):
    return [f"({first},{second})" for first, second in zip(first_values, second_values)]


def common_tile_sizes(feature_size, dtype_pair, tile_m=None):
    return sorted(set(valid_tile_sizes(feature_size, dtype_pair, "k", tile_m=tile_m)) & set(valid_tile_sizes(feature_size, dtype_pair, "n", tile_m=tile_m)))


def choose_square_problem_size(feature_sizes, batch_sizes, dtype_pair, tile_m=None):
    best = None
    for batch in batch_sizes:
        for size in feature_sizes:
            tile_sizes = common_tile_sizes(size, dtype_pair, tile_m=tile_m)
            candidate = (len(tile_sizes), size, -batch)
            if best is None or candidate > best[0]:
                best = (candidate, batch, size, tile_sizes)
    if best is None or not best[3]:
        raise ValueError(f"No valid problem size found for dtype_pair={dtype_pair}, tile_m={tile_m}")
    _, batch, size, tile_sizes = best
    return batch, size, size, tile_sizes


def save_rows_csv(csv_path, rows, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_csv(csv_path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def needs_execution(output_dirs, rerun_failed):
    for output_dir in output_dirs:
        if load_cached_latency_ns(output_dir) is not None:
            continue
        if load_failed_result(output_dir) and not rerun_failed:
            continue
        return True
    return False
