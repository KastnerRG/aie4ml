import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import hls4ml
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from aie4ml.simulation import read_aie_report
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras.models import Sequential

IN_FEATURES = 128
OUT_FEATURES = 128
BATCH = 8
ITERS = 1
CAS_LENGTH_MAX = 32
CAS_NUM_MAX = 32
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tmp/tools/Xilinx/2025.2/Vitis/.settings64-Vitis.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "cas_num_len_runs"

np.random.seed(42)
tf.random.set_seed(42)


def source_vitis():
    proc = subprocess.run(
        ["bash", "-lc", f"source {VITIS_SETTINGS} >/dev/null 2>&1 && env -0"],
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


def build_model():
    model = Sequential(
        [
            QActivation(quantized_bits(8, 2), name="input_quant", input_shape=(IN_FEATURES,)),
            QDense(
                OUT_FEATURES,
                name="dense",
                kernel_quantizer=quantized_bits(8, 0, alpha=1),
                bias_quantizer=quantized_bits(8, 2, alpha=1),
            ),
            QActivation(quantized_bits(8, 2), name="output_quant"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_aie_model(cas_num, cas_length, output_dir):
    model = build_model()
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    dense_name = next(name for name in cfg["LayerName"] if "dense" in name.lower() or "fc" in name.lower())
    cfg["LayerName"][dense_name]["parallelism"] = {"cas_num": cas_num, "cas_length": cas_length}
    aie_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(output_dir),
        backend="aie",
        project_name="cas_num_len",
        batch_size=BATCH,
        iterations=ITERS,
        part=PLATFORM,
    )
    aie_model.compile()
    return aie_model


def run_dir(output_root, cas_num, cas_length):
    return output_root / f"cas_num_{cas_num}_cas_length_{cas_length}"


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


def measure_latency_ns(cas_num, cas_length, output_root, rerun_failed=False):
    output_dir = run_dir(output_root, cas_num, cas_length)
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_ns = load_cached_latency_ns(output_dir)
    if latency_ns is not None:
        return latency_ns, "cached"

    if load_failed_result(output_dir) and not rerun_failed:
        return np.nan, "failed"

    try:
        aie_model = build_aie_model(cas_num, cas_length, output_dir)
        aie_model.build()
        x = np.random.random((BATCH, IN_FEATURES)).astype(np.float32)
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


def save_latencies(output_root, cas_lengths, cas_nums, latencies_ns):
    np.save(output_root / "latencies_ns.npy", latencies_ns)
    x_labels = [f"({cas_num},{OUT_FEATURES // cas_num})" for cas_num in cas_nums]
    y_labels = [f"({cas_len},{IN_FEATURES // cas_len})" for cas_len in cas_lengths]
    rows = [",".join(["cas_len\\cas_num", *x_labels])]
    for label, row in zip(y_labels, latencies_ns / 1000.0):
        values = ["nan" if np.isnan(value) else f"{value:.6f}" for value in row]
        rows.append(",".join([label, *values]))
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def run_sweep(cas_lengths, cas_nums, output_root, rerun_failed=False):
    latencies = np.full((len(cas_lengths), len(cas_nums)), np.nan)
    for row, cas_length in enumerate(cas_lengths):
        for col, cas_num in enumerate(cas_nums):
            latency_ns, status = measure_latency_ns(cas_num, cas_length, output_root, rerun_failed=rerun_failed)
            latencies[row, col] = latency_ns
            summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
            print(
                f"[{status}] cas_num={cas_num:>2} cas_length={cas_length:>2} "
                f"tile_out={OUT_FEATURES // cas_num:>3} tile_in={IN_FEATURES // cas_length:>3} "
                f"latency_ns={summary}"
            )
            save_latencies(output_root, cas_lengths, cas_nums, latencies)
    return latencies


def plot_heatmap(cas_lengths, cas_nums, latencies_ns, output_root):
    x_labels = [f"({cas_num},{OUT_FEATURES // cas_num})" for cas_num in cas_nums]
    y_labels = [f"({cas_len},{IN_FEATURES // cas_len})" for cas_len in cas_lengths]
    latencies_us = latencies_ns / 1000.0
    masked = np.ma.masked_invalid(latencies_us)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(masked, origin="lower", cmap=cmap)
    ax.set_xticks(range(len(cas_nums)), labels=x_labels)
    ax.set_yticks(range(len(cas_lengths)), labels=y_labels)
    ax.set_xlabel("(cas_num, tile_out_size)")
    ax.set_ylabel("(cas_len, tile_in_size)")
    ax.set_title("Latency Per Inference")
    for row in range(latencies_us.shape[0]):
        for col in range(latencies_us.shape[1]):
            if not np.isnan(latencies_us[row, col]):
                ax.text(col, row, f"{latencies_us[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="latency (us)")
    fig.tight_layout()
    fig.savefig(output_root / "latency_heatmap.png", dpi=200)
    plt.show()


def needs_execution(cas_lengths, cas_nums, output_root, rerun_failed):
    for cas_length in cas_lengths:
        for cas_num in cas_nums:
            output_dir = run_dir(output_root, cas_num, cas_length)
            if load_cached_latency_ns(output_dir) is not None:
                continue
            if load_failed_result(output_dir) and not rerun_failed:
                continue
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    cas_lengths = sweep_factors(CAS_LENGTH_MAX)
    cas_nums = sweep_factors(CAS_NUM_MAX)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(cas_lengths, cas_nums, args.output_root, args.rerun_failed):
        source_vitis()
    latencies_ns = run_sweep(cas_lengths, cas_nums, args.output_root, rerun_failed=args.rerun_failed)
    plot_heatmap(cas_lengths, cas_nums, latencies_ns, args.output_root)


if __name__ == "__main__":
    main()
