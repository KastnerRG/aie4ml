import argparse
import os
import tempfile
from pathlib import Path

import hls4ml
import matplotlib.pyplot as plt
import numpy as np
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras.models import Sequential

from sweep_utils import (
    default_output_dtype,
    dtype_bits,
    dtype_frac,
    measure_latency_ns,
    needs_execution,
    seed_everything,
    source_vitis,
)

BATCH = 8
IN_FEATURES = 128
OUT_FEATURES = 128
NUM_LAYERS = 6
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
IO_ROUTE_MODES = ["auto", "direct", "memtile"]
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / f"model_sweep__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}__layers_{NUM_LAYERS}"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

seed_everything()


def build_dense_chain_model(in_features, out_features, num_layers, input_dtype="i8", weight_dtype="i8", output_dtype=None):
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


def configure_dense_chain_cfg(cfg, probe_model, io_mode=None, use_relu=False):
    layer_cfg = cfg.setdefault("LayerName", {})
    for layer in probe_model.get_layers():
        if layer.class_name == "Dense":
            layer_cfg.setdefault(layer.name, {})["aie_fused_activation"] = "relu" if use_relu else ""
    if io_mode is None:
        return
    for layer in probe_model.get_layers():
        if layer.class_name == "Input":
            continue
        tensor_routes = {}
        for source in getattr(layer, "inputs", []):
            if source not in probe_model.output_vars:
                continue
            tensor_name = probe_model.output_vars[source].name
            tensor_routes[tensor_name] = io_mode
            producer_cfg = layer_cfg.setdefault(source, {})
            producer_cfg.setdefault("io_route", {}).setdefault("outputs", {})[tensor_name] = io_mode
        if tensor_routes:
            consumer_cfg = layer_cfg.setdefault(layer.name, {})
            consumer_cfg.setdefault("io_route", {}).setdefault("inputs", {}).update(tensor_routes)


def build_dense_chain_aie_model(in_features, out_features, num_layers, batch, iterations, platform, output_dir, project_name, io_mode=None, use_relu=False):
    model = build_dense_chain_model(in_features, out_features, num_layers)
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    with tempfile.TemporaryDirectory(prefix="aie4ml_model_sweep_") as probe_dir:
        probe_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=cfg,
            output_dir=probe_dir,
            backend="aie",
            project_name=f"{project_name}_probe",
            batch_size=batch,
            iterations=iterations,
            part=platform,
        )
        configure_dense_chain_cfg(cfg, probe_model, io_mode=io_mode, use_relu=use_relu)
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


def save_grouped_bar_csv(output_root, x_values, series):
    np.savez(output_root / "latencies_ns.npz", **{label: values for label, values in series})
    header = ["x_value", *[label.replace(" ", "_") + "_us" for label, _ in series]]
    rows = [",".join(header)]
    for index, x_value in enumerate(x_values):
        values = [f'"{x_value}"']
        for _, latencies_ns in series:
            value = latencies_ns[index]
            values.append("nan" if np.isnan(value) else f"{value / 1000.0:.6f}")
        rows.append(",".join(values))
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def plot_grouped_bars(output_png, x_labels, series, x_title, y_title, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(x_labels))
    width = 0.8 / max(1, len(series))
    offsets = np.linspace(-(len(series) - 1) * width / 2, (len(series) - 1) * width / 2, len(series))
    ymax = 0.0
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for index, (label, latencies_ns) in enumerate(series):
        latencies_us = latencies_ns / 1000.0
        bars = ax.bar(
            x + offsets[index],
            np.nan_to_num(latencies_us, nan=0.0),
            width=width,
            color=["white" if np.isnan(value) else colors[index % len(colors)] for value in latencies_us],
            edgecolor="black",
            label=label,
        )
        for bar, value in zip(bars, latencies_us):
            y = 0.0 if np.isnan(value) else value
            ymax = max(ymax, y)
            ax.text(bar.get_x() + bar.get_width() / 2.0, y, "fail" if np.isnan(value) else f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels=x_labels)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
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


def measure_point(io_mode, use_relu, output_root, rerun_failed):
    output_dir = output_root / f"io_route_{io_mode}__relu_{int(use_relu)}"
    return measure_latency_ns(
        output_dir,
        lambda path: build_dense_chain_aie_model(
            IN_FEATURES,
            OUT_FEATURES,
            NUM_LAYERS,
            BATCH,
            ITERS,
            PLATFORM,
            path,
            "model_sweep",
            io_mode=io_mode,
            use_relu=use_relu,
        ),
        IN_FEATURES,
        BATCH,
        rerun_failed=rerun_failed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    output_dirs = [
        args.output_root / f"io_route_{io_mode}__relu_{int(use_relu)}"
        for io_mode in IO_ROUTE_MODES
        for use_relu in (False, True)
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if needs_execution(output_dirs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    latencies = {False: [], True: []}
    for io_mode in IO_ROUTE_MODES:
        for use_relu in (False, True):
            latency_ns, status = measure_point(io_mode, use_relu, args.output_root, args.rerun_failed)
            latencies[use_relu].append(latency_ns)
            summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
            print(f"[{status}] io_route={io_mode:>7} relu={'on' if use_relu else 'off':>3} latency_ns={summary}")

    series = [
        ("without relu", np.array(latencies[False], dtype=float)),
        ("with relu", np.array(latencies[True], dtype=float)),
    ]
    save_grouped_bar_csv(args.output_root, IO_ROUTE_MODES, series)
    output_png = RESULTS_ROOT / f"lat_bar__model_sweep__size_{BATCH}_{IN_FEATURES}_{OUT_FEATURES}__layers_{NUM_LAYERS}.png"
    plot_grouped_bars(
        output_png,
        IO_ROUTE_MODES,
        series,
        "io_route",
        "latency (us)",
        f"6-layer dense chain, size=({BATCH},{IN_FEATURES},{OUT_FEATURES}), dtype=i8",
    )


if __name__ == "__main__":
    main()
