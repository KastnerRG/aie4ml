import csv
import json
import os
import subprocess
import tempfile
import sys
from pathlib import Path

import hls4ml
import matplotlib.pyplot as plt
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


def build_dense_chain_model(
    in_features,
    out_features,
    num_layers,
    input_dtype="i8",
    weight_dtype="i8",
    output_dtype=None,
):
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



def build_dense_chain_aie_model(
    in_features,
    out_features,
    num_layers,
    batch,
    iterations,
    platform,
    output_dir,
    project_name,
    io_mode=None,
    use_relu=False,
    input_dtype="i8",
    weight_dtype="i8",
    output_dtype=None,
):
    model = build_dense_chain_model(
        in_features,
        out_features,
        num_layers,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
    )
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


def save_bar_csv(output_root, x_values, latencies_ns, metadata_rows):
    np.save(output_root / "latencies_ns.npy", latencies_ns)
    rows = ["x_value,latency_us,cas_length,cas_num,tile_m,tile_k,tile_n"]
    for x_value, latency_ns, meta in zip(x_values, latencies_ns, metadata_rows):
        latency_us = "nan" if np.isnan(latency_ns) else f"{latency_ns / 1000.0:.6f}"
        rows.append(
            f'"{x_value}",{latency_us},{meta.get("cas_length", "")},{meta.get("cas_num", "")},{meta["tile_m"]},{meta["tile_k"]},{meta["tile_n"]}'
        )
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def save_heatmap_csv(output_root, header, x_labels, y_labels, latencies_ns):
    np.save(output_root / "latencies_ns.npy", latencies_ns)
    rows = [",".join([header, *x_labels])]
    for label, row in zip(y_labels, latencies_ns / 1000.0):
        values = ["nan" if np.isnan(value) else f"{value:.6f}" for value in row]
        rows.append(",".join([label, *values]))
    (output_root / "latencies_us.csv").write_text("\n".join(rows) + "\n")


def plot_bar(output_png, x_labels, latencies_ns, x_title, y_title, title):
    latencies_us = latencies_ns / 1000.0
    heights = np.nan_to_num(latencies_us, nan=0.0)
    colors = ["white" if np.isnan(value) else "tab:blue" for value in latencies_us]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(x_labels)), heights, color=colors, edgecolor="black")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=45, ha="right")
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title)
    for bar, value in zip(bars, latencies_us):
        label = "fail" if np.isnan(value) else f"{value:.2f}"
        y = bar.get_height() if not np.isnan(value) else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2.0, y, label, ha="center", va="bottom", rotation=90)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


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


def save_rows_csv(csv_path, rows, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_csv(csv_path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def plot_tile_k_n_eff_lines(output_png, rows, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    api_order = [api_label(tiling) for tiling in ((4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8))]
    colors = {label: plt.get_cmap("tab10")(index) for index, label in enumerate(api_order)}
    styles = [
        ("in>out", ":", "in > out"),
        ("in<out", "-", "in < out"),
    ]
    x_rows = sorted(rows, key=lambda row: int(row["macs"]))
    x_labels = []
    x_positions = {}
    for row in x_rows:
        macs = int(row["macs"])
        if macs in x_positions:
            continue
        in_features = int(row["in_features"])
        out_features = int(row["out_features"])
        left, right = sorted((in_features, out_features))
        x_positions[macs] = len(x_labels)
        x_labels.append(f"{macs / 1000.0:.1f}k\n({row['batch']},{left}*{right})")
    plotted = set()
    for api in api_order:
        for orientation, linestyle, suffix in styles:
            series = [
                row
                for row in rows
                if row["api"] == api and row["orientation"] == orientation and row["status"] == "success"
            ]
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


def plot_tile_k_n_eff_heatmap(output_png, rows, title):
    ordered_rows = sorted(rows, key=lambda row: (int(row["macs"]), 0 if row["orientation"] == "in>out" else 1, row["shape"]))
    x_labels = []
    seen_shapes = set()
    for row in ordered_rows:
        if row["shape"] not in seen_shapes:
            x_labels.append(row["shape"])
            seen_shapes.add(row["shape"])
    y_labels = [api_label(tiling) for tiling in ((4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8))]
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
    fig, ax = plt.subplots(figsize=(12, 4.8))
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
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                "fail" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
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


def plot_overlaid_bars(output_png, x_labels, series, x_title, y_title, title):
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(index % cmap.N) for index, (label, _) in enumerate(series)}
    legend_done = set()
    max_height = 0.0
    for x_index in range(len(x_labels)):
        bars_at_x = []
        for label, latencies_ns in series:
            latencies_us = latencies_ns / 1000.0
            value = float(latencies_us[x_index]) if not np.isnan(latencies_us[x_index]) else np.nan
            bars_at_x.append((0.0 if np.isnan(value) else value, label, value))
            if not np.isnan(value):
                max_height = max(max_height, value)
        for _, label, value in sorted(bars_at_x, key=lambda item: item[0], reverse=True):
            bar = ax.bar(
                [x_index],
                [0.0 if np.isnan(value) else value],
                width=0.8,
                color=colors[label],
                edgecolor="black",
                label=label if label not in legend_done else None,
            )[0]
            legend_done.add(label)
            label_text = "fail" if np.isnan(value) else f"{value:.2f}"
            y = 0.0 if np.isnan(value) else value
            ax.text(bar.get_x() + bar.get_width() / 2.0, y, label_text, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=45, ha="right")
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title)
    ax.legend(title="layer")
    if max_height > 0:
        ax.set_ylim(0, max_height * 1.12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_heatmap(output_png, x_labels, y_labels, x_title, y_title, latencies_ns):
    latencies_us = latencies_ns / 1000.0
    masked = np.ma.masked_invalid(latencies_us)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(masked, origin="lower", cmap=cmap)
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title("Latency Per Inference")
    for row in range(latencies_us.shape[0]):
        for col in range(latencies_us.shape[1]):
            if not np.isnan(latencies_us[row, col]):
                ax.text(col, row, f"{latencies_us[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="latency (us)")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def run_heatmap_sweep(x_values, y_values, output_root, rerun_failed, measure_point, describe_point, save_progress):
    latencies = np.full((len(y_values), len(x_values)), np.nan)
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            latency_ns, status = measure_point(x_value, y_value, output_root, rerun_failed)
            latencies[row, col] = latency_ns
            summary = "nan" if np.isnan(latency_ns) else f"{latency_ns:.3f}"
            print(describe_point(x_value, y_value, summary, status))
            save_progress(latencies)
    return latencies


def needs_execution(output_dirs, rerun_failed):
    for output_dir in output_dirs:
        if load_cached_latency_ns(output_dir) is not None:
            continue
        if load_failed_result(output_dir) and not rerun_failed:
            continue
        return True
    return False
