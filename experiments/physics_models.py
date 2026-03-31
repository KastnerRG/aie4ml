import argparse
import json
import math
import os
import re
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hls4ml
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
AIE4ML_SRC = REPO_ROOT / "aie4ml" / "src"
if str(AIE4ML_SRC) not in sys.path:
    sys.path.insert(0, str(AIE4ML_SRC))

from aie4ml.simulation import read_aie_report
from qkeras import QActivation, QDense, quantized_bits, quantized_relu
from tensorflow.keras.models import Sequential

from sweep_utils import dtype_bits, dtype_frac, save_rows_csv, seed_everything, source_vitis

BATCH = 8
INPUT_DTYPE = "i8"
WEIGHT_DTYPE = "i8"
OUTPUT_DTYPE = "i8"
FIXED_TILING = {"tile_m": 4, "tile_k": 8, "tile_n": 8}
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh")
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "models"
CSV_PATH = OUTPUT_ROOT / "models.csv"
LOG_PATH = OUTPUT_ROOT / "models.log"
PROJECT_NAME = "physics_models"
RESULT_ATOL = 1e-3
BUILD_THREADS = 32
TOP_EXECUTION_CANDIDATES = 4
TOP_LOGGED_MODEL_CANDIDATES = 16
BEAM_WIDTH = 256
MAX_TILE_M = 8
MAX_TILE_K = 128
MAX_TILE_N = 128

seed_everything()


@dataclass(frozen=True)
class LayerSpec:
    in_features: int
    out_features: int
    relu: bool = False
    batchnorm: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: tuple[LayerSpec, ...]


@dataclass(frozen=True)
class DeviceSpec:
    generation: str
    columns: int
    rows: int
    column_start: int
    row_start: int
    weight_mem_bytes: int
    max_mem_in_ports: int
    max_mem_out_ports: int

    @property
    def usable_columns(self) -> int:
        return int(self.columns) - int(self.column_start)


@dataclass(frozen=True)
class DenseParams:
    kernel: np.ndarray
    bias: np.ndarray
    batchnorm_folded: bool


@dataclass(frozen=True)
class LayerCandidate:
    cas_length: int
    cas_num: int
    input_slice_raw: int
    output_slice_raw: int
    input_slice: int
    output_slice: int
    tile_bytes: int
    tile_workload: int
    padded_total_work: int
    raw_total_work: int
    padding_waste: int

    @property
    def width(self) -> int:
        return int(self.cas_length)

    @property
    def height(self) -> int:
        return int(self.cas_num)

    @property
    def parallel_factor(self) -> int:
        return int(self.cas_length) * int(self.cas_num)

    @property
    def efficiency(self) -> float:
        if self.padded_total_work <= 0:
            return 0.0
        return float(self.raw_total_work) / float(self.padded_total_work)


@dataclass(frozen=True)
class ModelCandidate:
    rank: int
    layer_candidates: tuple[LayerCandidate, ...]
    width_used: int
    bottleneck_workload: int
    workload_imbalance: int
    total_padding_waste: int
    sum_cas_length: int
    sum_parallel_factor: int
    max_rows: int


MODEL_SPECS = [
    ModelSpec(
        name="jet_tagger",
        layers=(
            LayerSpec(16, 64, relu=True, batchnorm=True),
            LayerSpec(64, 32, relu=True, batchnorm=True),
            LayerSpec(32, 32, relu=True, batchnorm=True),
            LayerSpec(32, 8, relu=True, batchnorm=False),
        ),
    ),
    ModelSpec(
        name="vae_lhc_medium",
        layers=(
            LayerSpec(64, 64, relu=True),
            LayerSpec(64, 32, relu=True),
            LayerSpec(32, 8, relu=True),
            LayerSpec(8, 32, relu=True),
            LayerSpec(32, 64, relu=True),
            LayerSpec(64, 64, relu=True),
        ),
    ),
    ModelSpec(
        name="tau_event_selection",
        layers=(
            LayerSpec(160, 64, relu=True),
            LayerSpec(64, 64, relu=True),
            LayerSpec(64, 64, relu=True),
            LayerSpec(64, 8, relu=True),
        ),
    ),
    ModelSpec(
        name="vae_lhc_large",
        layers=(
            LayerSpec(64, 128, relu=True),
            LayerSpec(128, 64, relu=True),
            LayerSpec(64, 12, relu=True),
            LayerSpec(12, 64, relu=True),
            LayerSpec(64, 128, relu=True),
            LayerSpec(128, 64, relu=True),
        ),
    ),
    ModelSpec(
        name="multi_qubit_readout_discriminator",
        layers=(
            LayerSpec(256, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 8, relu=True),
        ),
    ),
    ModelSpec(
        name="deep_autoencoder",
        layers=(
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 8, relu=True),
            LayerSpec(8, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
            LayerSpec(128, 128, relu=True),
        ),
    ),
]


def sanitize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())).strip("_")


def stable_seed(value: str) -> int:
    return int(zlib.adler32(value.encode("utf-8")) & 0xFFFFFFFF)


def load_device_spec(platform: str = PLATFORM) -> DeviceSpec:
    device_path = Path(__file__).resolve().parents[1] / "src" / "aie4ml" / "aie_devices.json"
    payload = json.loads(device_path.read_text())
    device = payload[platform]
    return DeviceSpec(
        generation=str(device["Generation"]),
        columns=int(device["Columns"]),
        rows=int(device["Rows"]),
        column_start=int(device["ColumnStart"]),
        row_start=int(device["RowStart"]),
        weight_mem_bytes=int(device["Memory"]["WeightMemBytes"]),
        max_mem_in_ports=int(device["MaxMemTileInPorts"]),
        max_mem_out_ports=int(device["MaxMemTileOutPorts"]),
    )


def result_path(output_dir: Path) -> Path:
    return output_dir / "result.json"


def load_result(output_dir: Path):
    path = result_path(output_dir)
    return json.loads(path.read_text()) if path.exists() else None


def save_result(output_dir: Path, payload: dict) -> None:
    result_path(output_dir).write_text(json.dumps(payload, indent=2))


def failure_summary(output_dir: Path, fallback=None):
    for name in ["AIECompiler.log", "log"]:
        path = output_dir / name
        if not path.exists():
            continue
        lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        for line in reversed(lines):
            if "ERROR" in line or "Failed" in line or "Compilation Failed" in line:
                return line
    return fallback


def needs_execution(run_specs: list[tuple[Path, str]], rerun_failed: bool) -> bool:
    for output_dir, expected_signature in run_specs:
        result = load_result(output_dir)
        if result and result.get("status") == "success" and result.get("model_signature") in (None, expected_signature):
            continue
        if (
            result
            and result.get("status") == "failed"
            and not rerun_failed
            and result.get("model_signature") in (None, expected_signature)
        ):
            continue
        if result is None or rerun_failed:
            return True
        if result.get("model_signature") not in (None, expected_signature):
            return True
    return False


def throughput_batches_per_s(latency_ns):
    if latency_ns is None or np.isnan(latency_ns) or latency_ns <= 0:
        return np.nan
    return 1e9 / latency_ns


def throughput_vectors_per_s(latency_ns):
    batches = throughput_batches_per_s(latency_ns)
    if np.isnan(batches):
        return np.nan
    return batches * BATCH


def gops_per_second(model_spec: ModelSpec, latency_ns):
    if latency_ns is None or np.isnan(latency_ns) or latency_ns <= 0:
        return np.nan
    total_ops = 2.0 * BATCH * sum(layer.in_features * layer.out_features for layer in model_spec.layers)
    return total_ops / latency_ns


def force_make_jobs(output_dir: Path, jobs: int = BUILD_THREADS) -> None:
    makefile_path = output_dir / "Makefile"
    if not makefile_path.exists():
        return
    content = makefile_path.read_text()
    old = "JOBS := $(shell n=$$(nproc); echo $$(( n > 32 ? 32 : n )))"
    new = f"JOBS := {int(jobs)}"
    if old in content:
        makefile_path.write_text(content.replace(old, new))


def active_cores_path(output_dir: Path) -> Path:
    return output_dir / "Work" / "aie" / "active_cores.json"


def extract_layout_metrics(output_dir: Path) -> dict:
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
    coords = []
    for entry in payload.get("ActiveCores", []):
        for name in entry.keys():
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


def _features_from_bytes(byte_alignment: int, element_bytes: int) -> int:
    if byte_alignment <= 0:
        return 1
    return max(1, math.ceil(int(byte_alignment) / max(1, int(element_bytes))))


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return max(1, b)
    if b <= 0:
        return max(1, a)
    return abs(int(a) * int(b)) // math.gcd(int(a), int(b))


def _lcm_many(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = _lcm(result, int(value))
    return result


def _device_lane_bytes(device: DeviceSpec) -> int:
    norm = (device.generation or "").upper()
    if any(token in norm for token in ("AIE-ML", "AIE-MLV2", "MLV2", "XDNA", "AIE2")):
        return 16
    return 16


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return max(0, int(value))
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def input_slice_alignment(device: DeviceSpec, tile_k: int, element_bytes: int) -> int:
    return _lcm_many((2 * int(tile_k), _features_from_bytes(_device_lane_bytes(device), element_bytes), _features_from_bytes(4, element_bytes)))


def output_slice_alignment(device: DeviceSpec, tile_n: int, element_bytes: int) -> int:
    return _lcm_many((2 * int(tile_n), _features_from_bytes(_device_lane_bytes(device), element_bytes), _features_from_bytes(4, element_bytes)))


def enumerate_layer_candidates(layer: LayerSpec, device: DeviceSpec) -> list[LayerCandidate]:
    input_elem_bytes = max(1, math.ceil(dtype_bits(INPUT_DTYPE) / 8))
    weight_bytes = max(1, math.ceil(dtype_bits(WEIGHT_DTYPE) / 8))
    align_k = input_slice_alignment(device, FIXED_TILING["tile_k"], input_elem_bytes)
    align_n = output_slice_alignment(device, FIXED_TILING["tile_n"], input_elem_bytes)
    raw_total_work = int(layer.in_features) * int(layer.out_features)

    candidates = []
    for cas_length in range(1, int(device.max_mem_in_ports) + 1):
        for cas_num in range(1, int(device.max_mem_out_ports) + 1):
            input_slice_raw = math.ceil(int(layer.in_features) / int(cas_length))
            output_slice_raw = math.ceil(int(layer.out_features) / int(cas_num))
            if (input_slice_raw * input_elem_bytes) % 4 != 0:
                continue

            input_slice = _align_up(input_slice_raw, align_k)
            output_slice = _align_up(output_slice_raw, align_n)
            tile_bytes = input_slice * output_slice * weight_bytes
            if tile_bytes > int(device.weight_mem_bytes):
                continue
            if FIXED_TILING["tile_m"] > MAX_TILE_M or input_slice > MAX_TILE_K or output_slice > MAX_TILE_N:
                continue

            padded_total_work = input_slice * output_slice * int(cas_length) * int(cas_num)
            candidates.append(
                LayerCandidate(
                    cas_length=int(cas_length),
                    cas_num=int(cas_num),
                    input_slice_raw=int(input_slice_raw),
                    output_slice_raw=int(output_slice_raw),
                    input_slice=int(input_slice),
                    output_slice=int(output_slice),
                    tile_bytes=int(tile_bytes),
                    tile_workload=int(input_slice * output_slice),
                    padded_total_work=int(padded_total_work),
                    raw_total_work=int(raw_total_work),
                    padding_waste=int(padded_total_work - raw_total_work),
                )
            )

    pruned = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other == candidate:
                continue
            if (
                other.tile_workload <= candidate.tile_workload
                and other.padded_total_work <= candidate.padded_total_work
                and other.cas_length <= candidate.cas_length
                and other.cas_num <= candidate.cas_num
                and (
                    other.tile_workload < candidate.tile_workload
                    or other.padded_total_work < candidate.padded_total_work
                    or other.cas_length < candidate.cas_length
                    or other.cas_num < candidate.cas_num
                )
            ):
                dominated = True
                break
        if not dominated:
            pruned.append(candidate)

    return sorted(
        pruned,
        key=lambda cand: (
            cand.tile_workload,
            cand.padding_waste,
            -cand.cas_length,
            cand.cas_num,
            -cand.parallel_factor,
        ),
    )


def layer_shapes_summary(model_spec: ModelSpec) -> str:
    return "|".join(f"{layer.in_features}x{layer.out_features}" for layer in model_spec.layers)


def validate_model_specs(model_specs: list[ModelSpec]) -> None:
    for model_spec in model_specs:
        if not model_spec.layers:
            raise ValueError(f"{model_spec.name}: model must contain at least one layer.")
        for layer_index, layer in enumerate(model_spec.layers):
            if layer.in_features <= 0 or layer.out_features <= 0:
                raise ValueError(f"{model_spec.name}: layer {layer_index} has non-positive dimensions.")
        for layer_index in range(len(model_spec.layers) - 1):
            left = model_spec.layers[layer_index]
            right = model_spec.layers[layer_index + 1]
            if int(left.out_features) != int(right.in_features):
                raise ValueError(
                    f"{model_spec.name}: layer {layer_index} out={left.out_features} does not match "
                    f"layer {layer_index + 1} in={right.in_features}."
                )


def layer_parallelism_summary(model_candidate: ModelCandidate) -> str:
    return "|".join(f"{candidate.cas_length}x{candidate.cas_num}" for candidate in model_candidate.layer_candidates)


def model_signature(model_spec: ModelSpec, model_candidate: ModelCandidate) -> str:
    payload = {
        "model_name": model_spec.name,
        "layers": [
            {
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "relu": bool(layer.relu),
                "batchnorm": bool(layer.batchnorm),
            }
            for layer in model_spec.layers
        ],
        "parallelism": [
            {"cas_length": int(candidate.cas_length), "cas_num": int(candidate.cas_num)}
            for candidate in model_candidate.layer_candidates
        ],
        "tiling": {key: int(value) for key, value in FIXED_TILING.items()},
        "batch": int(BATCH),
        "input_dtype": INPUT_DTYPE,
        "weight_dtype": WEIGHT_DTYPE,
        "output_dtype": OUTPUT_DTYPE,
        "platform": PLATFORM,
    }
    return str(zlib.adler32(json.dumps(payload, sort_keys=True).encode("utf-8")) & 0xFFFFFFFF)


def candidate_dir_name(model_candidate: ModelCandidate) -> str:
    parts = [
        f"cand_{model_candidate.rank:02d}",
        f"cols_{model_candidate.width_used}",
        f"wl_{model_candidate.bottleneck_workload}",
    ]
    for index, candidate in enumerate(model_candidate.layer_candidates):
        parts.append(f"L{index}")
        parts.append(f"PK_{candidate.cas_length}")
        parts.append(f"PN_{candidate.cas_num}")
    return "_".join(parts)


def model_output_dir(root: Path, model_spec: ModelSpec, model_candidate: ModelCandidate) -> Path:
    return root / sanitize_name(model_spec.name) / candidate_dir_name(model_candidate)


def model_input_dim(model_spec: ModelSpec) -> int:
    return int(model_spec.layers[0].in_features)


def model_output_dim(model_spec: ModelSpec) -> int:
    return int(model_spec.layers[-1].out_features)


def routing_slack_columns(model_spec: ModelSpec) -> int:
    depth = len(model_spec.layers)
    if depth >= 9:
        return 2
    if depth >= 8:
        return 1
    return 0


def preferred_width_cap(model_spec: ModelSpec, device: DeviceSpec) -> int:
    return max(1, int(device.usable_columns) - routing_slack_columns(model_spec))


def select_execution_candidates(ranked_candidates: list[ModelCandidate], max_candidates: int) -> list[ModelCandidate]:
    limit = max(1, int(max_candidates))
    if len(ranked_candidates) <= limit:
        return list(ranked_candidates)

    selected = []
    selected_ranks = set()

    def try_add(candidate: ModelCandidate) -> bool:
        if candidate.rank in selected_ranks:
            return False
        selected.append(candidate)
        selected_ranks.add(candidate.rank)
        return len(selected) >= limit

    if try_add(ranked_candidates[0]):
        return selected

    for accessor in (
        lambda cand: cand.width_used,
        lambda cand: cand.max_rows,
        lambda cand: cand.sum_parallel_factor,
        lambda cand: cand.sum_cas_length,
    ):
        seen_values = {accessor(candidate) for candidate in selected}
        for candidate in ranked_candidates[1:]:
            value = accessor(candidate)
            if value in seen_values:
                continue
            seen_values.add(value)
            if try_add(candidate):
                return selected

    for candidate in ranked_candidates[1:]:
        if try_add(candidate):
            return selected

    return selected


def build_ranked_model_candidates(
    model_spec: ModelSpec,
    device: DeviceSpec,
    beam_width: int = BEAM_WIDTH,
    top_logged: int = TOP_LOGGED_MODEL_CANDIDATES,
) -> tuple[list[list[LayerCandidate]], list[ModelCandidate]]:
    layer_candidates = [enumerate_layer_candidates(layer, device) for layer in model_spec.layers]
    if any(not candidates for candidates in layer_candidates):
        return layer_candidates, []

    min_widths = [min(candidate.width for candidate in candidates) for candidates in layer_candidates]
    suffix_min_widths = [0] * (len(min_widths) + 1)
    for index in range(len(min_widths) - 1, -1, -1):
        suffix_min_widths[index] = suffix_min_widths[index + 1] + int(min_widths[index])

    beam = [
        {
            "selected": [],
            "used_width": 0,
            "workloads": [],
            "padding_waste": 0,
            "sum_cas_length": 0,
            "sum_parallel_factor": 0,
            "max_rows": 0,
        }
    ]
    soft_width_cap = preferred_width_cap(model_spec, device)

    def state_score(state):
        if state["workloads"]:
            bottleneck = max(state["workloads"])
            imbalance = sum(bottleneck - workload for workload in state["workloads"])
        else:
            bottleneck = 0
            imbalance = 0
        used_width = int(state["used_width"])
        routing_width_excess = max(0, used_width - soft_width_cap)
        routing_width_gap = max(0, soft_width_cap - used_width)
        return (
            bottleneck,
            imbalance,
            routing_width_excess,
            state["padding_waste"],
            routing_width_gap,
            state["max_rows"],
            -min(used_width, soft_width_cap),
            -state["sum_cas_length"],
            -state["sum_parallel_factor"],
        )

    for layer_index, candidates in enumerate(layer_candidates):
        next_beam = []
        remaining_layers = len(layer_candidates) - layer_index - 1
        for state in beam:
            for candidate in candidates:
                used_width = int(state["used_width"]) + candidate.width + (1 if state["selected"] else 0)
                min_future_width = suffix_min_widths[layer_index + 1] + remaining_layers
                if used_width + min_future_width > int(device.usable_columns):
                    continue
                next_beam.append(
                    {
                        "selected": [*state["selected"], candidate],
                        "used_width": used_width,
                        "workloads": [*state["workloads"], candidate.tile_workload],
                        "padding_waste": int(state["padding_waste"]) + candidate.padding_waste,
                        "sum_cas_length": int(state["sum_cas_length"]) + candidate.cas_length,
                        "sum_parallel_factor": int(state["sum_parallel_factor"]) + candidate.parallel_factor,
                        "max_rows": max(int(state["max_rows"]), candidate.cas_num),
                    }
                )
        beam = sorted(next_beam, key=state_score)[: max(1, int(beam_width))]
        if not beam:
            break

    ranked = []
    for rank, state in enumerate(sorted(beam, key=state_score)[: max(1, int(top_logged))]):
        if len(state["selected"]) != len(model_spec.layers):
            continue
        workloads = list(state["workloads"])
        bottleneck = max(workloads)
        imbalance = sum(bottleneck - workload for workload in workloads)
        ranked.append(
            ModelCandidate(
                rank=int(rank),
                layer_candidates=tuple(state["selected"]),
                width_used=int(state["used_width"]),
                bottleneck_workload=int(bottleneck),
                workload_imbalance=int(imbalance),
                total_padding_waste=int(state["padding_waste"]),
                sum_cas_length=int(state["sum_cas_length"]),
                sum_parallel_factor=int(state["sum_parallel_factor"]),
                max_rows=int(state["max_rows"]),
            )
        )

    return layer_candidates, ranked


def generate_dense_params(model_spec: ModelSpec) -> list[DenseParams]:
    rng = np.random.default_rng(stable_seed(model_spec.name))
    params = []
    for layer_index, layer in enumerate(model_spec.layers):
        scale = math.sqrt(6.0 / max(1, int(layer.in_features) + int(layer.out_features)))
        kernel = rng.uniform(-0.6 * scale, 0.6 * scale, size=(layer.in_features, layer.out_features)).astype(np.float32)
        bias = rng.uniform(-0.1, 0.1, size=(layer.out_features,)).astype(np.float32)
        batchnorm_folded = False
        if layer.batchnorm:
            gamma = rng.uniform(0.75, 1.25, size=(layer.out_features,)).astype(np.float32)
            beta = rng.uniform(-0.1, 0.1, size=(layer.out_features,)).astype(np.float32)
            moving_mean = rng.uniform(-0.1, 0.1, size=(layer.out_features,)).astype(np.float32)
            moving_var = rng.uniform(0.75, 1.25, size=(layer.out_features,)).astype(np.float32)
            eps = 1e-3
            scale_vec = gamma / np.sqrt(moving_var + eps)
            kernel = kernel * scale_vec.reshape(1, -1)
            bias = (bias - moving_mean) * scale_vec + beta
            batchnorm_folded = True
        params.append(DenseParams(kernel=kernel, bias=bias, batchnorm_folded=batchnorm_folded))
    return params


def build_qkeras_model(model_spec: ModelSpec, dense_params: list[DenseParams]) -> Sequential:
    tf.keras.backend.clear_session()
    input_bits = dtype_bits(INPUT_DTYPE)
    weight_bits = dtype_bits(WEIGHT_DTYPE)
    output_bits = dtype_bits(OUTPUT_DTYPE)
    input_frac = dtype_frac(INPUT_DTYPE)
    output_frac = dtype_frac(OUTPUT_DTYPE)

    layers = [
        QActivation(
            quantized_bits(input_bits, input_frac),
            name="input_quant",
            input_shape=(model_input_dim(model_spec),),
        )
    ]
    for layer_index, layer in enumerate(model_spec.layers):
        layers.append(
            QDense(
                layer.out_features,
                name=f"dense_{layer_index}",
                kernel_quantizer=quantized_bits(weight_bits, 0, alpha=1),
                bias_quantizer=quantized_bits(output_bits, output_frac, alpha=1),
            )
        )
        activation = quantized_relu(output_bits, output_frac) if layer.relu else quantized_bits(output_bits, output_frac)
        layers.append(QActivation(activation, name=f"act_{layer_index}"))

    model = Sequential(layers)
    model.compile(optimizer="adam", loss="mse")

    for layer_index, params in enumerate(dense_params):
        model.get_layer(f"dense_{layer_index}").set_weights([params.kernel, params.bias])
    return model


def build_aie_model(model: Sequential, model_candidate: ModelCandidate, output_dir: Path):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    layer_cfg = cfg.setdefault("LayerName", {})
    for layer_index, candidate in enumerate(model_candidate.layer_candidates):
        dense_name = f"dense_{layer_index}"
        layer_cfg.setdefault(dense_name, {})
        layer_cfg[dense_name]["parallelism"] = {
            "cas_num": int(candidate.cas_num),
            "cas_length": int(candidate.cas_length),
        }
        layer_cfg[dense_name]["tiling"] = {key: int(value) for key, value in FIXED_TILING.items()}

    aie_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=str(output_dir),
        backend="aie",
        project_name=PROJECT_NAME,
        batch_size=BATCH,
        iterations=ITERS,
        part=PLATFORM,
    )
    aie_model.compile()
    return aie_model


def predict_hls_reference(model: Sequential, x: np.ndarray) -> np.ndarray:
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    with tempfile.TemporaryDirectory(prefix="physics_models_hls_ref_") as tmp_dir:
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=cfg,
            output_dir=tmp_dir,
            project_name="physics_models_hls_ref",
            bit_exact=True,
        )
        hls_model.compile()
        return np.asarray(hls_model.predict(x))


def fixed_input(model_spec: ModelSpec) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(model_spec.name + "_input"))
    return rng.uniform(-1.0, 1.0, size=(BATCH, model_input_dim(model_spec))).astype(np.float32)


def bn_handling_note(model_spec: ModelSpec) -> str:
    if any(layer.batchnorm for layer in model_spec.layers):
        return "batchnorm_folded_into_dense_weights_and_bias"
    return "no_batchnorm_layers"


def measure_candidate(
    model_spec: ModelSpec,
    model_candidate: ModelCandidate,
    output_dir: Path,
    rerun_failed: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_signature = model_signature(model_spec, model_candidate)

    cached = load_result(output_dir)
    if cached and cached.get("status") == "success" and cached.get("model_signature") in (None, expected_signature):
        return cached, "cached"
    if cached and cached.get("status") == "failed" and not rerun_failed and cached.get("model_signature") in (None, expected_signature):
        return cached, "failed"

    try:
        seed_everything()
        dense_params = generate_dense_params(model_spec)
        model = build_qkeras_model(model_spec, dense_params)
        x = fixed_input(model_spec)
        reference = predict_hls_reference(model, x)
        aie_model = build_aie_model(model, model_candidate, output_dir)
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
            "throughput_batches_per_s": throughput_batches_per_s(latency_ns),
            "throughput_vectors_per_s": throughput_vectors_per_s(latency_ns),
            "gops_per_s": gops_per_second(model_spec, latency_ns),
            "output_ok": output_ok,
            "max_abs_error": max_abs_error,
            "error": error,
            "bn_handling": bn_handling_note(model_spec),
            "reference_kind": "hls_bit_exact",
            "candidate_parallelism": layer_parallelism_summary(model_candidate),
            "candidate_width_used": model_candidate.width_used,
            "candidate_bottleneck_workload": model_candidate.bottleneck_workload,
            "candidate_workload_imbalance": model_candidate.workload_imbalance,
            "candidate_padding_waste": model_candidate.total_padding_waste,
            "model_signature": expected_signature,
        }
        payload.update(extract_layout_metrics(output_dir))
        save_result(output_dir, payload)
        return payload, "new"
    except Exception as exc:
        payload = {
            "status": "failed",
            "latency_ns": np.nan,
            "throughput_batches_per_s": np.nan,
            "throughput_vectors_per_s": np.nan,
            "gops_per_s": np.nan,
            "output_ok": False,
            "max_abs_error": np.nan,
            "error": failure_summary(output_dir, str(exc)),
            "bn_handling": bn_handling_note(model_spec),
            "reference_kind": "hls_bit_exact",
            "candidate_parallelism": layer_parallelism_summary(model_candidate),
            "candidate_width_used": model_candidate.width_used,
            "candidate_bottleneck_workload": model_candidate.bottleneck_workload,
            "candidate_workload_imbalance": model_candidate.workload_imbalance,
            "candidate_padding_waste": model_candidate.total_padding_waste,
            "model_signature": expected_signature,
        }
        payload.update(extract_layout_metrics(output_dir))
        save_result(output_dir, payload)
        return payload, "failed"


def csv_row(model_spec: ModelSpec, model_candidate: ModelCandidate, output_dir: Path, result: dict) -> dict:
    layout = extract_layout_metrics(output_dir)
    return {
        "model_name": model_spec.name,
        "candidate_rank": model_candidate.rank,
        "candidate_config": candidate_dir_name(model_candidate),
        "num_layers": len(model_spec.layers),
        "batch": BATCH,
        "input_features": model_input_dim(model_spec),
        "output_features": model_output_dim(model_spec),
        "layer_shapes": layer_shapes_summary(model_spec),
        "layer_parallelism": layer_parallelism_summary(model_candidate),
        "tile_m": FIXED_TILING["tile_m"],
        "tile_k": FIXED_TILING["tile_k"],
        "tile_n": FIXED_TILING["tile_n"],
        "build_threads": BUILD_THREADS,
        "width_budget": load_device_spec().usable_columns,
        "width_used": model_candidate.width_used,
        "width_slack": load_device_spec().usable_columns - model_candidate.width_used,
        "max_rows_per_layer": model_candidate.max_rows,
        "sum_cas_length": model_candidate.sum_cas_length,
        "sum_parallel_factor": model_candidate.sum_parallel_factor,
        "bottleneck_tile_workload": model_candidate.bottleneck_workload,
        "workload_imbalance": model_candidate.workload_imbalance,
        "total_padding_waste": model_candidate.total_padding_waste,
        "bn_handling": bn_handling_note(model_spec),
        "reference_kind": result.get("reference_kind", "hls_bit_exact"),
        "latency_ns": result["latency_ns"],
        "latency_us": np.nan if np.isnan(result["latency_ns"]) else result["latency_ns"] / 1000.0,
        "throughput_batches_per_s": result["throughput_batches_per_s"],
        "throughput_vectors_per_s": result["throughput_vectors_per_s"],
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


def fieldnames() -> list[str]:
    return [
        "model_name",
        "candidate_rank",
        "candidate_config",
        "num_layers",
        "batch",
        "input_features",
        "output_features",
        "layer_shapes",
        "layer_parallelism",
        "tile_m",
        "tile_k",
        "tile_n",
        "build_threads",
        "width_budget",
        "width_used",
        "width_slack",
        "max_rows_per_layer",
        "sum_cas_length",
        "sum_parallel_factor",
        "bottleneck_tile_workload",
        "workload_imbalance",
        "total_padding_waste",
        "bn_handling",
        "reference_kind",
        "latency_ns",
        "latency_us",
        "throughput_batches_per_s",
        "throughput_vectors_per_s",
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
    ]


def write_models_log(
    log_path: Path,
    device: DeviceSpec,
    model_specs: list[ModelSpec],
    per_layer_candidates: dict[str, list[list[LayerCandidate]]],
    ranked_candidates: dict[str, list[ModelCandidate]],
) -> None:
    lines = [
        f"platform: {PLATFORM}",
        f"usable_columns: {device.usable_columns} (absolute cols {device.column_start}..{device.columns - 1})",
        f"api_tiling: {FIXED_TILING}",
        f"dtype: {INPUT_DTYPE}x{WEIGHT_DTYPE}={OUTPUT_DTYPE}",
        f"build_threads: {BUILD_THREADS}",
        "routing_width_soft_cap: reserve 1 column of slack for models with >=8 layers and 2 columns for models with >=9 layers.",
        "batchnorm_handling: batchnorm layers are folded into the preceding dense weights/bias for AIE deployment and reference checking.",
        "",
    ]

    for model_spec in model_specs:
        lines.append(f"## {model_spec.name}")
        lines.append(f"layers: {layer_shapes_summary(model_spec)}")
        lines.append(f"preferred_width_cap: {preferred_width_cap(model_spec, device)}")
        lines.append(f"relu_layers: {[index for index, layer in enumerate(model_spec.layers) if layer.relu]}")
        lines.append(f"batchnorm_layers: {[index for index, layer in enumerate(model_spec.layers) if layer.batchnorm]}")
        lines.append("")

        for layer_index, layer_candidates in enumerate(per_layer_candidates[model_spec.name]):
            layer = model_spec.layers[layer_index]
            lines.append(
                f"layer_{layer_index}: in={layer.in_features} out={layer.out_features} relu={int(layer.relu)} batchnorm={int(layer.batchnorm)}"
            )
            for candidate in layer_candidates:
                lines.append(
                    "  "
                    f"PK={candidate.cas_length} PN={candidate.cas_num} "
                    f"in_slice={candidate.input_slice_raw}->{candidate.input_slice} "
                    f"out_slice={candidate.output_slice_raw}->{candidate.output_slice} "
                    f"tile_workload={candidate.tile_workload} "
                    f"eff={candidate.efficiency:.4f} "
                    f"padding_waste={candidate.padding_waste}"
                )
            lines.append("")

        lines.append("top_model_candidates:")
        for model_candidate in ranked_candidates[model_spec.name]:
            lines.append(
                "  "
                f"rank={model_candidate.rank} "
                f"width_used={model_candidate.width_used} "
                f"bottleneck_workload={model_candidate.bottleneck_workload} "
                f"workload_imbalance={model_candidate.workload_imbalance} "
                f"padding_waste={model_candidate.total_padding_waste} "
                f"parallelism={layer_parallelism_summary(model_candidate)}"
            )
        lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines).rstrip() + "\n")


def append_log_line(log_path: Path, line: str) -> None:
    with log_path.open("a") as handle:
        handle.write(line.rstrip() + "\n")


def selected_model_specs(all_specs: list[ModelSpec], names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return all_specs
    selected = {sanitize_name(name) for name in names}
    return [spec for spec in all_specs if sanitize_name(spec.name) in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-candidates", type=int, default=TOP_EXECUTION_CANDIDATES)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    device = load_device_spec()
    specs = selected_model_specs(MODEL_SPECS, args.models)
    validate_model_specs(specs)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_layer_candidates = {}
    all_ranked_candidates = {}
    execution_candidates = {}
    selected_runs = []
    for model_spec in specs:
        layer_candidates, ranked = build_ranked_model_candidates(
            model_spec,
            device,
            beam_width=max(1, int(args.beam_width)),
            top_logged=max(TOP_LOGGED_MODEL_CANDIDATES, int(args.max_candidates) * 32),
        )
        all_layer_candidates[model_spec.name] = layer_candidates
        all_ranked_candidates[model_spec.name] = ranked
        execution_candidates[model_spec.name] = select_execution_candidates(ranked, args.max_candidates)
        for model_candidate in execution_candidates[model_spec.name]:
            output_dir = model_output_dir(args.output_root, model_spec, model_candidate)
            selected_runs.append((output_dir, model_signature(model_spec, model_candidate)))

    log_path = args.output_root / LOG_PATH.name if args.output_root != OUTPUT_ROOT else LOG_PATH
    write_models_log(log_path, device, specs, all_layer_candidates, all_ranked_candidates)

    if needs_execution(selected_runs, args.rerun_failed):
        source_vitis(VITIS_SETTINGS)

    rows = []
    csv_path = args.output_root / CSV_PATH.name if args.output_root != OUTPUT_ROOT else CSV_PATH
    for model_spec in specs:
        ranked = all_ranked_candidates[model_spec.name]
        selected = execution_candidates[model_spec.name]
        if not ranked:
            empty_layers = [index for index, candidates in enumerate(all_layer_candidates[model_spec.name]) if not candidates]
            if empty_layers:
                append_log_line(
                    log_path,
                    f"{model_spec.name}: no valid per-layer candidate for layers {empty_layers} "
                    f"under dtype={INPUT_DTYPE}, api={FIXED_TILING}, and current raw-slice alignment rules.",
                )
            else:
                append_log_line(log_path, f"{model_spec.name}: no one-band candidate found within {device.usable_columns} usable columns.")
            continue

        append_log_line(log_path, f"{model_spec.name}: executing {len(selected)} ranked candidates.")
        for model_candidate in selected:
            output_dir = model_output_dir(args.output_root, model_spec, model_candidate)
            result, execution_status = measure_candidate(model_spec, model_candidate, output_dir, args.rerun_failed)
            rows.append(csv_row(model_spec, model_candidate, output_dir, result))
            save_rows_csv(csv_path, rows, fieldnames())

            latency_summary = "nan" if np.isnan(result["latency_ns"]) else f"{result['latency_ns']:.3f}"
            vectors_summary = (
                "nan" if np.isnan(result["throughput_vectors_per_s"]) else f"{result['throughput_vectors_per_s']:.6f}"
            )
            append_log_line(
                log_path,
                f"run model={model_spec.name} rank={model_candidate.rank} "
                f"status={execution_status}/{result['status']} "
                f"parallelism={layer_parallelism_summary(model_candidate)} "
                f"latency_ns={latency_summary} "
                f"throughput_vectors_per_s={vectors_summary} "
                f"error={result['error'] or ''}",
            )
            print(
                f"[{execution_status}] model={model_spec.name:>32} "
                f"rank={model_candidate.rank:>2} "
                f"cols={model_candidate.width_used:>2}/{device.usable_columns} "
                f"parallelism={layer_parallelism_summary(model_candidate):<40} "
                f"latency_ns={latency_summary} "
                f"throughput_vectors_per_s={vectors_summary} "
                f"output_ok={result['output_ok']} status={result['status']}"
                + (f" error={result['error']}" if result["error"] else "")
            )


if __name__ == "__main__":
    main()
