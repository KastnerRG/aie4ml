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

IN_FEATURES = 128
OUT_FEATURES = 128
CAS_LEN = 8
CAS_NUM = 1
BATCH = 8
ITERS = 1
PLATFORM = "xilinx_vek280_base_202520_1"
VITIS_SETTINGS = os.environ.get("VITIS_SETTINGS", "/tmp/tools/Xilinx/2025.2/Vitis/.settings64-Vitis.sh")
OUTPUT_DIR = Path(__file__).resolve().parent / f"single_layer_in{IN_FEATURES}_out{OUT_FEATURES}_c{CAS_NUM}x{CAS_LEN}"

np.random.seed(42)
tf.random.set_seed(42)

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

cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
cfg["LayerName"]["dense"]["parallelism"] = {"cas_num": CAS_NUM, "cas_length": CAS_LEN}

aie_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    output_dir=str(OUTPUT_DIR),
    backend="aie",
    project_name="single_layer",
    batch_size=BATCH,
    iterations=ITERS,
    part=PLATFORM,
)

aie_model.compile()
aie_model.build()

x = np.random.random((BATCH, IN_FEATURES)).astype(np.float32)
aie_model.predict(x, simulator="aie")
latency_ns = read_aie_report(aie_model)["fifo_latency"]["ns"]

print(f"IN_FEATURES={IN_FEATURES} OUT_FEATURES={OUT_FEATURES} CAS_LEN={CAS_LEN} CAS_NUM={CAS_NUM} latency_ns={latency_ns:.3f}")
