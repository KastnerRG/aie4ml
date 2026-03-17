# Tutorial 1 FIFO Latency With ADF Event API Profiling

Scope: this note documents how `aie4ml` now measures FIFO latency for `tutorials/tutorial_1.ipynb`, why this method is valid, and where the implementation and proof artifacts live in the repo.

## Short answer

For `tutorial_1`, FIFO latency is now measured with the ADF Event API in AIE simulation, not by inventing timestamps in input files.

The current metric is:

- `PLIO_ifm_0 -> PLIO_ofm_0`
- measured with `event::io_stream_start_difference_cycles`
- recorded by the generated AIE testbench in `aiesimulator_output/data/fifo_latency.json`
- parsed by `read_aie_report(...)`

This is a real simulator/runtime metric because the cycle count comes from the ADF profiling API on platform IO objects. It is not inferred from plain text input files.

## 1. What ADF Event API profiling is

The ADF Event API is the profiling interface provided by the AIE runtime for graph objects and platform IO objects such as `PLIO`.

The relevant definitions are in the Vitis header:

- `tools/Xilinx/2025.2/Vitis/aietools/include/adf/new_frontend/adf.h`

Two key API facts matter here:

- `io_stream_start_difference_cycles` is defined as cycles elapsed between the first stream-running events of two platform IO objects.
- `start_profiling(io1, io2, option)` returns a profiling handle that can later be read with `read_profiling(handle)`.

This is exactly the primitive needed for an automated first-input to first-output latency proxy.

## 2. Why this method was chosen

This method avoids the main weakness of file-based timestamping:

- the input files used by the tutorial are plain `default case` PLIO files
- those files do not carry trustworthy simulator-observed ingress timestamps
- adding synthetic `T 0 ns` markers would only create an artificial reference point

The simulator logs confirm the tutorial still uses plain default-format inputs:

- `tutorials/proj_aie/AIESimulator.log`
- `tutorials/proj_aie_tuned/AIESimulator.log`

and the regenerated input files are plain values:

- `tutorials/proj_aie/data/ifm_c0.txt`
- `tutorials/proj_aie_tuned/data/ifm_c0.txt`
- `tutorials/proj_aie_tuned/data/ifm_c1.txt`

So the latency measurement must come from runtime profiling, not from the input files themselves.

## 3. What metric is being measured

For now, the metric is intentionally simple:

- `ifm0 -> ofm0`

More precisely:

- first running event on `PLIO_ifm_0`
- to first running event on `PLIO_ofm_0`
- measured in AIE cycles by `event::io_stream_start_difference_cycles`

For `proj_aie`, this is the natural single-input latency metric.

For `proj_aie_tuned`, this is a proxy, not the final multi-input latency definition, because the tuned design has:

- two input PLIOs: `PLIO_ifm_0`, `PLIO_ifm_1`
- one output PLIO: `PLIO_ofm_0`

The current proxy was chosen deliberately to keep the first implementation simple and automated.

## 4. How it works in the tutorial flow

The flow is:

1. `tutorial_1.ipynb` builds an AIE project through the `aie` backend.
2. `aie4ml` generates `app.cpp` for the project.
3. In AIE simulation builds, that generated `app.cpp` starts profiling on `dut.plio_data[0]` and `dut.plio_ofm[0]`.
4. The graph runs normally.
5. After the run completes, the app reads the profiling handle and writes the result to `aiesimulator_output/data/fifo_latency.json`.
6. `read_aie_report(...)` reads that JSON, converts cycles to ns using the compiled AIE frequency, and appends it to the report as `fifo_latency`.

This does not replace the existing throughput path.

Throughput is still computed from output `TLAST` intervals in:

- `aiesimulator_output/data/y_p*.txt`

So the current report contains:

- `output_interval`: steady-state throughput timing
- `fifo_latency`: first-stream-event latency proxy

## 5. The code path inside `aie4ml`

### Generated app instrumentation

The instrumentation is added in the firmware template:

- `src/aie4ml/templates/firmware/app.cpp.jinja`

The generated app:

- calls `event::start_profiling(dut.plio_data[0], dut.plio_ofm[0], event::io_stream_start_difference_cycles)`
- runs the graph
- calls `event::read_profiling(handle)`
- writes `fifo_latency.json`

The same logic can be inspected in the regenerated tutorial projects:

- `tutorials/proj_aie/app.cpp`
- `tutorials/proj_aie_tuned/app.cpp`

### Report parsing

Python-side parsing lives in:

- `src/aie4ml/simulation.py`

The key functions are:

- `read_aie_report(...)`
- `_analyze_aie_fifo_latency(...)`
- `_read_aie_frequency_mhz(...)`

What they do:

- load `aiesimulator_output/data/fifo_latency.json`
- expose the raw cycle count
- read the AIE frequency from `Work/app.aiecompile_summary`, with `AIESimulator.log` as fallback
- convert cycles to ns

### Input-file generation

Input-file generation remains plain and untimed:

- `src/aie4ml/aie_backend.py`
- `src/aie4ml/simulation.py`

The important point is that no latency value is derived from synthetic input timestamps anymore.

## 6. Proof points in the generated tutorial projects

### API is really used in generated app code

The tuned generated app contains the profiling calls and JSON writeout in:

- `tutorials/proj_aie_tuned/app.cpp`

and similarly for the 1-input project:

- `tutorials/proj_aie/app.cpp`

### Simulator produced real profiling output

The generated latency report files are:

- `tutorials/proj_aie/aiesimulator_output/data/fifo_latency.json`
- `tutorials/proj_aie_tuned/aiesimulator_output/data/fifo_latency.json`

These files contain:

- method name
- metric name
- input port
- output port
- validity flag
- measured cycle count

### Frequency used for cycle to ns conversion

The compile summaries contain the AIE frequency:

- `tutorials/proj_aie/Work/app.aiecompile_summary`
- `tutorials/proj_aie_tuned/Work/app.aiecompile_summary`

The simulator logs also print the same frequency:

- `tutorials/proj_aie/AIESimulator.log`
- `tutorials/proj_aie_tuned/AIESimulator.log`

### Throughput path is unchanged

The output timing files used for throughput remain:

- `tutorials/proj_aie/aiesimulator_output/data/y_p0.txt`
- `tutorials/proj_aie_tuned/aiesimulator_output/data/y_p0.txt`

`read_aie_report(...)` still parses `TLAST`-to-`TLAST` intervals from those files for `output_interval`.

## 7. Benefits of this method

Compared with synthetic file timestamps, the event-API method has these advantages:

- it measures a real runtime event on `PLIO` objects
- it does not depend on made-up input timestamps
- it is automated and notebook-friendly
- it does not require opening waveform or trace GUIs
- it is much lighter than a full trace-debug flow

Compared with full event trace, it is more limited, but much easier to integrate into `read_aie_report(...)`.

## 8. Current limitation

For `proj_aie_tuned`, the current latency is only a proxy:

- `PLIO_ifm_0 -> PLIO_ofm_0`

It is not yet the full multi-input latency definition such as:

- `max(PLIO_ifm_0, PLIO_ifm_1) -> PLIO_ofm_0`

That was a deliberate first step to keep the implementation minimal and robust.

## 9. Current tutorial results

From the latest executed `tutorial_1.ipynb`:

- `proj_aie`: `2825` cycles = `2260.0 ns`
- `proj_aie_tuned`: `2364` cycles = `1891.2 ns`

Those values come from:

- `read_aie_report("tutorials/proj_aie")`
- `read_aie_report("tutorials/proj_aie_tuned")`

while throughput remains:

- `proj_aie`: average output interval `557.511 ns`
- `proj_aie_tuned`: average output interval `487.467 ns`

So latency and throughput are now measured by two different, appropriate mechanisms:

- latency from ADF Event API profiling
- throughput from output packet timing
