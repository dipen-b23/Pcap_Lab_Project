from __future__ import annotations
import io
import json
import sys
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from backend import backend_info, build_command, run_backend  # noqa: E402

st.set_page_config(page_title="Parallel Bilateral Denoising", layout="wide")
st.title("Parallel Image Denoising")
st.caption("C11 bilateral filtering with capability-aware OpenMP, MPI, CUDA, and hybrid modes")
name = "denoise_cuda.exe" if sys.platform == "win32" else "denoise_cuda"
default_exe = next((ROOT / folder / name for folder in ("build-verified", "build") if (ROOT / folder / name).is_file()), ROOT / "build" / name)
exe = Path(st.sidebar.text_input("Backend executable", str(default_exe)))
try:
    capabilities = backend_info(exe)
    st.sidebar.json(capabilities)
except Exception as exc:
    capabilities = {}
    st.sidebar.error(str(exc))

upload = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg", "pgm", "ppm"])
size = st.sidebar.selectbox("Synthetic size", [128, 256, 512, 1024, 2048, 4096], index=1)
noise = st.sidebar.slider("Noise sigma", 0.0, 75.0, 25.0)
seed = st.sidebar.number_input("Noise seed", min_value=0, value=42)
window = st.sidebar.selectbox("Window", [5, 9, 15], index=1)
ss = st.sidebar.number_input("Spatial sigma", min_value=0.1, value=3.0)
sr = st.sidebar.number_input("Range sigma", min_value=0.1, value=25.0)
threads = st.sidebar.selectbox("OpenMP threads", [1, 2, 4, 8], index=2)
processes = st.sidebar.selectbox("MPI processes", [1, 2, 4], index=1)
cpu_percent = st.sidebar.slider(
    "Hybrid CPU share (%)",
    min_value=0,
    max_value=100,
    value=1,
    step=1
)
repetitions = st.sidebar.number_input("Benchmark repetitions", 1, 20, 5)
available = ["serial"] + (["openmp"] if capabilities.get("openmp", {}).get("runtime") else [])
available += [m for m in ("cuda", "hybrid") if capabilities.get(m, {}).get("runtime")]
mode = st.sidebar.selectbox("Execution mode", available)
config = {"mode": mode, "width": size, "height": size, "noise_sigma": noise, "seed": seed,
          "window": window, "sigma_spatial": ss, "sigma_range": sr, "threads": threads,
          "processes": processes, "cpu_percent": cpu_percent, "repetitions": repetitions}

if "runs" not in st.session_state: st.session_state.runs = []
if st.button("Run selected mode", type="primary", disabled=not capabilities):
    with tempfile.TemporaryDirectory(prefix="denoise-") as directory:
        work = Path(directory); input_path = None
        if upload:
            upload.seek(0); image = Image.open(upload).convert("L"); input_path = work / "input.pgm"; image.save(input_path)
        output, clean, noisy = work / f"{mode}.pgm", work / "clean.pgm", work / "noisy.pgm"
        command = build_command(exe, "denoise", config, input_path=input_path, output_path=output,
                                clean_path=clean, noisy_path=noisy)
        try:
            with st.spinner("Running native backend..."): result = run_backend(command)
            result["config"] = config; st.session_state.runs.append(result)
            cols = st.columns(3)
            for col, title, path in zip(cols, ("Clean", "Noisy", mode.title()), (clean, noisy, output)):
                col.image(Image.open(path), caption=title, width="stretch")
            st.success(f"Completed in {result['wall_ms']:.3f} ms")
            st.download_button("Download output", output.read_bytes(), file_name=output.name)
        except Exception as exc: st.error(str(exc))

if st.button("Generate noisy input", disabled=not capabilities):
    with tempfile.TemporaryDirectory(prefix="denoise-preview-") as directory:
        work = Path(directory); input_path = None
        if upload:
            upload.seek(0); image = Image.open(upload).convert("L"); input_path = work / "input.pgm"; image.save(input_path)
        preview_config = config | {"mode": "serial"}
        output, clean, noisy = work / "preview.pgm", work / "clean.pgm", work / "noisy.pgm"
        try:
            run_backend(build_command(exe, "denoise", preview_config, input_path=input_path,
                                      output_path=output, clean_path=clean, noisy_path=noisy))
            left, right = st.columns(2); left.image(Image.open(clean), caption="Clean"); right.image(Image.open(noisy), caption="Noisy")
        except Exception as exc: st.error(str(exc))

if st.button("Run all available modes", disabled=not capabilities):
    with tempfile.TemporaryDirectory(prefix="denoise-all-") as directory:
        work = Path(directory); input_path = None
        if upload:
            upload.seek(0); image = Image.open(upload).convert("L"); input_path = work / "input.pgm"; image.save(input_path)
        clean, noisy = work / "clean.pgm", work / "noisy.pgm"
        columns = st.columns(min(3, len(available)))
        for index, current_mode in enumerate(available):
            output = work / f"{current_mode}.pgm"
            try:
                result = run_backend(build_command(exe, "denoise", config | {"mode": current_mode},
                                                    input_path=input_path, output_path=output,
                                                    clean_path=clean, noisy_path=noisy))
                result["config"] = config | {"mode": current_mode}; st.session_state.runs.append(result)
                columns[index % len(columns)].image(Image.open(output), caption=current_mode.title())
            except Exception as exc: st.error(f"{current_mode}: {exc}")

if st.button("Validate selected mode", disabled=not capabilities):
    try:
        validation = run_backend(build_command(exe, "validate", config))
        st.session_state.validation = validation; st.json(validation)
    except Exception as exc: st.error(str(exc))

if st.button("Run benchmark", disabled=not capabilities):
    command = build_command(exe, "benchmark", config)
    try:
        result = run_backend(command); result["config"] = config; st.session_state.runs.append(result)
    except Exception as exc: st.error(str(exc))

if st.session_state.runs:
    rows = []
    for run in st.session_state.runs:
        row = {k: v for k, v in run.items() if not isinstance(v, dict)}
        row.update({k: v for k, v in run.get("config", {}).items()})
        rows.append(row)
    frame = pd.DataFrame(rows); st.dataframe(frame, width="stretch")
    if "wall_ms" in frame:
        figure, axis = plt.subplots(); frame.plot.bar(x="mode", y="wall_ms", ax=axis, legend=False); axis.set_ylabel("Time (ms)"); st.pyplot(figure)
        serial = frame.loc[frame["mode"] == "serial", "wall_ms"]
        if not serial.empty:
            frame["speedup"] = float(serial.iloc[-1]) / frame["wall_ms"]
        frame["throughput_mpix_s"] = frame["width"] * frame["height"] / (frame["wall_ms"] * 1000)
        for x, y, title in (("threads", "wall_ms", "Threads vs time"), ("width", "wall_ms", "Image size vs time"),
                            ("window", "wall_ms", "Window vs time"), ("mode", "throughput_mpix_s", "Throughput")):
            if x in frame and frame[x].nunique() > 1:
                fig, ax = plt.subplots(); frame.plot(x=x, y=y, marker="o", ax=ax); ax.set_title(title); st.pyplot(fig)
        if {"h2d_ms", "kernel_ms", "d2h_ms"} <= set(frame.columns):
            fig, ax = plt.subplots(); frame.plot.bar(x="mode", y=["h2d_ms", "kernel_ms", "d2h_ms"], stacked=True, ax=ax); ax.set_title("CUDA transfer and kernel time"); st.pyplot(fig)
    csv_data = frame.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_data, "benchmark_results.csv", "text/csv")
    st.download_button("Download configuration", json.dumps(config, indent=2), "experiment.json", "application/json")
    if "validation" in st.session_state:
        st.download_button("Download validation", json.dumps(st.session_state.validation, indent=2), "validation.json", "application/json")
    summary = f"# Experiment Summary\n\nMode: {mode}\n\nRuns: {len(rows)}\n"
    st.download_button("Download Markdown summary", summary, "summary.md", "text/markdown")
