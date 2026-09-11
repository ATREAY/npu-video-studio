"""ONNX Runtime session wrapper with QNN(NPU) -> CPU provider fallback.

QNN EP wiring (Windows-on-ARM only) — UNVALIDATED, see packaging/README.md:
`onnxruntime-qnn` (pip, win_arm64 wheel confirmed to exist) ships its own
`onnxruntime_providers_qnn.dll` + the Qualcomm `QnnHtp.dll`/`QnnCpu.dll`/`QnnGpu.dll`
backends in its OWN package directory — separate from wherever `onnxruntime` itself
lives. That matches Microsoft's plugin-EP pattern used for onnxruntime-directml /
onnxruntime-openvino: the provider DLL is found via the process DLL search path
(`os.add_dll_directory`, Windows-only), then `providers=["QNNExecutionProvider"]` with
`backend_path` pointing at the Qualcomm backend loads it. Written against the
documented API; never executed on real Windows-on-ARM hardware (this dev box is
headless Linux) — the whole block is try/except-guarded so any mismatch falls back to
CPU_EP cleanly instead of crashing. Confirm this end-to-end the moment hardware is
available; see packaging/README.md "first things to test".
"""
from __future__ import annotations

import os
import sys
import time
import pathlib
import numpy as np
import onnxruntime as ort

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"

# cgroup-restricted machines (our cluster) reject ORT's default thread affinity;
# pin the count explicitly. Overridable.
_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "4"))


def available_providers() -> list[str]:
    return list(ort.get_available_providers())


def _resolve_qnn_backend() -> str | None:
    """Best-effort: locate the Qualcomm HTP backend DLL and put its folder + the
    onnxruntime_qnn folder on the DLL search path. Returns the backend_path to pass
    as a QNN provider option, or None if onnxruntime-qnn isn't installed / this isn't
    Windows. Never raises — caller falls back to CPU on any problem.
    """
    override = os.environ.get("QNN_BACKEND_PATH")
    if override:
        return override
    if sys.platform != "win32":
        return None
    try:
        import onnxruntime_qnn as oq
    except ImportError:
        return None
    try:
        lib_dir = os.path.dirname(os.path.abspath(oq.__file__))
        if hasattr(os, "add_dll_directory"):        # py3.8+, Windows only
            os.add_dll_directory(lib_dir)
        return oq.get_qnn_htp_path()                # absolute path to QnnHtp.dll
    except Exception:
        return None


class Session:
    """One compiled ONNX model, run on the best available provider."""

    def __init__(self, onnx_path: str | pathlib.Path, prefer: str = "auto",
                 htp_performance: str = "burst"):
        self.path = str(onnx_path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)

        providers: list = []
        prov_opts: list = []
        want_npu = prefer in ("auto", "npu", "qnn")
        if want_npu and QNN_EP in ort.get_available_providers():
            backend = _resolve_qnn_backend()
            if backend is not None:
                providers.append(QNN_EP)
                prov_opts.append({
                    "backend_path": backend,
                    "htp_performance_mode": htp_performance,
                    "htp_graph_finalization_optimization_mode": "3",
                })
        providers.append(CPU_EP)
        prov_opts.append({})

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = _INTRA_THREADS
        so.inter_op_num_threads = 1
        so.log_severity_level = 3

        try:
            self.sess = ort.InferenceSession(self.path, sess_options=so,
                                             providers=providers,
                                             provider_options=prov_opts)
        except Exception:
            if len(providers) > 1:  # QNN construction failed -> retry CPU-only
                self.sess = ort.InferenceSession(self.path, sess_options=so,
                                                 providers=[CPU_EP], provider_options=[{}])
            else:
                raise
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.input_shapes = {i.name: i.shape for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.active_provider = self.sess.get_providers()[0]
        self.on_npu = self.active_provider == QNN_EP
        self.last_ms = 0.0

    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        feed = {n: a for n, a in zip(self.input_names, inputs)}
        t0 = time.perf_counter()
        out = self.sess.run(None, feed)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return out

    def __repr__(self) -> str:
        return (f"Session({os.path.basename(self.path)}, "
                f"provider={self.active_provider}, in={self.input_shapes})")
