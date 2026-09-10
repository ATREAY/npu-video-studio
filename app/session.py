"""ONNX Runtime session wrapper with QNN(NPU) -> CPU provider fallback."""
from __future__ import annotations

import os
import time
import pathlib
import numpy as np
import onnxruntime as ort

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"

# On a Snapdragon device this env var points ORT at the HTP (NPU) backend .dll/.so.
# On dev boxes it is unset and we go straight to CPU.
_QNN_BACKEND = os.environ.get("QNN_BACKEND_PATH", "QnnHtp.dll")

# cgroup-restricted machines (our cluster) reject ORT's default thread affinity;
# pin the count explicitly. Overridable.
_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "4"))


def available_providers() -> list[str]:
    return list(ort.get_available_providers())


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
            providers.append(QNN_EP)
            prov_opts.append({
                "backend_path": _QNN_BACKEND,
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

        self.sess = ort.InferenceSession(self.path, sess_options=so,
                                         providers=providers,
                                         provider_options=prov_opts)
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
