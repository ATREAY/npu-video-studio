"""ONNX Runtime session helper. Runs on the Snapdragon X device (Windows on ARM).

Loads a compiled .onnx model on the QNN Execution Provider (Hexagon NPU / HTP).
Falls back to CPU EP so the pipeline still runs on a judge's non-Snapdragon machine.
"""
from __future__ import annotations

import pathlib
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # allow import on the cluster where ORT-QNN isn't installed
    ort = None

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"


class NpuModel:
    def __init__(self, onnx_path: str | pathlib.Path, prefer_npu: bool = True,
                 htp_performance: str = "burst"):
        if ort is None:
            raise RuntimeError("onnxruntime not available in this environment")
        self.path = str(onnx_path)
        providers, opts = [], []
        if prefer_npu and QNN_EP in ort.get_available_providers():
            providers.append(QNN_EP)
            opts.append({"backend_path": "QnnHtp.dll",
                         "htp_performance_mode": htp_performance,
                         "htp_graph_finalization_optimization_mode": "3"})
        providers.append(CPU_EP)
        opts.append({})
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.path, sess_options=so,
                                         providers=providers, provider_options=opts)
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.on_npu = self.sess.get_providers()[0] == QNN_EP

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        feed = {name: arr for name, arr in zip(self.inputs, arrays)}
        return self.sess.run(None, feed)

    def providers(self) -> list[str]:
        return self.sess.get_providers()
