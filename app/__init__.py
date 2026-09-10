"""NPU Video Studio — on-device webcam enhancement pipeline.

Runs anywhere: ONNX Runtime with the QNN Execution Provider (Hexagon NPU) on a
Snapdragon X PC, transparently falling back to the CPU EP elsewhere (dev boxes,
judges without Snapdragon hardware).
"""
__version__ = "0.1.0"
