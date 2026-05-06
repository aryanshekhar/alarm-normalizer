"""
Inference module — real-time KPI stream processing.

Key exports:
    SimbaInferenceEngine    — sliding window buffer + model inference
    InferenceResult         — result container for one inference window
    CellDetection           — per-cell anomaly detection with fault type and confidence
    SlidingWindowBuffer     — fixed-size deque buffer for live KPI streams
"""
from inference.inference_engine import (
    SimbaInferenceEngine,
    InferenceResult,
    CellDetection,
    SlidingWindowBuffer,
)
