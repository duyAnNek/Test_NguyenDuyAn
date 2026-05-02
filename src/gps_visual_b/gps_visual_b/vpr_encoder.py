"""
Global image descriptor for VPR: ORB + VLAD-style encoding (CPU) or DINOv2 ONNX (CPU).

NetVLAD / EigenPlaces / AnyLoc motivate learned place descriptors; here ONNX path is
pluggable while ORB-VLAD provides a zero-weight fallback suitable for integration tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore


def l2n(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


class BaseVPREncoder(ABC):
    dim: int

    @abstractmethod
    def encode_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return float32 (D,) L2-normalized."""


class ORBVLADEncoder(BaseVPREncoder):
    """
    ORB local features aggregated into a single vector (VLAD-style residuals vs K centers).
    """

    def __init__(self, n_clusters: int = 16, max_features: int = 800) -> None:
        self.n_clusters = int(n_clusters)
        self.max_features = int(max_features)
        rng = np.random.RandomState(0)
        self._centers = rng.randn(self.n_clusters, 32).astype(np.float32)
        self._centers /= (np.linalg.norm(self._centers, axis=1, keepdims=True) + 1e-6)
        self.dim = self.n_clusters * 32
        self._orb = cv2.ORB_create(nfeatures=self.max_features, scaleFactor=1.2)

    def encode_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(desc) < 4:
            return l2n(np.zeros(self.dim, dtype=np.float32))
        d = desc.astype(np.float32)
        # assign each desc to nearest center (cosine in L2-normalized 32-D Hamming space approximated in L2)
        dists = np.linalg.norm(d[:, None, :] - self._centers[None, :, :], axis=2)
        a = np.argmin(dists, axis=1)
        vlad = np.zeros((self.n_clusters, 32), dtype=np.float32)
        for i in range(len(d)):
            k = int(a[i])
            vlad[k] += d[i] - self._centers[k]
        vec = vlad.reshape(-1)
        return l2n(vec)


class DinoV2OnnxEncoder(BaseVPREncoder):
    """
    DINOv2 (or ViT backbone) exported to ONNX; global average pool on spatial map.
    Expect input NCHW float32 RGB normalized by caller or model-specific.
    """

    def __init__(self, onnx_path: str, input_size: Tuple[int, int] = (224, 224)) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime required for DinoV2OnnxEncoder")
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size
        shp = self._session.get_inputs()[0].shape
        if isinstance(shp[2], int) and shp[2] > 0:
            self._input_size = (int(shp[2]), int(shp[3]) if isinstance(shp[3], int) and shp[3] > 0 else int(shp[2]))
        out0 = self._session.get_outputs()[0]
        # dim inferred after first run
        self.dim = 0
        self._out_name = out0.name

    def encode_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self._input_size
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = cv2.resize(rgb, (w, h))
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = self._session.run([self._out_name], {self._input_name: x})[0]
        y = np.asarray(out, dtype=np.float32)
        if y.ndim == 4:
            y = y.mean(axis=(2, 3))
        if y.ndim == 2:
            y = y[0]
        vec = y.reshape(-1)
        if self.dim == 0:
            self.dim = int(vec.shape[0])
        return l2n(vec)


def build_encoder(
    mode: str,
    onnx_path: Optional[str] = None,
    orb_vlad_clusters: int = 16,
) -> BaseVPREncoder:
    mode = (mode or "orb_vlad").lower()
    if mode == "dinov2_onnx":
        if not onnx_path:
            raise ValueError("dinov2_onnx requires onnx_path")
        return DinoV2OnnxEncoder(onnx_path)
    return ORBVLADEncoder(n_clusters=orb_vlad_clusters)
