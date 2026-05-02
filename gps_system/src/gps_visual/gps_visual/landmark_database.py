"""
Module 4 — Visual Landmark Database.

L_i = { id, class, p3D (x,y,z), d_visual (1D float descriptor), t_first, t_last, n_obs }

Persistence: SQLite (default) or pickle snapshot. Query combines cosine similarity
on descriptors with planar distance (radius_m) in the same map frame as stored p3D.
"""

from __future__ import annotations

import pickle
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return (v / n).astype(np.float32)


@dataclass
class LandmarkRecord:
    id: str
    class_name: str
    p3d: np.ndarray  # shape (3,) float64 map/body frame
    descriptor: np.ndarray  # float32 (D,) L2-normalized recommended
    t_first: float
    t_last: float
    n_obs: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_tuple_sql(self) -> Tuple:
        return (
            self.id,
            self.class_name,
            float(self.p3d[0]),
            float(self.p3d[1]),
            float(self.p3d[2]),
            self.descriptor.astype(np.float32).tobytes(),
            int(self.descriptor.shape[0]),
            self.t_first,
            self.t_last,
            self.n_obs,
            pickle.dumps(self.meta),
        )


class LandmarkDatabase:
    """
    Threading: single-writer recommended; SQLite connection per call for nodes.
    """

    SUPPORTED_CLASSES = frozenset(
        {
            "pothole",
            "traffic_sign",
            "building",
            "street_name",
            "lane_marking",
            "speed_hump",
            "manhole_cover",
        }
    )

    def __init__(self, sqlite_path: Optional[str] = None) -> None:
        self._sqlite_path = sqlite_path
        self._memory: Dict[str, LandmarkRecord] = {}
        self._dim: Optional[int] = None
        if sqlite_path:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite(sqlite_path)

    def _init_sqlite(self, path: str) -> None:
        con = sqlite3.connect(path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS landmarks (
                    id TEXT PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    px REAL, py REAL, pz REAL,
                    desc_blob BLOB NOT NULL,
                    desc_dim INTEGER NOT NULL,
                    t_first REAL, t_last REAL,
                    n_obs INTEGER NOT NULL,
                    meta_blob BLOB
                );
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_lm_class ON landmarks(class_name);"
            )
            con.commit()
        finally:
            con.close()

    def _conn(self) -> sqlite3.Connection:
        if not self._sqlite_path:
            raise RuntimeError("SQLite path not set (in-memory dict only).")
        return sqlite3.connect(self._sqlite_path)

    def add_landmark(
        self,
        class_name: str,
        p3d: Sequence[float],
        descriptor: np.ndarray,
        *,
        landmark_id: Optional[str] = None,
        merge_if_close: bool = False,
        merge_radius_m: float = 3.0,
        merge_cos_thresh: float = 0.92,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        if class_name not in self.SUPPORTED_CLASSES:
            raise ValueError(f"Unsupported class_name: {class_name}")
        d = _l2_normalize(np.asarray(descriptor, dtype=np.float32))
        if self._dim is None:
            self._dim = int(d.shape[0])
        elif int(d.shape[0]) != self._dim:
            raise ValueError(f"Descriptor dim mismatch: got {d.shape[0]} expected {self._dim}")
        p = np.asarray(p3d, dtype=np.float64).reshape(3)
        now = time.time()
        lid = landmark_id or str(uuid.uuid4())

        if merge_if_close and self._sqlite_path:
            hits = self.query_landmark(d, top_k=3, radius_m=merge_radius_m, query_xy=(float(p[0]), float(p[1])))
            for rec, sc in hits:
                if rec.class_name == class_name and sc >= merge_cos_thresh:
                    self._merge_update(rec.id, d, now, p)
                    return rec.id

        rec = LandmarkRecord(
            id=lid,
            class_name=class_name,
            p3d=p,
            descriptor=d,
            t_first=now,
            t_last=now,
            n_obs=1,
            meta=dict(meta or {}),
        )
        self._memory[lid] = rec
        if self._sqlite_path:
            con = self._conn()
            try:
                con.execute(
                    """
                    INSERT INTO landmarks (id, class_name, px, py, pz, desc_blob, desc_dim, t_first, t_last, n_obs, meta_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rec.to_tuple_sql(),
                )
                con.commit()
            finally:
                con.close()
        return lid

    def _merge_update(self, lid: str, d_new: np.ndarray, t: float, p_new: np.ndarray) -> None:
        rec = self._get_record(lid)
        if rec is None:
            return
        # running average of descriptor (then re-normalize)
        n = rec.n_obs + 1
        merged = _l2_normalize((rec.descriptor.astype(np.float64) * rec.n_obs + d_new.astype(np.float64)) / n)
        rec.descriptor = merged.astype(np.float32)
        rec.p3d = (rec.p3d * rec.n_obs + p_new) / n
        rec.t_last = t
        rec.n_obs = n
        self._memory[lid] = rec
        if self._sqlite_path:
            con = self._conn()
            try:
                con.execute(
                    """
                    UPDATE landmarks SET desc_blob=?, desc_dim=?, px=?, py=?, pz=?, t_last=?, n_obs=?
                    WHERE id=?
                    """,
                    (
                        rec.descriptor.tobytes(),
                        rec.descriptor.shape[0],
                        float(rec.p3d[0]),
                        float(rec.p3d[1]),
                        float(rec.p3d[2]),
                        rec.t_last,
                        rec.n_obs,
                        lid,
                    ),
                )
                con.commit()
            finally:
                con.close()

    def _get_record(self, lid: str) -> Optional[LandmarkRecord]:
        if lid in self._memory:
            return self._memory[lid]
        if not self._sqlite_path:
            return None
        con = self._conn()
        try:
            row = con.execute(
                "SELECT id, class_name, px, py, pz, desc_blob, desc_dim, t_first, t_last, n_obs, meta_blob FROM landmarks WHERE id=?",
                (lid,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return None
        meta = pickle.loads(row[10]) if row[10] else {}
        desc = np.frombuffer(row[5], dtype=np.float32, count=int(row[6]))
        rec = LandmarkRecord(
            id=row[0],
            class_name=row[1],
            p3d=np.array([row[2], row[3], row[4]], dtype=np.float64),
            descriptor=desc,
            t_first=row[7],
            t_last=row[8],
            n_obs=int(row[9]),
            meta=meta,
        )
        self._memory[lid] = rec
        return rec

    def _iter_all_sql(self) -> List[LandmarkRecord]:
        if not self._sqlite_path:
            return list(self._memory.values())
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT id, class_name, px, py, pz, desc_blob, desc_dim, t_first, t_last, n_obs, meta_blob FROM landmarks"
            ).fetchall()
        finally:
            con.close()
        out: List[LandmarkRecord] = []
        for row in rows:
            meta = pickle.loads(row[10]) if row[10] else {}
            desc = np.frombuffer(row[5], dtype=np.float32, count=int(row[6]))
            out.append(
                LandmarkRecord(
                    id=row[0],
                    class_name=row[1],
                    p3d=np.array([row[2], row[3], row[4]], dtype=np.float64),
                    descriptor=desc,
                    t_first=row[7],
                    t_last=row[8],
                    n_obs=int(row[9]),
                    meta=meta,
                )
            )
        return out

    def all_landmarks(self) -> List[LandmarkRecord]:
        return list(self._iter_all_sql())

    def query_landmark(
        self,
        descriptor: np.ndarray,
        top_k: int = 5,
        radius_m: float = 50.0,
        query_xy: Tuple[float, float] = (0.0, 0.0),
        class_filter: Optional[Sequence[str]] = None,
    ) -> List[Tuple[LandmarkRecord, float]]:
        """
        Returns list of (LandmarkRecord, score) sorted by score descending.
        score = cosine_similarity * spatial_gate (linear ramp inside radius).
        """
        q = _l2_normalize(np.asarray(descriptor, dtype=np.float32))
        allowed = set(class_filter) if class_filter is not None else None
        candidates = self._iter_all_sql()
        qx, qy = float(query_xy[0]), float(query_xy[1])
        scored: List[Tuple[LandmarkRecord, float]] = []
        for rec in candidates:
            if allowed is not None and rec.class_name not in allowed:
                continue
            dx = float(rec.p3d[0]) - qx
            dy = float(rec.p3d[1]) - qy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > radius_m:
                continue
            cos = float(np.clip(np.dot(q, rec.descriptor.astype(np.float64)), -1.0, 1.0))
            spatial = max(0.0, 1.0 - dist / max(radius_m, 1e-6))
            score = cos * (0.25 + 0.75 * spatial)
            scored.append((rec, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]

    def save_pickle(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self._dim,
            "records": {k: (v.class_name, v.p3d, v.descriptor, v.t_first, v.t_last, v.n_obs, v.meta) for k, v in self._memory.items()},
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(path: str | Path, sqlite_path: Optional[str] = None) -> "LandmarkDatabase":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        db = LandmarkDatabase(sqlite_path=sqlite_path)
        db._dim = payload.get("dim")
        for lid, (cn, p3d, desc, t0, t1, n_obs, meta) in payload.get("records", {}).items():
            db._memory[lid] = LandmarkRecord(
                id=lid,
                class_name=cn,
                p3d=np.asarray(p3d, dtype=np.float64),
                descriptor=np.asarray(desc, dtype=np.float32),
                t_first=float(t0),
                t_last=float(t1),
                n_obs=int(n_obs),
                meta=dict(meta or {}),
            )
        return db
