"""Offline checks for Part B scenarios (no acceptance KPI thresholds in assertions)."""

import math
import tempfile

import numpy as np

from gps_visual_b.gps_integrity import GPSIntegrityMonitor, GpsIntegrityState, GpsQualitySample
from gps_visual_b.lane_bev import LanePosition
from gps_visual_b.lane_bev import LaneBEVDetector
from gps_visual_b.landmark_database import LandmarkDatabase
from gps_visual_b.uturn_detector import UTurnDetector
from gps_visual_b.vpr_encoder import ORBVLADEncoder


def test_s2_uturn_detector_event():
    d = UTurnDetector(angle_deg=150.0, window_sec=2.0)
    fired = False
    t = 0.0
    headings = [0.0, 0.5, 1.0, math.pi - 0.1]
    for h in headings:
        t += 0.4
        if d.update(h):
            fired = True
    assert fired


def test_s3_lane_left_bias():
    det = LaneBEVDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[350:460, 50:280] = 255
    lp = det.infer_lane_position(img)
    assert lp == LanePosition.LEFT


def test_s1_low_light_lane_no_crash():
    det = LaneBEVDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = 5
    det.infer_lane_position(img)


def test_landmark_db_query_order():
    path = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name
    db = LandmarkDatabase(sqlite_path=path)
    z = np.zeros(512, dtype=np.float32)
    z[0] = 1.0
    db.add_landmark("pothole", (10.0, 0.0, 0.0), z)
    z2 = np.zeros(512, dtype=np.float32)
    z2[1] = 1.0
    db.add_landmark("pothole", (100.0, 0.0, 0.0), z2)
    q = np.zeros(512, dtype=np.float32)
    q[0] = 1.0
    hits = db.query_landmark(q, top_k=2, radius_m=50.0, query_xy=(10.0, 0.0))
    assert hits[0][0].p3d[0] == 10.0


def test_gps_integrity_fsm():
    m = GPSIntegrityMonitor(hdop_degraded=5.0, min_satellites=4)
    q_bad = GpsQualitySample(hdop=10.0, n_satellites=3, fix_ok=True, stamp_sec=0.0)
    assert m.update(q_bad, 10.0, 106.0) == GpsIntegrityState.GPS_DEGRADED
    q_lost = GpsQualitySample(hdop=99.0, n_satellites=0, fix_ok=False, stamp_sec=1.0)
    assert m.update(q_lost, 10.0, 106.0) == GpsIntegrityState.GPS_LOST


def test_orb_vlad_encoder_dim():
    enc = ORBVLADEncoder(n_clusters=8)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[80:160, 100:220] = 200
    v = enc.encode_bgr(img)
    assert v.shape == (8 * 32,)
