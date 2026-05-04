"""
MediaPipe Hands wrapper — compatible with mediapipe >= 0.10 (Tasks API).

mediapipe 0.10.21+ removed mp.solutions.hands. This module uses the new
HandLandmarker Tasks API instead, downloading the required .task model
file automatically on first run.

Output contract (unchanged from old version):
  process(frame) → list of dicts, one per detected hand:
    fingertip  : (x, y) int pixels   — index finger tip (landmark 8)
    thumb_tip  : (x, y) int pixels   — thumb tip (landmark 4)
    hand_size  : float pixels         — wrist → mid-MCP distance
    landmarks  : list[21] (x,y) tuples
"""
import cv2
import numpy as np
import os
import time as _time_mod
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

# ── Model download ────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_MODEL     = os.path.join(_DIR, 'hand_landmarker.task')
_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

def _ensure_model():
    if os.path.exists(_MODEL):
        return
    print('[AETHER] Downloading hand_landmarker.task model (~25 MB) ...')
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL)
        print('[AETHER] Model saved to', _MODEL)
    except Exception as e:
        raise RuntimeError(
            f'[AETHER] Model download failed: {e}\n'
            f'Download manually from:\n  {_MODEL_URL}\n'
            f'and place it at:\n  {_MODEL}'
        )

# ── Hand skeleton connections for debug drawing ───────────────────────────────
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm
]


# ── HandTracker ───────────────────────────────────────────────────────────────
class HandTracker:
    _FINGERTIP = 8
    _THUMB_TIP = 4
    _WRIST     = 0
    _MID_MCP   = 9

    def __init__(self):
        _ensure_model()

        base_opts = _mp_python.BaseOptions(model_asset_path=_MODEL)
        options   = _mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = _mp_vision.HandLandmarker.create_from_options(options)
        self._t0       = _time_mod.perf_counter()
        self._last_ts  = -1   # guard against duplicate timestamps

    # Process MediaPipe on a downscaled frame — landmarks are normalised 0-1
    # so the output pixel positions are simply multiplied back to full-res.
    _DETECT_SCALE = 0.35

    def process(self, frame):
        h, w = frame.shape[:2]

        # Strictly-increasing millisecond timestamp required by VIDEO mode
        ts_ms = int((_time_mod.perf_counter() - self._t0) * 1000)
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms

        # Downscale for faster inference; landmarks are scale-independent
        small    = cv2.resize(frame,
                              (int(w * self._DETECT_SCALE),
                               int(h * self._DETECT_SCALE)),
                              interpolation=cv2.INTER_AREA)
        rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect_for_video(mp_image, ts_ms)

        out = []
        if not result.hand_landmarks:
            return out

        for lm_list in result.hand_landmarks:
            def to_px(i):
                return (int(lm_list[i].x * w), int(lm_list[i].y * h))

            wrist   = np.array([lm_list[self._WRIST].x   * w,
                                lm_list[self._WRIST].y   * h])
            mid_mcp = np.array([lm_list[self._MID_MCP].x * w,
                                lm_list[self._MID_MCP].y * h])

            out.append({
                'fingertip': to_px(self._FINGERTIP),
                'thumb_tip': to_px(self._THUMB_TIP),
                'hand_size': max(float(np.linalg.norm(mid_mcp - wrist)), 1.0),
                'landmarks': [to_px(i) for i in range(21)],
            })

        return out

    def draw_debug(self, frame, hands_data):
        for hand in hands_data:
            lms = hand['landmarks']
            for a, b in _CONNECTIONS:
                cv2.line(frame, lms[a], lms[b], (0, 200, 80), 1)
            for pt in lms:
                cv2.circle(frame, pt, 3, (0, 255, 140), -1)
            cv2.circle(frame, hand['fingertip'], 6, (0, 220, 255), -1)

    def release(self):
        self._detector.close()
