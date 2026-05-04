"""
MediaPipe Pose wrapper — full-body skeleton tracking (33 landmarks).
Used by the One Piece theme so moves are detected from the whole body,
not just finger trails.

Auto-downloads pose_landmarker_lite.task on first run (~9 MB).

Output from process():
  None if no person detected, else a dict:
    'landmarks'   : list of 33 (x, y) pixel tuples
    'r_shoulder'  : (x, y)
    'l_shoulder'  : (x, y)
    'r_elbow'     : (x, y)
    'l_elbow'     : (x, y)
    'r_wrist'     : (x, y)
    'l_wrist'     : (x, y)
    'r_hip'       : (x, y)
    'l_hip'       : (x, y)
    'r_knee'      : (x, y)
    'l_knee'      : (x, y)
    'visibility'  : list of 33 visibility floats (0–1)
"""
import cv2
import numpy as np
import math
import os
import time as _time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

# ── Model auto-download ───────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_MODEL     = os.path.join(_DIR, 'pose_landmarker.task')
_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
)

def _ensure_model():
    if os.path.exists(_MODEL):
        return
    print('[AETHER] Downloading pose_landmarker.task (~9 MB) ...')
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL)
        print('[AETHER] Pose model saved to', _MODEL)
    except Exception as e:
        raise RuntimeError(
            f'[AETHER] Pose model download failed: {e}\n'
            f'Download manually from:\n  {_MODEL_URL}\n'
            f'and save it as:\n  {_MODEL}'
        )

# ── Landmark indices ──────────────────────────────────────────────────────────
R_SHOULDER = 12;  L_SHOULDER = 11
R_ELBOW    = 14;  L_ELBOW    = 13
R_WRIST    = 16;  L_WRIST    = 15
R_HIP      = 24;  L_HIP      = 23
R_KNEE     = 26;  L_KNEE     = 25
R_ANKLE    = 28;  L_ANKLE    = 27
NOSE       =  0

# ── Skeleton connections to draw ──────────────────────────────────────────────
_CONNECTIONS = [
    # Torso box
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP,      R_HIP),
    # Left arm
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW,    L_WRIST),
    # Right arm
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW,    R_WRIST),
    # Left leg
    (L_HIP,  L_KNEE),
    (L_KNEE, L_ANKLE),
    # Right leg
    (R_HIP,  R_KNEE),
    (R_KNEE, R_ANKLE),
    # Head spine
    (NOSE, L_SHOULDER),
    (NOSE, R_SHOULDER),
]

# Joint dots to render (skip feet/toes)
_JOINT_DOTS = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
               L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE]


# ── PoseTracker class ─────────────────────────────────────────────────────────

_SMOOTH_KEYS = ('r_shoulder', 'l_shoulder', 'r_elbow', 'l_elbow',
                'r_wrist', 'l_wrist', 'r_hip', 'l_hip', 'r_knee', 'l_knee')

class PoseSmoother:
    """
    Exponential moving average on key joint positions.
    alpha=1.0 → no smoothing (raw), alpha=0.4 → very smooth but laggy.
    """
    def __init__(self, alpha=0.55):
        self._alpha = alpha
        self._prev  = None

    def smooth(self, pose_data):
        if pose_data is None:
            self._prev = None
            return None
        if self._prev is None:
            self._prev = {k: pose_data[k] for k in _SMOOTH_KEYS}
            return pose_data

        result = dict(pose_data)
        a, b = self._alpha, 1.0 - self._alpha
        for k in _SMOOTH_KEYS:
            cx = int(a * pose_data[k][0] + b * self._prev[k][0])
            cy = int(a * pose_data[k][1] + b * self._prev[k][1])
            result[k]       = (cx, cy)
            self._prev[k]   = (cx, cy)
        return result

    def reset(self):
        self._prev = None


class PoseTracker:
    _DETECT_SCALE = 0.35     # downscale for faster inference

    def __init__(self):
        _ensure_model()
        opts = _mp_vision.PoseLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_MODEL),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._det    = _mp_vision.PoseLandmarker.create_from_options(opts)
        self._t0     = _time.perf_counter()
        self._last_ts = -1

    def process(self, frame):
        """
        Returns a pose dict (see module docstring) or None.
        frame must be a BGR uint8 numpy array.
        """
        h, w = frame.shape[:2]

        ts_ms = int((_time.perf_counter() - self._t0) * 1000)
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms

        small  = cv2.resize(frame,
                            (int(w * self._DETECT_SCALE),
                             int(h * self._DETECT_SCALE)),
                            interpolation=cv2.INTER_AREA)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._det.detect_for_video(mp_img, ts_ms)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]

        def px(i):
            return (int(lms[i].x * w), int(lms[i].y * h))

        vis = [lms[i].visibility for i in range(33)]

        return {
            'landmarks':  [px(i) for i in range(33)],
            'visibility': vis,
            'r_shoulder': px(R_SHOULDER),  'l_shoulder': px(L_SHOULDER),
            'r_elbow':    px(R_ELBOW),     'l_elbow':    px(L_ELBOW),
            'r_wrist':    px(R_WRIST),     'l_wrist':    px(L_WRIST),
            'r_hip':      px(R_HIP),       'l_hip':      px(L_HIP),
            'r_knee':     px(R_KNEE),      'l_knee':     px(L_KNEE),
        }

    # ── Skeleton ──────────────────────────────────────────────────────────────

    def draw_skeleton(self, frame, pose_data, pulse=1.0):
        """
        Subtle wrist-only glow dots — no lines drawn over the body.
        Just a soft halo at each wrist so the rubber anchor is visible.
        """
        if pose_data is None:
            return
        vis = pose_data['visibility']
        ov  = np.zeros_like(frame)

        for idx, pt in [(R_WRIST, pose_data['r_wrist']),
                        (L_WRIST, pose_data['l_wrist'])]:
            if vis[idx] > 0.35:
                r = int(10 + 4 * pulse)
                cv2.circle(ov, pt, r, (30, 80, 220), -1)   # dim orange dot

        bl  = cv2.GaussianBlur(ov, (21, 21), 0)
        out = frame.astype(np.float32) + bl.astype(np.float32) * 0.6
        np.clip(out, 0, 255, out=out)
        frame[:] = out.astype(np.uint8)

    # ── Rubber arm stretch visual ─────────────────────────────────────────────

    def draw_rubber_stretch(self, frame, pose_data, side: str, ratio: float):
        """
        Realistic rubber arm — copies REAL ARM PIXELS from the camera and
        pastes them beyond the wrist so the arm literally looks stretched.
        No fake drawn geometry — uses your actual arm from the video feed.
        """
        if ratio < 0.05:
            return

        if side == 'right':
            wrist  = pose_data['r_wrist']
            elbow  = pose_data['r_elbow']
            vis_w  = pose_data['visibility'][R_WRIST]
            vis_e  = pose_data['visibility'][R_ELBOW]
        else:
            wrist  = pose_data['l_wrist']
            elbow  = pose_data['l_elbow']
            vis_w  = pose_data['visibility'][L_WRIST]
            vis_e  = pose_data['visibility'][L_ELBOW]

        if vis_w < 0.30 or vis_e < 0.30:
            return

        H, W = frame.shape[:2]

        # Arm direction: elbow → wrist
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        seg = max(1.0, math.sqrt(dx * dx + dy * dy))
        nx, ny    = dx / seg, dy / seg      # unit along arm
        perp_x, perp_y = -ny, nx           # unit perpendicular

        stretch_px = int(min(ratio, 1.5) * 260)
        if stretch_px < 4:
            return

        # Wider cross-section → more visible stretched arm
        arm_half = min(32, max(16, int(seg * 0.32)))
        offsets  = np.arange(-arm_half, arm_half + 1, dtype=np.float32)

        # ── Sample the wrist cross-section (pull back 8px for clean pixels) ──
        pull_back = 8
        src_cx = wrist[0] - nx * pull_back
        src_cy = wrist[1] - ny * pull_back

        sx = np.clip((src_cx + perp_x * offsets).astype(np.int32), 0, W - 1)
        sy = np.clip((src_cy + perp_y * offsets).astype(np.int32), 0, H - 1)
        arm_strip = frame[sy, sx].astype(np.float32)

        # Slightly boost brightness so the stretched part stands out
        arm_strip = np.clip(arm_strip * 1.25, 0, 255)

        # ── Paste strip along stretch — slower fade so arm stays solid ───────
        step = 2
        for s in range(step, stretch_px + 1, step):
            fade = max(0.0, 1.0 - (s / stretch_px) * 0.35)   # stays opaque longer

            cx = wrist[0] + nx * s
            cy = wrist[1] + ny * s

            dx2 = np.clip((cx + perp_x * offsets).astype(np.int32), 0, W - 1)
            dy2 = np.clip((cy + perp_y * offsets).astype(np.int32), 0, H - 1)

            bg = frame[dy2, dx2].astype(np.float32)
            frame[dy2, dx2] = np.clip(
                arm_strip * fade + bg * (1.0 - fade), 0, 255
            ).astype(np.uint8)

        # ── Edge outline — thin bright rim makes the arm silhouette pop ──────
        ov = np.zeros_like(frame)
        for edge_off in [arm_half, -arm_half]:
            for s in range(0, stretch_px + 1, 4):
                ex = int(wrist[0] + nx * s + perp_x * edge_off)
                ey = int(wrist[1] + ny * s + perp_y * edge_off)
                if 0 <= ex < W and 0 <= ey < H:
                    ov[ey, ex] = (60, 120, 255)   # warm orange edge

        bl  = cv2.GaussianBlur(ov, (9, 9), 0)
        out = frame.astype(np.float32) + bl.astype(np.float32) * 0.8
        np.clip(out, 0, 255, out=out)
        frame[:] = out.astype(np.uint8)

        # ── Glowing tip ────────────────────────────────────────────────────
        tip = (int(wrist[0] + nx * stretch_px),
               int(wrist[1] + ny * stretch_px))
        ov2 = np.zeros_like(frame)
        cv2.circle(ov2, tip, 16, (60, 120, 255), -1)
        cv2.circle(ov2, tip,  7, (200, 230, 255), -1)
        bl2  = cv2.GaussianBlur(ov2, (23, 23), 0)
        out2 = frame.astype(np.float32) + bl2.astype(np.float32) * 1.0
        np.clip(out2, 0, 255, out=out2)
        frame[:] = out2.astype(np.uint8)

    # ── Sword / weapon ────────────────────────────────────────────────────────

    def draw_sword(self, frame, pose_data, active=False):
        """
        Draw a glowing sword attached to the right wrist.
        The blade points along the elbow → wrist direction.
        active=True makes it brighter (during a move).
        """
        if pose_data is None:
            return
        if pose_data['visibility'][R_WRIST] < 0.4:
            return

        wrist = pose_data['r_wrist']
        elbow = pose_data['r_elbow']

        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        length = max(1.0, math.sqrt(dx * dx + dy * dy))
        nx = dx / length      # unit direction along arm
        ny = dy / length

        BLADE  = 180          # blade length in pixels
        GUARD  = 22           # crossguard half-length

        tip = (int(wrist[0] + nx * BLADE), int(wrist[1] + ny * BLADE))
        # Perpendicular for crossguard
        gx, gy = -ny, nx
        guard1 = (int(wrist[0] + gx * GUARD), int(wrist[1] + gy * GUARD))
        guard2 = (int(wrist[0] - gx * GUARD), int(wrist[1] - gy * GUARD))

        ov = np.zeros_like(frame)

        if active:
            blade_col  = (80, 255, 255)    # bright cyan-white when firing
            guard_col  = (0,  200, 255)
            glow_str   = 1.2
        else:
            blade_col  = (180, 240, 200)   # pale green-white at rest
            guard_col  = (60,  180, 100)
            glow_str   = 0.55

        # Wide blade glow layer
        cv2.line(ov, wrist, tip, blade_col, 6)
        # Sharp core
        cv2.line(ov, wrist, tip, (255, 255, 255), 2)
        # Crossguard
        cv2.line(ov, guard1, guard2, guard_col, 4)
        # Pommel dot at wrist
        cv2.circle(ov, wrist, 5, guard_col, -1)

        bl  = cv2.GaussianBlur(ov, (15, 15), 0)
        out = frame.astype(np.float32) + ov.astype(np.float32) + bl.astype(np.float32) * glow_str
        np.clip(out, 0, 255, out=out)
        frame[:] = out.astype(np.uint8)

    def release(self):
        self._det.close()
