"""Lightweight geometric shape classifier running on the fingertip trail."""
import cv2
import numpy as np
import time as _time_mod
import config


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_corners(pts):
    """
    Return indices of sharp-turn points in pts.
    Uses a sliding window to measure the angle between incoming and outgoing
    direction vectors.
    """
    n   = len(pts)
    win = config.CORNER_WINDOW
    if n < 3 * win:
        return []

    arr  = np.asarray(pts, dtype=np.float64)
    curv = np.zeros(n)

    for i in range(win, n - win):
        v1 = arr[i]     - arr[i - win]
        v2 = arr[i + win] - arr[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 > 0 and l2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
            curv[i] = np.arccos(cos_a)

    corners = []
    for i in range(win, n - win):
        if curv[i] < config.CORNER_MIN_ANG:
            continue
        # local maximum check inside ±CORNER_MIN_GAP/2
        lo = max(win, i - config.CORNER_MIN_GAP // 2)
        hi = min(n - win, i + config.CORNER_MIN_GAP // 2 + 1)
        if curv[i] < curv[lo:hi].max():
            continue
        # ensure minimum gap from previous corner
        if corners and (i - corners[-1]) < config.CORNER_MIN_GAP:
            if curv[i] > curv[corners[-1]]:
                corners[-1] = i   # replace with stronger corner
            continue
        corners.append(i)

    return corners


def _segment_lengths(pts, corner_indices):
    """Return Euclidean distances between consecutive corner positions."""
    arr    = np.asarray(pts, dtype=np.float64)
    corner_pts = arr[corner_indices]
    lengths = []
    for i in range(len(corner_pts) - 1):
        lengths.append(float(np.linalg.norm(corner_pts[i + 1] - corner_pts[i])))
    return lengths


# ── Public shape detectors ────────────────────────────────────────────────────

def detect_circle(pts):
    if len(pts) < config.CIRCLE_MIN_PTS:
        return False
    arr      = np.asarray(pts, dtype=np.float64)
    centroid = arr.mean(axis=0)
    dists    = np.linalg.norm(arr - centroid, axis=1)
    avg      = dists.mean()
    if avg < config.CIRCLE_MIN_RADIUS:
        return False
    if dists.std() / avg > config.CIRCLE_STD_RATIO:
        return False
    # closure — start and end must be near each other
    if np.linalg.norm(arr[0] - arr[-1]) > config.CIRCLE_CLOSE_PX:
        return False
    return True


def detect_spiral(pts):
    if len(pts) < config.SPIRAL_MIN_PTS:
        return False
    arr      = np.asarray(pts, dtype=np.float64)
    centroid = arr.mean(axis=0)
    delta    = arr - centroid
    angles   = np.arctan2(delta[:, 1], delta[:, 0])
    radii    = np.linalg.norm(delta, axis=1)

    angles_uw = np.unwrap(angles)
    total_rot = abs(angles_uw[-1] - angles_uw[0])

    if total_rot < config.SPIRAL_MIN_REVS * 2 * np.pi:
        return False

    # Direction consistency
    diffs    = np.diff(angles_uw)
    pos_frac = (diffs > 0).mean()
    if max(pos_frac, 1 - pos_frac) < 0.72:
        return False

    # Radius must vary (otherwise it is just a circle)
    r_mean = radii.mean()
    if r_mean > 0 and radii.std() / r_mean < 0.10:
        return False

    return True


def detect_line(pts):
    if len(pts) < config.LINE_MIN_PTS:
        return False
    arr      = np.asarray(pts, dtype=np.float64)
    centered = arr - arr.mean(axis=0)
    _, s, _  = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-6:
        return False
    # For a straight line the second singular value is tiny
    if s[1] / s[0] > config.LINE_PERP_RATIO:
        return False
    return True


def detect_triangle(pts):
    corners = _find_corners(pts)
    if len(corners) != 3:
        return False
    lengths = _segment_lengths(pts, corners)
    if len(lengths) < 2:
        return False
    if min(lengths) < config.CORNER_SEG_MIN:
        return False
    mean_l = np.mean(lengths)
    if mean_l > 0 and np.std(lengths) / mean_l > 0.55:
        return False
    return True


def detect_square(pts):
    corners = _find_corners(pts)
    if len(corners) != 4:
        return False
    lengths = _segment_lengths(pts, corners)
    if len(lengths) < 3:
        return False
    if min(lengths) < config.CORNER_SEG_MIN:
        return False
    # Angles at corners should be near 90°
    arr = np.asarray(pts, dtype=np.float64)
    cpts = arr[corners]
    for i in range(4):
        p0 = cpts[i]
        p1 = cpts[(i + 1) % 4]
        p2 = cpts[(i + 2) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1 or l2 < 1:
            return False
        cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
        angle = np.arccos(cos_a)
        if abs(angle - np.pi / 2) > np.radians(32):
            return False
    return True


# ── Intent guard ─────────────────────────────────────────────────────────────

def _sufficient_movement(pts):
    """
    Returns True only when the trail shows deliberate drawing motion.

    Checks two independent conditions:
      1. Total path length > MIN_GESTURE_DISTANCE  (filters micro-tremors)
      2. Bounding-box max dimension > MIN_GESTURE_BBOX  (filters in-place wobble)

    Both must pass so that a shape can be detected.
    """
    if len(pts) < 10:
        return False
    arr = np.asarray(pts, dtype=np.float64)

    # Total path length
    total_dist = np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()
    if total_dist < config.MIN_GESTURE_DISTANCE:
        return False

    # Bounding box
    bbox_w = arr[:, 0].max() - arr[:, 0].min()
    bbox_h = arr[:, 1].max() - arr[:, 1].min()
    if max(bbox_w, bbox_h) < config.MIN_GESTURE_BBOX:
        return False

    return True


# ── Main classifier ───────────────────────────────────────────────────────────

class ShapeDetector:
    """
    Runs all shape detectors on the current trail in priority order.
    Includes a cooldown so the same shape is not reported every frame.
    """
    _PRIORITY = ['circle', 'spiral', 'triangle', 'square', 'line']

    def __init__(self):
        self._last_shape = None
        self._last_time  = 0.0
        self._circle_dual_armed = False
        self._circle_dual_gap    = False
        self._circle_dual_time  = 0.0

    def detect_circle_dual(self, pts):
        """Return 'circle_dual' when two circle closures occur within 1.5s."""
        now = _time_mod.time()

        if not detect_circle(pts):
            if self._circle_dual_armed:
                self._circle_dual_gap = True
            if self._circle_dual_armed and (now - self._circle_dual_time) > 1.5:
                self._circle_dual_armed = False
                self._circle_dual_gap    = False
            return None

        if (self._circle_dual_armed and self._circle_dual_gap and
                (now - self._circle_dual_time) <= 1.5):
            self._circle_dual_armed = False
            self._circle_dual_gap    = False
            self._circle_dual_time  = 0.0
            self._last_shape = 'circle_dual'
            self._last_time  = now
            return 'circle_dual'

        self._circle_dual_armed = True
        self._circle_dual_gap    = False
        self._circle_dual_time   = now
        self._last_shape = 'circle'
        self._last_time  = now
        return 'circle'

    def detect(self, pts):
        """Return detected shape name or None."""
        now = _time_mod.time()

        # ── Intent guard: finger must have actually moved ─────────────────────
        if not _sufficient_movement(pts):
            return None

        circle_shape = self.detect_circle_dual(pts)
        if circle_shape:
            shape = circle_shape
        else:
            if now - self._last_time < config.SHAPE_COOLDOWN:
                return None

            if detect_spiral(pts):
                shape = 'spiral'
            elif detect_triangle(pts):
                shape = 'triangle'
            elif detect_square(pts):
                shape = 'square'
            elif detect_line(pts):
                shape = 'line'
            else:
                return None

        self._last_shape = shape
        self._last_time  = now
        return shape

    def reset(self):
        self._last_shape = None
        self._last_time  = 0.0
        self._circle_dual_armed = False
        self._circle_dual_gap    = False
        self._circle_dual_time  = 0.0


# ── Debug overlay ─────────────────────────────────────────────────────────────

def draw_debug_corners(overlay, pts):
    """Draw curvature corners (debug mode)."""
    corners = _find_corners(pts)
    arr     = np.asarray(pts, dtype=np.int32)
    for i in corners:
        cv2.circle(overlay, tuple(arr[i]), 8, (0, 0, 255), 2)
