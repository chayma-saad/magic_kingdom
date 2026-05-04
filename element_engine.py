"""
Element Engine — infers the active magical element from recent motion
properties of the fingertip trail.

Rules (from highest to lowest priority):
  VOID  — chaotic: high velocity variance AND high curvature variance
  FIRE  — sharp:   acceleration spike AND high mean curvature
  LIGHT — precise: low curvature variance AND slow deliberate motion
  WATER — fluid:   low velocity variance, moderate speed, curved path
  NEUTRAL — default when nothing matches strongly
"""
import numpy as np
from collections import deque
import config


_HYSTERESIS = 5    # frames a new element must dominate before switching


class ElementEngine:
    def __init__(self):
        self.current       = 'NEUTRAL'
        self._vel_hist     = deque(maxlen=config.ELEM_WINDOW)
        self._candidate    = 'NEUTRAL'
        self._cand_streak  = 0         # consecutive frames candidate held

    # ── Public API ────────────────────────────────────────────────────────────
    def update(self, pts):
        """
        Accepts the current trail point list (list of (x,y) tuples).
        Returns the inferred element string.
        """
        if len(pts) < 4:
            return self.current

        arr    = np.asarray(pts[-config.ELEM_WINDOW:], dtype=np.float64)
        n      = len(arr)

        # Per-frame velocities
        deltas = arr[1:] - arr[:-1]
        vels   = np.linalg.norm(deltas, axis=1)
        if len(vels) == 0:
            return self.current

        avg_vel = vels.mean()
        vel_std = vels.std()
        max_acc = abs(np.diff(vels)).max() if len(vels) > 1 else 0.0

        # Per-frame curvatures (angle change per sample)
        curv_vals = []
        win = 2
        for i in range(win, n - win):
            v1 = arr[i]     - arr[i - win]
            v2 = arr[i + win] - arr[i]
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 > 0 and l2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
                curv_vals.append(np.arccos(cos_a))

        if curv_vals:
            curv_arr  = np.array(curv_vals)
            avg_curv  = curv_arr.mean()
            curv_std  = curv_arr.std()
        else:
            avg_curv = curv_std = 0.0

        # ── Classification ────────────────────────────────────────────────────
        if vel_std > config.VEL_STD_HIGH and curv_std > config.CURV_HIGH:
            element = 'VOID'
        elif max_acc > config.ACCEL_SPIKE and avg_curv > config.CURV_HIGH:
            element = 'FIRE'
        elif curv_std < config.CURV_LOW and avg_vel < config.VEL_SMOOTH:
            element = 'LIGHT'
        elif vel_std < config.VEL_STD_LOW and avg_curv > config.CURV_LOW:
            element = 'WATER'
        else:
            element = 'NEUTRAL'

        # ── Hysteresis gate ───────────────────────────────────────────────────
        if element == self._candidate:
            self._cand_streak += 1
        else:
            self._candidate   = element
            self._cand_streak = 1

        if self._cand_streak >= _HYSTERESIS:
            self.current = element

        return self.current

    def reset(self):
        self.current      = 'NEUTRAL'
        self._candidate   = 'NEUTRAL'
        self._cand_streak = 0
        self._vel_hist.clear()
