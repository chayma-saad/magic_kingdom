"""
One Piece motion detector — recognises Luffy-style physical moves from
raw hand velocity and position patterns instead of drawn trail shapes.

Moves detected (single hand unless noted):
  'Gomu Gomu no Pistol'       — fast horizontal punch (quick side-sweep)
  'Gomu Gomu no Cannon'       — fast downward slam (hand drops quickly)
  'Gear Second'               — pumping motion (hand bobs up-down 4+ times)
  'Gear Third: King Kong Gun' — sudden fist-open explosion (closed → all fingers spread)
  'Elephant Gatling'          — rapid alternating spikes from BOTH hands (tracked by caller)

All moves have a per-move cooldown so they can't spam.
"""
import time
import math
import numpy as np
from collections import deque

# ── Thresholds ────────────────────────────────────────────────────────────────

_PUNCH_VEL        = 26    # px/frame for a horizontal punch
_PUNCH_FRAMES     = 6     # look-back window in frames
_PUNCH_HORIZ_BIAS = 1.6   # dx must be this many times dy

_SLAM_VEL         = 24    # px/frame for a downward slam
_SLAM_FRAMES      = 6
_SLAM_DOWN_BIAS   = 1.5   # dy (downward) must exceed dx by this factor

_PUMP_REVERSALS   = 4     # direction changes in Y to count as a pump
_PUMP_MIN_DELTA   = 10    # px minimum to count a direction change

_FIST_OPEN_THRESH = 3     # fingers that must go from ≤1 to ≥3 extended

COOLDOWNS = {
    'Gomu Gomu no Pistol':       2.5,
    'Gomu Gomu no Cannon':       2.5,
    'Gear Second':               4.0,
    'Gear Third: King Kong Gun': 4.0,
    'Elephant Gatling':          3.0,
}


# ── Finger extension helper (reuses same logic as hand_sign_detector) ─────────

def _count_extended(lms):
    """Return number of non-thumb fingers currently extended."""
    wrist = np.array(lms[0], dtype=np.float32)
    count = 0
    for tip_i, pip_i in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip_d = np.linalg.norm(np.array(lms[tip_i], np.float32) - wrist)
        pip_d = np.linalg.norm(np.array(lms[pip_i], np.float32) - wrist)
        if tip_d > pip_d * 1.10:
            count += 1
    return count


# ── Per-hand detector ─────────────────────────────────────────────────────────

class OnePieceDetector:
    """One instance per tracked hand."""

    def __init__(self):
        self._pos   = deque(maxlen=30)   # (x, y) history
        self._cool  = {}                 # move → last trigger time
        self._pump_reversals = 0
        self._last_y_dir     = 0         # +1 / -1 / 0
        self._prev_extended  = 0         # finger count previous frame

    def update(self, hand_data: dict, vel_magnitude: float) -> str | None:
        """
        Call once per frame. Returns move name string when triggered, else None.
        vel_magnitude = pixel/frame speed of fingertip from the main loop.
        """
        pos  = hand_data['fingertip']
        lms  = hand_data['landmarks']
        now  = time.time()
        self._pos.append(pos)

        n_extended = _count_extended(lms)

        def on_cooldown(move):
            return now - self._cool.get(move, 0) < COOLDOWNS.get(move, 3.0)

        def fire(move):
            self._cool[move] = now
            return move

        # ── Gear Third: sudden fist-open ──────────────────────────────────────
        # Transition: ≤ 1 finger → ≥ 3 fingers extended in a single frame
        if (self._prev_extended <= 1 and n_extended >= 3
                and not on_cooldown('Gear Third: King Kong Gun')):
            self._prev_extended = n_extended
            return fire('Gear Third: King Kong Gun')
        self._prev_extended = n_extended

        # ── Gear Second: pumping (Y direction reversals) ──────────────────────
        if len(self._pos) >= 2:
            dy = self._pos[-1][1] - self._pos[-2][1]
            if abs(dy) >= _PUMP_MIN_DELTA:
                new_dir = 1 if dy > 0 else -1
                if self._last_y_dir != 0 and new_dir != self._last_y_dir:
                    self._pump_reversals += 1
                self._last_y_dir = new_dir
            # Fire after enough reversals
            if self._pump_reversals >= _PUMP_REVERSALS and not on_cooldown('Gear Second'):
                self._pump_reversals = 0
                self._last_y_dir     = 0
                return fire('Gear Second')

        # Need enough history for punch/slam checks
        if len(self._pos) < _PUNCH_FRAMES:
            return None

        dx_raw = self._pos[-1][0] - self._pos[-_PUNCH_FRAMES][0]
        dy_raw = self._pos[-1][1] - self._pos[-_PUNCH_FRAMES][1]
        adx, ady = abs(dx_raw), abs(dy_raw)

        # ── Gomu Gomu no Pistol: fast horizontal ─────────────────────────────
        if (vel_magnitude >= _PUNCH_VEL and
                adx > ady * _PUNCH_HORIZ_BIAS and
                not on_cooldown('Gomu Gomu no Pistol')):
            return fire('Gomu Gomu no Pistol')

        # ── Gomu Gomu no Cannon: fast downward slam ───────────────────────────
        if (vel_magnitude >= _SLAM_VEL and
                dy_raw > _SLAM_FRAMES * 4 and           # net downward
                ady > adx * _SLAM_DOWN_BIAS and
                not on_cooldown('Gomu Gomu no Cannon')):
            return fire('Gomu Gomu no Cannon')

        return None

    def reset(self):
        self._pos.clear()
        self._cool.clear()
        self._pump_reversals = 0
        self._last_y_dir     = 0
        self._prev_extended  = 0


# ── Two-hand Gatling tracker ──────────────────────────────────────────────────

class GatlingTracker:
    """
    Tracks rapid alternating velocity spikes across BOTH hands.
    When enough alternate-hand spikes happen in the window → Elephant Gatling.
    """
    _WINDOW   = 2.5      # seconds
    _MIN_HITS = 6        # spikes needed (3 per hand alternating)
    _SPIKE_VEL = 18      # px/frame to count as a hit
    COOLDOWN   = 3.0

    def __init__(self):
        self._hits  = deque()   # (timestamp, hand_idx)
        self._cool  = 0.0

    def register_spike(self, hand_idx: int, vel: float):
        now = time.time()
        if vel >= self._SPIKE_VEL:
            self._hits.append((now, hand_idx))
        # Prune old hits
        while self._hits and now - self._hits[0][0] > self._WINDOW:
            self._hits.popleft()

    def check(self) -> bool:
        """Returns True (and resets) if Gatling should fire."""
        now = time.time()
        if now - self._cool < self.COOLDOWN:
            return False
        if len(self._hits) < self._MIN_HITS:
            return False
        # Require alternating hands
        hands_seen = [h for _, h in self._hits]
        alternating = sum(
            1 for i in range(1, len(hands_seen))
            if hands_seen[i] != hands_seen[i - 1]
        )
        if alternating >= self._MIN_HITS - 2:
            self._hits.clear()
            self._cool = now
            return True
        return False

    def reset(self):
        self._hits.clear()
        self._cool = 0.0
