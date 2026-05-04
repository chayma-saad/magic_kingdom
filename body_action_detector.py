"""
Body action detector for One Piece theme.

Uses POSITION-based state machines, not velocity — completely immune to
jitter, sway, and breathing.

How each move works
-------------------
Gomu Gomu no Pistol
    Extend ONE arm far to the side (wrist past shoulder by 0.7 × shoulder_width).
    The arm returns to rest before you can fire again.

Gomu Gomu no Cannon
    Push ONE arm straight DOWN (wrist below shoulder by 1.3 × shoulder_width).

Gomu Gomu no Bazooka
    Spread BOTH arms wide at the same time (wrists > 2.5 × shoulder_width apart).

Gear Second
    Bounce your whole body visibly up and down 5 times (big hip oscillation).

Gear Third: King Kong Gun
    Hold ONE arm fully extended sideways (> 1.2 × shoulder_width) for 2 seconds.

Elephant Gatling
    NOT here — use Bazooka + Pistol combo instead (Gatling is unreliable in 2D).

All measurements are in multiples of shoulder_width so they scale with
how close or far you stand from the camera.
"""
import time
import math
import numpy as np
from collections import deque

# ── Thresholds (shoulder_width units unless noted) ────────────────────────────

PISTOL_EXTEND   = 1.00   # wrist past shoulder horizontally (raised: need big punch)
PISTOL_RETURN   = 0.40   # must return to this before re-firing

# Cannon: wrist must drop BELOW the knee — arms hanging at sides never reach here
CANNON_BELOW_KNEE = 30   # wrist must be this many pixels BELOW the knee

BAZOOKA_SPREAD  = 2.50   # wrist-to-wrist distance (both arms wide)

GEAR3_EXTEND    = 1.20   # arm held fully out horizontally
GEAR3_HOLD_SECS = 2.0

PUMP_MIN_DY_FR  = 0.08   # hip travel per reversal as fraction of frame_h
PUMP_REVERSALS  = 5

COOLDOWNS = {
    'Gomu Gomu no Pistol':       2.0,
    'Gomu Gomu no Cannon':       2.0,
    'Gomu Gomu no Bazooka':      3.0,
    'Gear Second':               5.0,
    'Gear Third: King Kong Gun': 5.0,
}


# ── Per-arm state machine ─────────────────────────────────────────────────────

class _ArmState:
    """Fires exactly ONCE when ratio crosses extend threshold, then waits
    until ratio drops below return threshold before it can fire again."""

    def __init__(self):
        self.extended = False   # arm is currently past the threshold

    def tick(self, ratio, extend_th, return_th):
        """Returns True on the frame the arm crosses extend_th."""
        if not self.extended and ratio >= extend_th:
            self.extended = True
            return True            # fire!
        if self.extended and ratio < return_th:
            self.extended = False  # ready for next punch
        return False

    def reset(self):
        self.extended = False


# ── Main detector ─────────────────────────────────────────────────────────────

class BodyActionDetector:

    def __init__(self):
        self._r_horiz = _ArmState()   # right arm horizontal (Pistol)
        self._l_horiz = _ArmState()   # left arm horizontal
        self._r_down  = _ArmState()   # right arm downward (Cannon)
        self._l_down  = _ArmState()   # left arm downward

        self._hip_y        = deque(maxlen=50)
        self._pump_rev     = 0
        self._last_hip_dir = 0

        self._gear3_side   = None
        self._gear3_since  = None

        self._cool = {}

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sw(pose_data):
        """Shoulder width in pixels (our scale reference)."""
        rs = pose_data['r_shoulder']
        ls = pose_data['l_shoulder']
        return max(1, abs(rs[0] - ls[0]))

    def r_below_knee(self, pose_data):
        return pose_data['r_wrist'][1] > pose_data['r_knee'][1] + CANNON_BELOW_KNEE

    def l_below_knee(self, pose_data):
        return pose_data['l_wrist'][1] > pose_data['l_knee'][1] + CANNON_BELOW_KNEE

    @staticmethod
    def _arm_dist(wrist, shoulder):
        dx = wrist[0] - shoulder[0]
        dy = wrist[1] - shoulder[1]
        return math.sqrt(dx * dx + dy * dy)

    def arm_extensions(self, pose_data):
        """
        Returns (r_ratio, l_ratio) — direction-agnostic, mirror-safe.
        Measures how far each wrist is from its shoulder relative to
        shoulder_width. 0 = arm at normal rest, 1 = fully extended out.
        Works for any arm direction (sideways, up, diagonal).
        """
        sw = self._sw(pose_data)
        r_dist = self._arm_dist(pose_data['r_wrist'], pose_data['r_shoulder'])
        l_dist = self._arm_dist(pose_data['l_wrist'], pose_data['l_shoulder'])
        # Subtract baseline (~1.0 sw = normal hanging arm) so 0 = at rest
        r_ratio = max(0.0, r_dist / sw - 0.9)
        l_ratio = max(0.0, l_dist / sw - 0.9)
        return r_ratio, l_ratio

    def r_horiz_ratio(self, pose_data):
        r, _ = self.arm_extensions(pose_data)
        return r

    def l_horiz_ratio(self, pose_data):
        _, l = self.arm_extensions(pose_data)
        return l

    # ── Update (call every frame) ─────────────────────────────────────────────

    def update(self, pose_data, frame_w, frame_h):
        """
        Returns (move_name, effect_xy) when a move fires, else (None, None).
        """
        rw  = pose_data['r_wrist']
        lw  = pose_data['l_wrist']
        rs  = pose_data['r_shoulder']
        ls  = pose_data['l_shoulder']
        rh  = pose_data['r_hip']
        lh  = pose_data['l_hip']
        now = time.time()
        sw  = self._sw(pose_data)

        def on_cd(move):
            return now - self._cool.get(move, 0) < COOLDOWNS.get(move, 3.0)

        def fire(move, xy):
            self._cool[move] = now
            return move, xy

        r_h = self.r_horiz_ratio(pose_data)
        l_h = self.l_horiz_ratio(pose_data)

        # ── Gear Third — arm held way out ─────────────────────────────────────
        for side, ratio, wrist in [('right', r_h, rw), ('left', l_h, lw)]:
            if ratio >= GEAR3_EXTEND:
                if self._gear3_side == side and self._gear3_since is not None:
                    if (now - self._gear3_since >= GEAR3_HOLD_SECS
                            and not on_cd('Gear Third: King Kong Gun')):
                        self._gear3_side  = None
                        self._gear3_since = None
                        return fire('Gear Third: King Kong Gun', wrist)
                elif self._gear3_side != side:
                    self._gear3_side  = side
                    self._gear3_since = now
            else:
                if self._gear3_side == side:
                    self._gear3_side  = None
                    self._gear3_since = None

        # ── Gear Second — body bounce ─────────────────────────────────────────
        hip_y = (rh[1] + lh[1]) // 2
        self._hip_y.append(hip_y)
        if len(self._hip_y) >= 2:
            dy       = self._hip_y[-1] - self._hip_y[-2]
            min_dy   = frame_h * PUMP_MIN_DY_FR
            if abs(dy) >= min_dy:
                nd = 1 if dy > 0 else -1
                if self._last_hip_dir != 0 and nd != self._last_hip_dir:
                    self._pump_rev += 1
                self._last_hip_dir = nd
            if self._pump_rev >= PUMP_REVERSALS and not on_cd('Gear Second'):
                self._pump_rev     = 0
                self._last_hip_dir = 0
                cx = (rs[0] + ls[0]) // 2
                return fire('Gear Second', (cx, hip_y))

        # ── Bazooka — both arms spread wide ──────────────────────────────────
        wrist_gap = abs(rw[0] - lw[0]) / sw
        if wrist_gap >= BAZOOKA_SPREAD and not on_cd('Gomu Gomu no Bazooka'):
            cx = (rw[0] + lw[0]) // 2
            cy = (rw[1] + lw[1]) // 2
            return fire('Gomu Gomu no Bazooka', (cx, cy))

        # ── Pistol — arm punches out horizontally (state machine, no spam) ───
        if not on_cd('Gomu Gomu no Pistol'):
            if self._r_horiz.tick(r_h, PISTOL_EXTEND, PISTOL_RETURN):
                return fire('Gomu Gomu no Pistol', rw)
            if self._l_horiz.tick(l_h, PISTOL_EXTEND, PISTOL_RETURN):
                return fire('Gomu Gomu no Pistol', lw)

        # ── Cannon — wrist drops BELOW the knee (truly slammed down) ─────────
        if not on_cd('Gomu Gomu no Cannon'):
            if self._r_down.tick(1.0 if self.r_below_knee(pose_data) else 0.0,
                                 0.9, 0.1):
                return fire('Gomu Gomu no Cannon', rw)
            if self._l_down.tick(1.0 if self.l_below_knee(pose_data) else 0.0,
                                 0.9, 0.1):
                return fire('Gomu Gomu no Cannon', lw)

        return None, None

    def reset(self):
        self._r_horiz.reset(); self._l_horiz.reset()
        self._r_down.reset();  self._l_down.reset()
        self._hip_y.clear()
        self._pump_rev = 0;  self._last_hip_dir = 0
        self._gear3_side = None;  self._gear3_since = None
        self._cool.clear()
