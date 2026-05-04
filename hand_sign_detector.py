"""
Naruto hand sign detector.

Uses MediaPipe hand landmarks (21 points) to identify static finger
configurations rather than drawn trail shapes.

Signs recognised (single-hand pose):
  'ram'   — index + middle up, ring + pinky curled  → Kage Bunshin
  'bird'  — only index finger up                    → Chidori
  'tiger' — all four fingers spread open            → Katon: Goka Mekkyaku
  'boar'  — closed fist (no fingers extended)       → Rasengan
  'snake' — only pinky extended                     → Kagemane no Jutsu

Each sign must be held for HOLD_SECS before it fires.
A COOLDOWN_SEC lockout prevents accidental re-triggers.
"""
import numpy as np
import time

HOLD_SECS    = 1.2   # seconds the sign must be stable before triggering
COOLDOWN_SEC = 3.5   # seconds before the same sign can fire again

# ── Landmark indices for each finger: (TIP, PIP) ──────────────────────────────
_FINGERS = {
    'index':  (8,  6),
    'middle': (12, 10),
    'ring':   (16, 14),
    'pinky':  (20, 18),
}

# ── Public label / jutsu name tables ──────────────────────────────────────────
SIGN_LABELS = {
    'ram':   'Ram  (Ne)',
    'bird':  'Bird  (Tori)',
    'tiger': 'Tiger (Tora)',
    'boar':  'Boar  (I)',
    'snake': 'Snake (Mi)',
}

SIGN_JUTSU = {
    'ram':   'Kage Bunshin',
    'bird':  'Chidori',
    'tiger': 'Katon: Goka Mekkyaku',
    'boar':  'Rasengan',
    'snake': 'Kagemane no Jutsu',
}

# HUD colour per jutsu (BGR)
SIGN_COLORS = {
    'ram':   (255, 220, 80),    # chakra blue-gold
    'bird':  (255, 180, 30),    # electric blue
    'tiger': (0,   80, 255),    # fire orange
    'boar':  (255, 200, 100),   # chakra blue
    'snake': (180, 0,  100),    # shadow purple
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finger_states(lms):
    """
    Returns dict finger_name → bool (extended?).
    Uses wrist-distance method so the result is orientation-independent.
    """
    wrist = np.array(lms[0], dtype=np.float32)
    states = {}
    for name, (tip_i, pip_i) in _FINGERS.items():
        tip_d = np.linalg.norm(np.array(lms[tip_i], np.float32) - wrist)
        pip_d = np.linalg.norm(np.array(lms[pip_i], np.float32) - wrist)
        states[name] = tip_d > pip_d * 1.10     # 10 % margin
    return states


def _classify(st):
    idx, mid, ring, pink = (st['index'], st['middle'],
                            st['ring'],  st['pinky'])
    if idx and mid and not ring and not pink:
        return 'ram'
    if idx and not mid and not ring and not pink:
        return 'bird'
    if idx and mid and ring and pink:
        return 'tiger'
    if not idx and not mid and not ring and not pink:
        return 'boar'
    if not idx and not mid and not ring and pink:
        return 'snake'
    return None


# ── Detector class ────────────────────────────────────────────────────────────

class HandSignDetector:
    """One instance per tracked hand."""

    def __init__(self):
        self._sign  = None      # current held sign
        self._since = None      # time the current sign was first detected
        self._cool  = {}        # sign → timestamp of last trigger

    def update(self, hand_data: dict) -> str | None:
        """
        Feed one hand dict from HandTracker.process().
        Returns the triggered jutsu name string on the frame it fires, else None.
        """
        sign = _classify(_finger_states(hand_data['landmarks']))
        now  = time.time()

        if sign != self._sign:
            self._sign  = sign
            self._since = now if sign else None
            return None

        if not sign or self._since is None:
            return None

        # Still in cooldown?
        if now - self._cool.get(sign, 0) < COOLDOWN_SEC:
            return None

        # Hold long enough?
        if now - self._since >= HOLD_SECS:
            self._cool[sign] = now
            self._since      = now      # reset so it doesn't retrigger next frame
            return SIGN_JUTSU[sign]

        return None

    # ── Properties for HUD rendering ─────────────────────────────────────────

    @property
    def hold_progress(self) -> float:
        """0.0–1.0 fill progress for the current sign being held."""
        if not self._sign or self._since is None:
            return 0.0
        return min(1.0, (time.time() - self._since) / HOLD_SECS)

    @property
    def current_sign(self) -> str | None:
        return self._sign

    @property
    def current_label(self) -> str:
        return SIGN_LABELS.get(self._sign, '')

    @property
    def current_jutsu(self) -> str:
        return SIGN_JUTSU.get(self._sign, '')

    @property
    def current_color(self):
        return SIGN_COLORS.get(self._sign, (200, 200, 200))

    def reset(self):
        self._sign  = None
        self._since = None
        self._cool.clear()
