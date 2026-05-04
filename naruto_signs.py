"""
Real Naruto hand sign detector — uses both hands simultaneously.

Shadow Clone (Kage Bunshin no Jutsu)
-------------------------------------
From the reference image: one hand vertical (index+middle pointing UP),
one hand horizontal (index+middle pointing SIDEWAYS), fingers crossed
to form a CROSS / plus shape. Both hands close together, almost touching.

Detection steps:
  1. Both hands must be detected at the same time.
  2. Each hand must have index + middle extended, ring + pinky curled.
  3. One hand's index direction is mostly VERTICAL, the other mostly HORIZONTAL.
  4. The two hands must be close (wrist-to-wrist distance < 300 px).
  5. The angle between the two index-finger vectors must be 55°–125° (cross = 90°).
  6. Must hold this for HOLD_SECS before it triggers.
"""
import numpy as np
import time

HOLD_SECS = 1.5    # seconds to hold the sign
COOLDOWN  = 4.0    # seconds before the same sign can fire again


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_extended(lms, tip_i, pip_i):
    """True if the tip is further from the wrist than the PIP joint."""
    wrist = np.array(lms[0], dtype=np.float32)
    tip   = np.array(lms[tip_i], dtype=np.float32)
    pip   = np.array(lms[pip_i], dtype=np.float32)
    return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist) * 1.08


def _finger_states(lms):
    return {
        'index':  _is_extended(lms, 8,  6),
        'middle': _is_extended(lms, 12, 10),
        'ring':   _is_extended(lms, 16, 14),
        'pinky':  _is_extended(lms, 20, 18),
    }


def _index_direction(lms):
    """Unit vector from index MCP (5) to index TIP (8)."""
    mcp = np.array(lms[5], dtype=np.float32)
    tip = np.array(lms[8], dtype=np.float32)
    d   = tip - mcp
    n   = np.linalg.norm(d)
    return d / max(n, 1.0)


def _angle_deg(a, b):
    """Angle in degrees between two 2-D vectors."""
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


# ── Sign detection functions ──────────────────────────────────────────────────

def check_shadow_clone(hand0, hand1):
    """
    Returns True if both hands are making the Shadow Clone cross sign.
    hand0, hand1 = dicts from HandTracker.process().
    """
    lms0 = hand0['landmarks']
    lms1 = hand1['landmarks']

    # 1. Both hands: index + middle extended, ring + pinky curled
    st0 = _finger_states(lms0)
    st1 = _finger_states(lms1)

    ok0 = st0['index'] and st0['middle'] and not st0['ring'] and not st0['pinky']
    ok1 = st1['index'] and st1['middle'] and not st1['ring'] and not st1['pinky']
    if not (ok0 and ok1):
        return False

    # 2. Hands close together (wrist-to-wrist)
    w0   = np.array(lms0[0], dtype=np.float32)
    w1   = np.array(lms1[0], dtype=np.float32)
    dist = float(np.linalg.norm(w0 - w1))
    if dist > 320:
        return False

    # 3. Index directions must be roughly perpendicular (cross)
    dir0 = _index_direction(lms0)
    dir1 = _index_direction(lms1)
    ang  = _angle_deg(dir0, dir1)
    if not (50 < ang < 130):
        return False

    return True


# ── Sign registry — add more signs here later ─────────────────────────────────

_SIGNS = {
    'shadow_clone': {
        'check':    check_shadow_clone,      # fn(hand0, hand1) → bool
        'label':    'Kage Bunshin no Jutsu',
        'ref_img':  'chakra/clone_shadow.jpg',
        'voice':    'Kage Bunshin No Jutsu',
        'needs_two': True,
    },
}


# ── Main detector class ───────────────────────────────────────────────────────

class NarutoSignDetector:
    """
    Checks every registered sign each frame and fires when one is held
    long enough.  Replaces the old simplified hand_sign_detector for
    the Naruto theme.
    """

    def __init__(self):
        self._active_sign = None    # sign key currently being held
        self._since       = None    # time it started
        self._cool        = {}      # sign key → last trigger time

    def update(self, hands: list) -> str | None:
        """
        Call once per frame with the full hands list from HandTracker.
        Returns the triggered sign key (e.g. 'shadow_clone') or None.
        """
        now      = time.time()
        detected = None

        for key, info in _SIGNS.items():
            if info['needs_two'] and len(hands) < 2:
                continue
            if info['needs_two']:
                if info['check'](hands[0], hands[1]):
                    detected = key
                    break
            else:
                if len(hands) >= 1 and info['check'](hands[0]):
                    detected = key
                    break

        if detected != self._active_sign:
            self._active_sign = detected
            self._since       = now if detected else None
            return None

        if detected is None or self._since is None:
            return None

        if now - self._cool.get(detected, 0) < COOLDOWN:
            return None

        if now - self._since >= HOLD_SECS:
            self._cool[detected] = now
            self._since          = now
            return detected

        return None

    # ── State for HUD ─────────────────────────────────────────────────────────

    @property
    def hold_progress(self) -> float:
        if not self._active_sign or self._since is None:
            return 0.0
        return min(1.0, (time.time() - self._since) / HOLD_SECS)

    @property
    def active_sign(self) -> str | None:
        return self._active_sign

    @property
    def active_label(self) -> str:
        if self._active_sign and self._active_sign in _SIGNS:
            return _SIGNS[self._active_sign]['label']
        return ''

    @property
    def active_ref_img(self) -> str:
        if self._active_sign and self._active_sign in _SIGNS:
            return _SIGNS[self._active_sign].get('ref_img', '')
        return ''

    @property
    def active_voice(self) -> str:
        if self._active_sign and self._active_sign in _SIGNS:
            return _SIGNS[self._active_sign].get('voice', '')
        return ''

    def get_info(self, sign_key: str) -> dict:
        return _SIGNS.get(sign_key, {})

    def reset(self):
        self._active_sign = None
        self._since       = None
        self._cool.clear()
