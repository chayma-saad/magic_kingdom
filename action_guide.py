"""
Action Guide — section-specific reference panel visible in every world.

Shows by default when the camera opens so the user always knows what to do.
Toggle with B key.

Magic      → spell images from spells/ folder + gesture names
Naruto     → hand sign images from chakra/ folder + hold instructions
One Piece  → arm-movement text guide
Game       → game controls text guide
"""
import cv2
import numpy as np
import math
import os
import time

_DIR = os.path.dirname(os.path.abspath(__file__))

# Panel layout
_THUMB_W   = 140
_THUMB_H   = 95
_COLS      = 2
_H_GAP     = 8
_V_GAP     = 6
_LABEL_H   = 18
_HEADER_H  = 38
_PANEL_W   = _COLS * _THUMB_W + (_COLS - 1) * _H_GAP + 24
_ROW_H     = _THUMB_H + _V_GAP + _LABEL_H

# Pure text entry (no image)
_TEXT_ENTRY = 'TEXT'


def _load_images_from(folder: str):
    """Return list of (display_name, bgr_image) from a folder of JPEGs/PNGs."""
    entries = []
    full = os.path.join(_DIR, folder)
    if not os.path.isdir(full):
        return entries
    for fname in sorted(os.listdir(full)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(os.path.join(full, fname))
        if img is None:
            continue
        thumb = cv2.resize(img, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
        name  = (os.path.splitext(fname)[0]
                 .replace('_', ' ').replace('  ', ' ').strip())
        entries.append((name, thumb))
    return entries


def _text_entries(lines):
    """Convert text lines into fake thumbnail entries (solid colour + text)."""
    entries = []
    for line in lines:
        if not line:
            continue
        label, *rest = line.split('→') if '→' in line else (line, [])
        label  = label.strip()
        detail = rest[0].strip() if rest else ''
        # Create a simple coloured tile
        tile = np.zeros((_THUMB_H, _THUMB_W, 3), dtype=np.uint8)
        tile[:] = (18, 12, 28)
        cv2.rectangle(tile, (2, 2), (_THUMB_W-3, _THUMB_H-3), (80, 55, 100), 1)
        # Main label (action)
        cv2.putText(tile, label[:18], (8, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.48, (200, 180, 220), 1, cv2.LINE_AA)
        # Detail (result)
        if detail:
            for di, part in enumerate(detail.split(',')[:3]):
                cv2.putText(tile, part.strip()[:18], (8, 50 + di * 16),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.38, (140, 120, 160), 1, cv2.LINE_AA)
        entries.append((label, tile))
    return entries


# ── Per-section content definitions ──────────────────────────────────────────

def _build_magic_entries():
    return _load_images_from('spells')


def _build_naruto_entries():
    imgs = _load_images_from('chakra')
    # Add text instructions as extra tiles
    instructions = [
        'Ram sign → Kage Bunshin',
        'Both hands cross → hold 1.5s',
        'Index+Middle up → fingers cross',
        'Keep hands close together',
    ]
    return imgs + _text_entries(instructions)


def _build_one_piece_entries():
    return _text_entries([
        'Punch sideways → Pistol',
        'Slam arm DOWN → Cannon',
        'Spread BOTH arms wide → Bazooka',
        'Bounce body 5x → Gear 2nd',
        'Hold arm out 2s → Gear 3rd',
        'Rapid alternating → Gatling',
    ])


def _build_game_entries():
    return _text_entries([
        'Move hand fast → Slash fruit',
        'Hit fruit → Score points',
        'Miss fruit → Lose a life',
        'Hit BOMB → Lose a life',
        'Chain slices → COMBO bonus',
        'Reach 150 pts → WIN',
        'Press R → Restart game',
    ])


_BUILDERS = {
    'magic':     _build_magic_entries,
    'naruto':    _build_naruto_entries,
    'one_piece': _build_one_piece_entries,
    'game':      _build_game_entries,
}

_TITLES = {
    'magic':     'SPELL BOOK',
    'naruto':    'JUTSU GUIDE',
    'one_piece': 'MOVE GUIDE',
    'game':      'HOW TO PLAY',
}

_ACCENTS = {
    'magic':     (180, 150, 210),
    'naruto':    (80, 160, 255),
    'one_piece': (60, 100, 255),
    'game':      (60, 200, 80),
}


class ActionGuide:
    """Universal action guide — replaces the old magic-only SpellBook."""

    def __init__(self):
        self.visible   = False  # press B to open
        self._entries  = []
        self._scroll   = 0
        self._category = None
        self._title    = 'GUIDE'
        self._accent   = (180, 150, 210)

    def setup(self, theme):
        """Call once after theme is chosen. Loads appropriate content."""
        cat = getattr(theme, 'category', 'magic')
        # Naruto uses 'anime' category but has its own builder
        key = theme.id if theme.id in _BUILDERS else cat
        if key not in _BUILDERS:
            key = 'magic'
        self._entries  = _BUILDERS[key]()
        self._category = key
        self._title    = _TITLES.get(key, 'GUIDE')
        self._accent   = _ACCENTS.get(key, (180, 150, 210))
        self._scroll   = 0

    def toggle(self):
        self.visible  = not self.visible
        self._scroll  = 0

    def render(self, frame):
        if not self.visible or not self._entries:
            return

        H, W = frame.shape[:2]
        entries = self._entries
        max_rows    = max(1, (H - _HEADER_H - 12) // _ROW_H)
        total_rows  = math.ceil(len(entries) / _COLS)
        self._scroll = min(self._scroll, max(0, total_rows - max_rows))

        panel_h = _HEADER_H + max_rows * _ROW_H + 8
        panel_h = min(panel_h, H)

        panel = np.zeros((panel_h, _PANEL_W, 3), dtype=np.uint8)
        panel[:] = (12, 6, 20)
        cv2.rectangle(panel, (0, 0), (_PANEL_W-1, panel_h-1), self._accent, 1)

        # Header
        cv2.putText(panel, self._title, (10, 24),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, self._accent, 1, cv2.LINE_AA)
        cv2.putText(panel, '[B] close', (_PANEL_W - 76, 24),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.40, (80, 65, 95), 1, cv2.LINE_AA)
        cv2.line(panel, (6, _HEADER_H - 4), (_PANEL_W - 6, _HEADER_H - 4),
                 tuple(c // 2 for c in self._accent), 1)

        # Thumbnails
        for slot in range(max_rows * _COLS):
            idx = self._scroll * _COLS + slot
            if idx >= len(entries):
                break
            name, thumb = entries[idx]
            col  = slot % _COLS
            row  = slot // _COLS
            tx   = 12 + col * (_THUMB_W + _H_GAP)
            ty   = _HEADER_H + row * _ROW_H + 4
            if ty + _THUMB_H > panel_h:
                break

            # Slightly dim
            dim = (thumb.astype(np.float32) * 0.82).clip(0, 255).astype(np.uint8)
            panel[ty: ty+_THUMB_H, tx: tx+_THUMB_W] = dim
            cv2.rectangle(panel, (tx, ty), (tx+_THUMB_W, ty+_THUMB_H),
                          self._accent, 1)

            label = name[:21] + '..' if len(name) > 23 else name
            cv2.putText(panel, label, (tx, ty+_THUMB_H+13),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.36, self._accent, 1, cv2.LINE_AA)

        # Scroll hint
        if total_rows > max_rows:
            hint = f'{self._scroll+1}/{total_rows - max_rows + 1}'
            cv2.putText(panel, hint, (10, panel_h - 6),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.36, (80, 65, 95), 1, cv2.LINE_AA)

        # Paste on right edge
        px  = W - _PANEL_W
        ph  = min(panel_h, H)
        roi = frame[0:ph, px: px+_PANEL_W]
        frame[0:ph, px: px+_PANEL_W] = cv2.addWeighted(
            roi, 0.20, panel[0:ph], 0.92, 0)
