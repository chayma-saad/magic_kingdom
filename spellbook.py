"""
In-game spell reference book.
Loads the JPEG/PNG images from the spells/ folder and renders them as a
right-side overlay panel so the player can see every spell's visual.
Toggle with the 'B' key.
"""
import cv2
import numpy as np
import os
import math
import time

_SPELLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spells')

# Thumbnail dimensions inside the panel
_THUMB_W   = 138
_THUMB_H   = 96
_COLS      = 2
_H_GUTTER  = 10   # gap between columns
_V_GUTTER  = 8    # gap between rows
_LABEL_H   = 16   # space below each thumbnail for the name

_HEADER_H  = 36


class SpellBook:
    def __init__(self):
        self.visible  = False   # hidden on launch; show a toast instead
        self._entries = []          # list of (display_name, thumbnail_bgr)
        self._scroll  = 0           # first visible row index
        self._panel_h = 0           # computed once after loading
        self._max_visible_rows = 1
        self._welcome_until = time.time() + 3.0
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.isdir(_SPELLS_DIR):
            return
        for fname in sorted(os.listdir(_SPELLS_DIR)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(_SPELLS_DIR, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            thumb = cv2.resize(img, (_THUMB_W, _THUMB_H),
                                interpolation=cv2.INTER_AREA)
            name  = (os.path.splitext(fname)[0]
                     .replace('_', ' ')
                     .replace('  ', ' ')
                     .strip())
            self._entries.append((name, thumb))

    # ── Public API ────────────────────────────────────────────────────────────

    def toggle(self):
        self.visible = not self.visible
        self._scroll = 0

    def scroll_up(self):
        self._scroll = max(0, self._scroll - 1)

    def scroll_down(self, max_visible_rows=None):
        if max_visible_rows is None:
            max_visible_rows = self._max_visible_rows
        self._scroll = min(max(0, max_visible_rows - 1), self._scroll + 1)

    def _draw_toast(self, frame, message):
        H, W = frame.shape[:2]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        (tw, th), _ = cv2.getTextSize(message, font, 0.62, 1)
        pad_x, pad_y = 18, 12
        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2
        x = max(12, (W - box_w) // 2)
        y = 18
        overlay = frame[y: y + box_h, x: x + box_w].astype(np.float32)
        frame[y: y + box_h, x: x + box_w] = np.clip(overlay * 0.42, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (x, y), (x + box_w - 1, y + box_h - 1), (90, 70, 110), 1)
        cv2.putText(frame, message, (x + pad_x, y + pad_y + th - 2),
                    font, 0.62, (235, 225, 240), 1, cv2.LINE_AA)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, frame):
        if not self._entries:
            return

        H, W = frame.shape[:2]
        panel_w = max(240, int(W * 0.24))
        content_w = _COLS * _THUMB_W + (_COLS - 1) * _H_GUTTER + 24
        cols = _COLS if panel_w >= content_w else 1
        row_h = _THUMB_H + _V_GUTTER + _LABEL_H

        # How many rows fit in the frame (minus header)
        max_visible_rows = max(1, (H - _HEADER_H - 12) // row_h)
        total_rows       = math.ceil(len(self._entries) / cols)
        self._max_visible_rows = max_visible_rows

        # Clamp scroll
        self._scroll = min(self._scroll, max(0, total_rows - max_visible_rows))

        panel_h = _HEADER_H + max_visible_rows * row_h + 10
        panel_h = min(panel_h, H)

        if not self.visible:
            if time.time() < self._welcome_until:
                self._draw_toast(frame, 'Welcome - press B to open your spell book')
            return

        # ── Build panel ──────────────────────────────────────────────────────
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (14, 4, 22)

        # Outer border
        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1),
                      (80, 50, 110), 1)

        # Header
        cv2.putText(panel, 'SPELL BOOK', (10, 23),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65,
                    (180, 150, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, '[B] close', (panel_w - 76, 23),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.42,
                    (90, 70, 110), 1, cv2.LINE_AA)
        cv2.line(panel, (6, _HEADER_H - 4), (panel_w - 6, _HEADER_H - 4),
                 (55, 35, 75), 1)

        # Thumbnails
        for slot in range(max_visible_rows * cols):
            entry_idx = self._scroll * cols + slot
            if entry_idx >= len(self._entries):
                break

            name, thumb = self._entries[entry_idx]
            col = slot % cols
            row = slot // cols

            tx = 12 + col * (_THUMB_W + _H_GUTTER)
            ty = _HEADER_H + row * row_h + 4

            if ty + _THUMB_H > panel_h:
                break

            # Slightly dimmed thumbnail
            dim = (thumb.astype(np.float32) * 0.80).clip(0, 255).astype(np.uint8)
            panel[ty: ty + _THUMB_H, tx: tx + _THUMB_W] = dim

            # Thumbnail border
            cv2.rectangle(panel, (tx, ty), (tx + _THUMB_W, ty + _THUMB_H),
                          (100, 65, 130), 1)

            # Name label (truncated to fit column width)
            label = name[:21] + '..' if len(name) > 23 else name
            cv2.putText(panel, label, (tx, ty + _THUMB_H + 13),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.37,
                        (140, 115, 165), 1, cv2.LINE_AA)

        # Scroll hint
        if total_rows > max_visible_rows:
            hint = f'{self._scroll + 1}/{total_rows - max_visible_rows + 1}'
            cv2.putText(panel, hint, (10, panel_h - 6),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.38,
                        (80, 65, 95), 1, cv2.LINE_AA)

        # ── Blend onto right edge of frame ───────────────────────────────────
        px = W - panel_w
        ph = min(panel_h, H)
        roi = frame[0:ph, px: px + panel_w]
        frame[0:ph, px: px + panel_w] = cv2.addWeighted(
            roi, 0.25, panel[0:ph], 0.90, 0)
