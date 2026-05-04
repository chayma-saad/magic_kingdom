"""
Chakra visual and audio effects for the Naruto theme.

CloneEffect   — ghost copies of the user appear beside them (Shadow Clone)
play_jutsu    — plays the MP3 from the chakra folder via pygame
draw_sign_hud — shows reference card image + hold progress bar
"""
import cv2
import numpy as np
import math
import os
import time

import pygame

_DIR       = os.path.dirname(os.path.abspath(__file__))
_SOUND_DIR = os.path.join(_DIR, 'chakra')


# ── Audio ─────────────────────────────────────────────────────────────────────

def play_jutsu(filename: str = 'Kage Bushin No Jutsu.mp3'):
    """
    Play an MP3 from the chakra/ folder using pygame.mixer.
    pygame is already used by the project for audio so no new dependency.
    """
    path = os.path.join(_SOUND_DIR, filename)
    if not os.path.isfile(path):
        return
    try:
        # Use a separate channel so it doesn't interrupt background hum
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
    except Exception:
        pass


# ── Clone visual effect ───────────────────────────────────────────────────────

class CloneEffect:
    """
    Ghost clone visual for Shadow Clone Jutsu.

    On trigger:
      - White chakra flash
      - 4 blue-tinted ghost copies of the user materialize beside them
      - Each clone has a glowing edge outline so it reads clearly
      - Clones held solid for ~2.5 s then fade out
    """

    FLASH_FRAMES = 8
    FADE_IN      = 18
    HOLD         = 80
    FADE_OUT     = 35
    TOTAL        = FLASH_FRAMES + FADE_IN + HOLD + FADE_OUT

    # Pixel offsets for each clone (left, right, far-left, far-right)
    OFFSETS = [-210, 210, -400, 400]

    def __init__(self):
        self._snap  = None
        self._frame = 0
        self.active = False

    def trigger(self, current_frame: np.ndarray):
        self._snap  = current_frame.copy()
        self._frame = 0
        self.active = True

    def render(self, frame: np.ndarray):
        if not self.active or self._snap is None:
            return

        f    = self._frame
        H, W = frame.shape[:2]

        # ── Clone opacity curve ────────────────────────────────────────────
        fi = self.FLASH_FRAMES
        if f < fi:
            base_alpha = 0.0
        elif f < fi + self.FADE_IN:
            base_alpha = (f - fi) / self.FADE_IN
        elif f < fi + self.FADE_IN + self.HOLD:
            base_alpha = 1.0
        else:
            done       = f - fi - self.FADE_IN - self.HOLD
            base_alpha = max(0.0, 1.0 - done / self.FADE_OUT)

        # ── White chakra flash ────────────────────────────────────────────
        if f < self.FLASH_FRAMES:
            flash_a = (self.FLASH_FRAMES - f) / self.FLASH_FRAMES * 0.88
            white   = np.full_like(frame, 255)
            cv2.addWeighted(frame, 1 - flash_a, white, flash_a, 0, dst=frame)

        # ── Render each clone ─────────────────────────────────────────────
        if base_alpha > 0:
            for i, off in enumerate(self.OFFSETS):
                # Outer clones slightly less opaque
                alpha = base_alpha * (0.88 - i * 0.06)
                if alpha <= 0:
                    continue

                # Shift the stored snapshot
                M       = np.float32([[1, 0, off], [0, 1, 0]])
                shifted = cv2.warpAffine(self._snap, M, (W, H),
                                         borderMode=cv2.BORDER_REPLICATE)

                # Improve visibility: apply a subtle blue tint and an outline
                tint_col = (180, 140, 255)  # BGR light-blue tint
                tint_img = np.full_like(shifted, tint_col)
                tinted = cv2.addWeighted(shifted, 0.65, tint_img, 0.35, 0)

                # Create a soft outline from edges and blend it over the tint
                gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 120)
                if edges is not None and edges.any():
                    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                    outline = np.zeros_like(shifted)
                    outline[edges > 0] = (220, 220, 255)
                    cv2.addWeighted(tinted, 1.0, outline, 0.7 * alpha, 0, dst=tinted)

                # Blend the tinted/outlined clone onto the frame with slightly increased alpha
                blend_a = min(1.0, alpha * 1.15)
                cv2.addWeighted(frame, 1.0, tinted, blend_a, 0, dst=frame)

        self._frame += 1
        if self._frame >= self.TOTAL:
            self.active = False


# ── Sign HUD (reference card + progress bar) ──────────────────────────────────

_ref_cache: dict = {}


def _load_ref(path: str, target_h: int = 180) -> np.ndarray | None:
    if path in _ref_cache:
        return _ref_cache[path]
    full = os.path.join(_DIR, path)
    img  = cv2.imread(full)
    if img is None:
        _ref_cache[path] = None
        return None
    h, w  = img.shape[:2]
    scale = target_h / h
    img   = cv2.resize(img, (int(w * scale), target_h),
                       interpolation=cv2.INTER_AREA)
    _ref_cache[path] = img
    return img


def draw_sign_hud(frame: np.ndarray,
                  sign_label: str,
                  progress: float,
                  ref_img_path: str = '',
                  accent: tuple = (255, 180, 40)):
    H, W   = frame.shape[:2]
    margin = 12
    font   = cv2.FONT_HERSHEY_COMPLEX_SMALL

    ref   = _load_ref(ref_img_path) if ref_img_path else None
    ref_h = ref.shape[0] if ref is not None else 0
    ref_w = ref.shape[1] if ref is not None else 0

    panel_w = max(ref_w, 280) + margin * 2
    bar_h   = 14
    text_h  = 22
    panel_h = ref_h + text_h + bar_h + margin * 3

    px = W - panel_w - margin
    py = H - panel_h - margin

    # Dark translucent background
    roi    = frame[py: py + panel_h, px: px + panel_w].astype(np.float32)
    frame[py: py + panel_h, px: px + panel_w] = np.clip(roi * 0.30, 0, 255).astype(np.uint8)

    cv2.rectangle(frame, (px, py), (px + panel_w - 1, py + panel_h - 1), accent, 1)

    y = py + margin

    if ref is not None:
        rx = px + (panel_w - ref_w) // 2
        frame[y: y + ref_h, rx: rx + ref_w] = ref
        y += ref_h + margin // 2

    cv2.putText(frame, sign_label, (px + margin, y + 14),
                font, 0.52, accent, 1, cv2.LINE_AA)
    y += text_h

    bar_x = px + margin
    bar_w = panel_w - margin * 2
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (30, 20, 40), -1)
    fill = int(bar_w * progress)
    if fill > 0:
        if progress > 0.85:
            pulse = int(200 + 55 * math.sin(time.time() * 12))
            bc    = (min(255, accent[0] + 40),
                     min(255, accent[1] + 40),
                     min(255, accent[2] + 40))
        else:
            bc = accent
        cv2.rectangle(frame, (bar_x, y), (bar_x + fill, y + bar_h), bc, -1)
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), accent, 1)
