"""
Screen-level post-processing effects.

All effects are composited in `ScreenEffects.apply(frame)` and return the
modified frame.  Main loop calls this once per frame after magic overlay
compositing but before the HUD is drawn.
"""
import cv2
import numpy as np
import math
import config


class ScreenEffects:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        # ── Flash ─────────────────────────────────────────────────────────────
        self._flash_f   = 0
        self._flash_max = 0
        self._flash_col = np.zeros(3, np.uint8)

        # ── Screen shake ──────────────────────────────────────────────────────
        self._shake_f   = 0
        self._shake_max = 0
        self._shake_mag = 0.0

        # ── Spell name overlay ────────────────────────────────────────────────
        self._name_f    = 0
        self._name_max  = 0
        self._name_text = ''
        self._name_col  = config.GOLD
        self._name_sub  = ''           # element subtitle

        # ── Background tint ───────────────────────────────────────────────────
        self._tint_f    = 0
        self._tint_max  = 0
        self._tint_col  = np.zeros(3, np.uint8)

        # ── Vortex (Domain of Aether) ─────────────────────────────────────────
        self._vortex_f   = 0
        self._vortex_max = 0

        # ── Chromatic aberration ──────────────────────────────────────────────
        self._chroma_f   = 0
        self._chroma_max = 0
        self._chroma_str = 4

        # ── Scanline / atmosphere ─────────────────────────────────────────────
        self._scan_alpha = 0.04      # subtle always-on CRT scanline overlay

    # ── Trigger API ───────────────────────────────────────────────────────────

    def flash(self, color=None, duration=10):
        self._flash_f   = duration
        self._flash_max = duration
        col = color or config.GOLD
        self._flash_col = np.array(col, np.uint8)

    def shake(self, magnitude=10, duration=7):
        self._shake_f   = duration
        self._shake_max = duration
        self._shake_mag = float(magnitude)

    def show_spell_name(self, name, color, element='', duration=80):
        self._name_text = name
        self._name_col  = color
        self._name_sub  = element
        self._name_f    = duration
        self._name_max  = duration

    def tint(self, color, duration=30):
        self._tint_f   = duration
        self._tint_max = duration
        self._tint_col = np.array(color, np.uint8)

    def vortex(self, duration=100):
        self._vortex_f   = duration
        self._vortex_max = duration

    def chromatic_aberration(self, strength=4, duration=8):
        """RGB channel split for cinematic impact on big spell releases."""
        self._chroma_f   = duration
        self._chroma_max = duration
        self._chroma_str = strength

    # ── Main apply ────────────────────────────────────────────────────────────

    def apply(self, frame):
        frame = self._apply_scanlines(frame)
        frame = self._apply_tint(frame)
        frame = self._apply_vortex(frame)
        frame = self._apply_flash(frame)
        frame = self._apply_chroma(frame)
        frame = self._apply_shake(frame)
        frame = self._apply_name(frame)
        return frame

    # ── Internal effect steps ─────────────────────────────────────────────────

    def _apply_chroma(self, frame):
        """Chromatic aberration — offset R and B channels in opposite directions."""
        if self._chroma_f <= 0:
            return frame
        decay = self._chroma_f / max(self._chroma_max, 1)
        s     = int(self._chroma_str * decay)
        if s < 1:
            self._chroma_f -= 1
            return frame
        b, g, r = cv2.split(frame)
        h, w    = frame.shape[:2]
        M_r = np.float32([[1, 0,  s], [0, 1, 0]])
        M_b = np.float32([[1, 0, -s], [0, 1, 0]])
        r   = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b   = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        self._chroma_f -= 1
        return cv2.merge([b, g, r])

    def _apply_scanlines(self, frame):
        """Subtle alternating horizontal line darkening for atmosphere."""
        if self._scan_alpha <= 0:
            return frame
        mask = np.zeros((self.h, self.w), np.float32)
        mask[::2, :] = self._scan_alpha
        darkened = frame.astype(np.float32)
        darkened -= darkened * mask[:, :, np.newaxis]
        return np.clip(darkened, 0, 255).astype(np.uint8)

    def _apply_tint(self, frame):
        if self._tint_f <= 0:
            return frame
        alpha = (self._tint_f / max(self._tint_max, 1)) * 0.20
        tint  = np.zeros_like(frame)
        tint[:] = self._tint_col
        out = cv2.addWeighted(frame, 1.0, tint, alpha, 0)
        self._tint_f -= 1
        return out

    def _apply_vortex(self, frame):
        if self._vortex_f <= 0:
            return frame
        t       = self._vortex_max - self._vortex_f
        fade    = self._vortex_f / self._vortex_max
        overlay = np.zeros_like(frame)
        cx, cy  = self.w // 2, self.h // 2

        # 5 expanding rings
        for ring in range(5):
            r = int((t * 4 + ring * 70) % (max(self.w, self.h) * 0.9))
            if r < 4:
                continue
            a = max(0.0, fade * (1.0 - ring * 0.18))
            col = tuple(int(ch * a) for ch in config.GOLD)
            cv2.circle(overlay, (cx, cy), r, col, 1)

        # Rotating spokes
        n_spokes = 12
        for i in range(n_spokes):
            ang  = t * 0.03 + i * (2 * math.pi / n_spokes)
            x2   = int(cx + self.w * 0.6 * math.cos(ang))
            y2   = int(cy + self.h * 0.6 * math.sin(ang))
            a    = max(0.0, fade * 0.25)
            col  = tuple(int(ch * a) for ch in config.NEON_BLUE)
            cv2.line(overlay, (cx, cy), (x2, y2), col, 1)

        out = cv2.addWeighted(frame, 1.0, overlay, fade * 0.5, 0)
        self._vortex_f -= 1
        return out

    def _apply_flash(self, frame):
        if self._flash_f <= 0:
            return frame
        alpha = (self._flash_f / max(self._flash_max, 1)) * 0.65
        fl    = np.zeros_like(frame)
        fl[:] = self._flash_col
        out = cv2.addWeighted(frame, 1 - alpha, fl, alpha, 0)
        self._flash_f -= 1
        return out

    def _apply_shake(self, frame):
        if self._shake_f <= 0:
            return frame
        decay = self._shake_f / max(self._shake_max, 1)
        mag   = self._shake_mag * decay
        dx    = int(np.random.uniform(-mag, mag))
        dy    = int(np.random.uniform(-mag * 0.6, mag * 0.6))
        M     = np.float32([[1, 0, dx], [0, 1, dy]])
        out   = cv2.warpAffine(frame, M, (self.w, self.h),
                               borderMode=cv2.BORDER_REFLECT)
        self._shake_f -= 1
        return out

    def _apply_name(self, frame):
        if self._name_f <= 0 or not self._name_text:
            return frame

        # fade-in over first 15 frames, fade-out over last 20
        progress_in  = (self._name_max - self._name_f) / 15.0
        progress_out = self._name_f / 20.0
        alpha        = min(1.0, progress_in, progress_out)

        col     = tuple(int(c * alpha) for c in self._name_col)
        glow_c  = tuple(int(c * alpha * 0.35) for c in self._name_col)
        shadow  = (0, 0, 0)

        font  = cv2.FONT_HERSHEY_COMPLEX
        scale = 1.35
        thick = 2

        (tw, th), _ = cv2.getTextSize(self._name_text, font, scale, thick)
        tx = (self.w - tw) // 2
        ty = int(self.h * 0.30)

        # Multi-pass glow halos (widest to narrowest)
        for off in [8, 5, 3]:
            for ddx, ddy in [(-off, 0), (off, 0), (0, -off), (0, off)]:
                cv2.putText(frame, self._name_text, (tx + ddx, ty + ddy),
                            font, scale, glow_c, thick + off, cv2.LINE_AA)

        # Drop shadow
        cv2.putText(frame, self._name_text, (tx + 3, ty + 4),
                    font, scale, shadow, thick + 2, cv2.LINE_AA)

        # Core text
        cv2.putText(frame, self._name_text, (tx, ty),
                    font, scale, col, thick, cv2.LINE_AA)

        # Element subtitle
        if self._name_sub:
            sub_scale = 0.55
            (sw, _), _ = cv2.getTextSize(self._name_sub, font, sub_scale, 1)
            sx  = (self.w - sw) // 2
            sy  = ty + th + 16
            sub_c = tuple(int(c * alpha * 0.75) for c in self._name_col)
            cv2.putText(frame, self._name_sub, (sx + 2, sy + 2),
                        font, sub_scale, shadow, 2, cv2.LINE_AA)
            cv2.putText(frame, self._name_sub, (sx, sy),
                        font, sub_scale, sub_c, 1, cv2.LINE_AA)

        self._name_f -= 1
        return frame
