"""
Trail — stores fingertip history and renders the glowing ink trail.

Rendering layers (all additively blended onto overlay):
  base_layer  — the core trail line, element colour fading to dim gold at tail
  glow_layer  — blurred halo around the core
  orb         — pulsing glowing dot at the live fingertip position
  sparkles    — small random bright flecks scattered along the trail
"""
import cv2
import numpy as np
from collections import deque
import config


class Trail:
    def __init__(self, max_len=config.TRAIL_MAX_LEN):
        self._pts  = deque(maxlen=max_len)
        self._fade = 0

    def add(self, pt):
        self._pts.append(pt)
        self._fade = 0

    def on_lost(self):
        self._fade += 1
        if self._fade >= config.TRAIL_FADE_FRAMES:
            self._pts.clear()

    def clear(self):
        self._pts.clear()
        self._fade = 0

    @property
    def pts(self):
        return list(self._pts)

    @property
    def fade_alpha(self):
        if self._fade == 0:
            return 1.0
        return max(0.0, 1.0 - self._fade / config.TRAIL_FADE_FRAMES)

    def __len__(self):
        return len(self._pts)


# ── Internal helper ───────────────────────────────────────────────────────────

def _lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colour tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _additive_blend(overlay, layer, glow_layer, blur_k=13, glow_alpha=0.55):
    """Blur glow_layer and additively merge both layers onto overlay."""
    blurred = cv2.GaussianBlur(glow_layer, (blur_k, blur_k), 0)
    tmp = overlay.astype(np.float32)
    tmp += layer.astype(np.float32)
    tmp += blurred.astype(np.float32) * glow_alpha
    np.clip(tmp, 0, 255, out=tmp)
    overlay[:] = tmp.astype(np.uint8)


# ── Renderer ──────────────────────────────────────────────────────────────────

class TrailRenderer:
    _TAIL_COLOR = (0, 80, 120)    # dim gold-blue tail colour

    def render(self, overlay, trail, element='NEUTRAL', charge=0.0):
        """
        overlay  : uint8 BGR array (modified in-place)
        trail    : Trail instance
        element  : current element string for colour
        charge   : 0.0-1.0  — how charged the active spell is (grows orb)
        """
        pts = trail.pts
        n   = len(pts)
        if n < 2:
            if n == 1:
                self._draw_orb(overlay, pts[0], element, charge, trail.fade_alpha)
            return

        tip_color  = config.ELEMENT_COLORS.get(element, config.GOLD)
        tail_color = self._TAIL_COLOR
        fa         = trail.fade_alpha

        base = np.zeros_like(overlay)
        glow = np.zeros_like(overlay)

        for i in range(1, n):
            t     = i / n                    # 0 = tail, 1 = tip
            alpha = t * fa
            if alpha < 0.02:
                continue

            # Dual-color gradient: tail_color → tip_color
            seg_col      = _lerp_color(tail_color, tip_color, t)
            draw_col     = tuple(min(255, int(ch * alpha))       for ch in seg_col)
            glow_col     = tuple(min(255, int(ch * alpha * 0.7)) for ch in seg_col)

            w_main = max(1, int(config.TRAIL_MAX_WIDTH * t))
            w_glow = w_main + 6

            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]),     int(pts[i][1]))

            cv2.line(base, p1, p2, draw_col, w_main)
            cv2.line(glow, p1, p2, glow_col, w_glow)

        _additive_blend(overlay, base, glow)

        # Sparkles — random bright flecks scattered along the trail
        self._draw_sparkles(overlay, pts, tip_color, fa)

        # Fingertip orb
        self._draw_orb(overlay, pts[-1], element, charge, fa)

    # ── Sub-renderers ─────────────────────────────────────────────────────────

    @staticmethod
    def _draw_orb(overlay, tip, element, charge, fa):
        """Pulsing glowing dot at the live fingertip — grows with charge."""
        import time as _t
        pulse  = 1.0 + 0.25 * np.sin(_t.time() * 8.0)
        base_r = int(5 + charge * 18)
        r      = max(3, int(base_r * pulse))

        color  = config.ELEMENT_COLORS.get(element, config.GOLD)
        alpha  = fa
        c      = tuple(min(255, int(ch * alpha)) for ch in color)
        white  = tuple(min(255, int(255 * alpha * 0.6)) for _ in range(3))

        orb_layer = np.zeros_like(overlay)
        cv2.circle(orb_layer, (int(tip[0]), int(tip[1])), r + 8, c, -1)
        blurred = cv2.GaussianBlur(orb_layer, (19, 19), 0)
        tmp = overlay.astype(np.float32) + blurred.astype(np.float32) * 0.6
        np.clip(tmp, 0, 255, out=tmp)
        overlay[:] = tmp.astype(np.uint8)

        # Bright core
        cv2.circle(overlay, (int(tip[0]), int(tip[1])), max(2, r // 2), white, -1)

    @staticmethod
    def _draw_sparkles(overlay, pts, color, fa, density=0.04):
        """Randomly place small bright dots along the trail."""
        n = len(pts)
        for i in range(n):
            if np.random.random() > density:
                continue
            x, y = int(pts[i][0]), int(pts[i][1])
            a    = (i / n) * fa
            c    = tuple(min(255, int(ch * a * 1.2)) for ch in color)
            r    = np.random.randint(1, 3)
            cv2.circle(overlay, (x, y), r, c, -1)
