"""
World selection screen — immersive, atmospheric.

Each world fills the entire screen with its own colour, animated
particles, and identity.  Cards are minimal icons at the bottom;
the centre showcases the selected world.

Controls
--------
1–8 / LEFT-RIGHT / A-D  navigate
ENTER / SPACE           enter the world
ESC                     exit
"""
import cv2
import numpy as np
import math
import time
import random

from theme import THEMES, DEFAULT_THEME

_W, _H = 1280, 720
_N     = len(THEMES)

# ── Per-world atmosphere definitions ─────────────────────────────────────────
# bg_top / bg_bot: BGR gradient colours
# particles: list of strings describing particle style
_ATMOS = {
    'academy':   dict(bg_top=(8, 2, 18),    bg_bot=(30, 8, 60),
                     ptcl='gold_dust',  icon='✦'),
    'naruto':    dict(bg_top=(5, 20, 40),   bg_bot=(10, 55, 120),
                     ptcl='leaves',     icon='卍'),
    'one_piece': dict(bg_top=(60, 30, 10),  bg_bot=(120, 70, 15),
                     ptcl='sea_foam',   icon='☠'),
    'game':      dict(bg_top=(5, 30, 5),    bg_bot=(10, 80, 10),
                     ptcl='fruits',     icon='🎮'),
}

# Fruit mini-icons for game world (drawn as coloured circles)
_FRUIT_COLS = [(30,30,210),(0,140,255),(40,180,40),(0,230,255),(50,60,220)]


# ── Particle systems ──────────────────────────────────────────────────────────

class _Particles:
    def __init__(self, style):
        self._style = style
        self._pts   = []
        self._t     = 0
        self._init()

    def _init(self):
        self._pts = []
        for _ in range(60):
            self._pts.append(self._new())

    def _new(self):
        s = self._style
        x = random.uniform(0, _W)
        y = random.uniform(0, _H)
        if s == 'embers':
            return [x, _H, random.uniform(-1,1), random.uniform(-5,-2),
                    random.choice([(0,80,255),(0,120,255),(0,180,255)])]
        if s == 'bubbles':
            return [x, y, random.uniform(-0.5,0.5), random.uniform(-1.5,-0.5),
                    (255,220,100), random.randint(4,12)]
        if s == 'leaves':
            return [random.uniform(-20,_W), random.uniform(-20,0),
                    random.uniform(-1,1), random.uniform(1,3),
                    (0,180,255), random.uniform(0,2*math.pi)]
        if s == 'void_motes':
            return [x, y, random.uniform(-0.5,0.5), random.uniform(-0.3,0.3),
                    (255,0,200), random.uniform(2,6)]
        if s == 'sea_foam':
            return [x, _H*0.6+random.uniform(0,80), random.uniform(1,3), 0,
                    (220,200,180), random.randint(5,15)]
        if s == 'light_rays':
            ang = random.uniform(0, 2*math.pi)
            return [_W//2, _H//2, math.cos(ang)*0.8, math.sin(ang)*0.8,
                    (255,255,220), random.uniform(50,200)]
        if s == 'fruits':
            return [x, _H+40, random.uniform(-1,1), random.uniform(-8,-4),
                    random.choice(_FRUIT_COLS), random.randint(14,22)]
        # gold_dust default
        return [x, y, random.uniform(-0.4,0.4), random.uniform(-0.8,-0.2),
                (0,200,255), random.randint(2,5)]

    def update_draw(self, img, t):
        s  = self._style
        dead = []
        for i, p in enumerate(self._pts):
            p[0] += p[2]
            p[1] += p[3]
            px, py = int(p[0]), int(p[1])

            if s == 'embers':
                col = p[4]
                fade = max(0.0, 1.0 - ((_H - p[1]) / _H))
                c = tuple(int(ch * fade) for ch in col)
                if 0 <= px < _W and 0 <= py < _H:
                    cv2.circle(img, (px, py), 3, c, -1)
                if p[1] < -10:
                    dead.append(i)

            elif s == 'bubbles':
                if 0 <= px < _W and 0 <= py < _H:
                    cv2.circle(img, (px, py), p[5], (200,220,255), 1)
                if p[1] < -20:
                    dead.append(i)

            elif s == 'leaves':
                p[5] += 0.05   # rotate
                p[2]  = math.sin(p[5]) * 1.2
                p[3]  = abs(math.cos(p[5] * 0.5)) * 2 + 0.5
                if 0 <= px < _W and 0 <= py < _H:
                    tip = (px + int(8*math.cos(p[5])), py + int(5*math.sin(p[5])))
                    cv2.ellipse(img, (px, py), (10,5), int(math.degrees(p[5])),
                                0, 360, p[4], -1)
                if py > _H + 20:
                    dead.append(i)

            elif s == 'void_motes':
                if 0 <= px < _W and 0 <= py < _H:
                    pulse = 0.5 + 0.5 * math.sin(t * 3 + px)
                    cv2.circle(img, (px, py), int(p[5] * pulse), p[4], -1)
                # wrap around
                if p[0] < 0:   p[0] = _W
                if p[0] > _W:  p[0] = 0
                if p[1] < 0:   p[1] = _H
                if p[1] > _H:  p[1] = 0

            elif s == 'sea_foam':
                p[0] = (p[0] + p[2]) % _W
                if 0 <= px < _W and 0 <= py < _H:
                    cv2.circle(img, (px, py), p[5], p[4], 1)

            elif s == 'light_rays':
                if 0 <= px < _W and 0 <= py < _H:
                    cv2.line(img, (_W//2, _H//2), (px, py),
                             (int(60*max(0, 1-p[5]/_W)),)*3, 1)
                p[5] -= 1
                if p[5] < 0:
                    dead.append(i)

            elif s == 'fruits':
                p[3] += 0.35  # gravity
                if 0 <= px < _W and 0 <= py < _H:
                    cv2.circle(img, (px, py), p[5], p[4], -1)
                    cv2.circle(img, (px, py), p[5], (255,255,255), 1)
                if py > _H + 60:
                    dead.append(i)

            else:  # gold_dust
                if 0 <= px < _W and 0 <= py < _H:
                    a = max(0.0, 1.0 - ((_H//2 - p[1]) / (_H//2)))
                    c = tuple(int(ch * a) for ch in p[4])
                    cv2.circle(img, (px, py), p[5], c, -1)
                if p[1] < -10:
                    dead.append(i)

        for i in reversed(dead):
            self._pts[i] = self._new()


# ── Background gradient ───────────────────────────────────────────────────────

def _draw_bg(img, top, bot, accent):
    for y in range(_H):
        t   = y / _H
        row = tuple(int(top[c] + (bot[c] - top[c]) * t) for c in range(3))
        img[y, :] = row
    # Vignette (dark edges)
    cx, cy = _W // 2, _H // 2
    for y in range(0, _H, 2):
        for band in range(80):
            d = min(band, _H - 1 - y)
            a = (80 - band) / 80 * 0.5
            img[y, :] = np.clip(img[y, :].astype(float) * (1 - a * (y < 80 or y > _H-80)), 0, 255)
            break  # just top/bottom edges via different method

    # Simpler vignette via radial darkening mask
    Y, X = np.mgrid[0:_H, 0:_W].astype(np.float32)
    dist = np.sqrt(((X - cx) / (_W * 0.6))**2 + ((Y - cy) / (_H * 0.6))**2)
    vign = np.clip(1.0 - dist * 0.5, 0.3, 1.0)
    img[:] = np.clip(img.astype(np.float32) * vign[:, :, None], 0, 255).astype(np.uint8)


# ── Bottom nav strip ──────────────────────────────────────────────────────────

def _draw_nav(img, selected, t):
    strip_y = _H - 70
    spacing = 80
    total   = _N * spacing
    start_x = (_W - total) // 2 + spacing // 2

    for i, theme in enumerate(THEMES):
        cx = start_x + i * spacing
        cy = strip_y + 20
        sel = (i == selected)

        r   = 22 if sel else 14
        col = theme.accent if sel else tuple(c // 3 for c in theme.accent)

        # Glow if selected
        if sel:
            pulse = 0.7 + 0.3 * math.sin(t * 5)
            glow  = np.zeros_like(img)
            cv2.circle(glow, (cx, cy), r + 10, theme.accent, -1)
            bl = cv2.GaussianBlur(glow, (21, 21), 0)
            img[:] = np.clip(img.astype(np.float32) + bl.astype(np.float32) * 0.5 * pulse,
                             0, 255).astype(np.uint8)

        cv2.circle(img, (cx, cy), r, col, -1)
        cv2.circle(img, (cx, cy), r, (255, 255, 255) if sel else (60, 60, 60), 1)

        # Key number
        cv2.putText(img, theme.key, (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.45,
                    (255, 255, 255) if sel else (120, 120, 120), 1, cv2.LINE_AA)


# ── Centre showcase ───────────────────────────────────────────────────────────

def _draw_centre(img, theme, t):
    font_big  = cv2.FONT_HERSHEY_COMPLEX
    font_sm   = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # ── World name (large) ────────────────────────────────────────────
    name   = theme.name.upper()
    scale  = 2.0
    thick  = 3
    (tw, th), _ = cv2.getTextSize(name, font_big, scale, thick)
    tx = (_W - tw) // 2
    ty = _H // 2 - 60

    # Glow halo
    glow_layer = np.zeros_like(img)
    cv2.putText(glow_layer, name, (tx, ty), font_big, scale, theme.accent, thick + 8, cv2.LINE_AA)
    bl = cv2.GaussianBlur(glow_layer, (31, 31), 0)
    img[:] = np.clip(img.astype(np.float32) + bl.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)

    # Core text
    cv2.putText(img, name, (tx + 2, ty + 2), font_big, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, name, (tx, ty), font_big, scale, theme.accent, thick, cv2.LINE_AA)

    # ── Tagline ───────────────────────────────────────────────────────
    tag = theme.tagline
    (tw2, _), _ = cv2.getTextSize(tag, font_sm, 0.80, 1)
    cv2.putText(img, tag, ((_W - tw2) // 2, ty + th + 20),
                font_sm, 0.80, tuple(int(c * 0.70) for c in theme.accent), 1, cv2.LINE_AA)

    # ── Element label badge ───────────────────────────────────────────
    badge = f'[ {theme.element_label} ]'
    (bw, _), _ = cv2.getTextSize(badge, font_sm, 0.60, 1)
    bx = (_W - bw) // 2
    by = ty + th + 54
    cv2.putText(img, badge, (bx, by), font_sm, 0.60,
                tuple(int(c * 0.55) for c in theme.accent), 1, cv2.LINE_AA)

    # ── Horizontal accent line ────────────────────────────────────────
    line_y = ty - 28
    line_w = min(600, tw + 80)
    lx     = (_W - line_w) // 2
    pulse  = 0.5 + 0.5 * math.sin(t * 2)
    col    = tuple(int(c * pulse) for c in theme.accent)
    cv2.line(img, (lx, line_y), (lx + line_w, line_y), col, 1)
    cv2.line(img, (lx, by + 14), (lx + line_w, by + 14), col, 1)


# ── Public entry point ────────────────────────────────────────────────────────

def show_selection_screen():
    win = 'AETHER SCRIPT'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, _W, _H)

    selected = 0
    t0       = time.time()

    # Pre-build particle systems for all themes
    ptcl_sys = {t.id: _Particles(_ATMOS[t.id]['ptcl']) for t in THEMES}

    while True:
        t     = time.time() - t0
        theme = THEMES[selected]
        atm   = _ATMOS[theme.id]

        img = np.zeros((_H, _W, 3), np.uint8)

        # ── Atmospheric background ────────────────────────────────────────
        _draw_bg(img, atm['bg_top'], atm['bg_bot'], theme.accent)

        # ── Animated particles ────────────────────────────────────────────
        ptcl_sys[theme.id].update_draw(img, t)

        # ── Centre world showcase ─────────────────────────────────────────
        _draw_centre(img, theme, t)

        # ── Bottom navigation dots ────────────────────────────────────────
        _draw_nav(img, selected, t)

        # ── Footer hint ───────────────────────────────────────────────────
        hint = 'A / D  or  ◄ ►  to explore   |   ENTER  to enter the world   |   ESC  to exit'
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.45, 1)
        cv2.putText(img, hint, ((_W - hw) // 2, _H - 12),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.45, (70, 60, 80), 1, cv2.LINE_AA)

        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(win)
            return DEFAULT_THEME
        elif key in (13, ord(' ')):
            cv2.destroyWindow(win)
            return THEMES[selected]
        elif ord('1') <= key <= ord('8'):
            idx = key - ord('1')
            if idx < _N:
                selected = idx
        elif key in (81, ord('a'), 2):
            selected = (selected - 1) % _N
        elif key in (83, ord('d'), 3):
            selected = (selected + 1) % _N
