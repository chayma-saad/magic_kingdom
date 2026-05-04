"""
Spell System — maps (shape, element) → spell name, manages the
CHARGING → RELEASE → DECAY state machine, and renders spell visuals.

Cinematic renderers aim to match the reference images in the spells/ folder:
  Arcane Ward     — multi-layer rune rings + brilliant white core
  Ignition Pulse  — thick lava-column triangle + massive corner fire bursts
  Reality Slice   — wide prismatic energy blade + multi-arc lightning
  Void Collapse   — swirling debris vortex + bright ring edge
  Domain of Aether— screen-filling golden spiral storm
  Kage Bunshin    — ghost clone silhouettes + blue chakra eruption
  Chidori         — converging lightning into a crackling core
  Rasengan        — spinning concentric sphere + massive explosion
  Katon: Goka     — expanding fireball + full-screen fire wall
  Kagemane        — dark shadow tendrils creeping outward
  Gomu Gomu no Pistol  — thick elastic beam + shockwave rings
  Gomu Gomu no Cannon  — giant compressed sphere burst
  Gear Second     — red steam aura + rising heat particles
  Gear Third      — enormous inflating fist sphere + mega explosion
  Elephant Gatling— rotating multi-impact barrage
"""
import cv2
import numpy as np
import math
import time
from collections import deque
import config
try:
    from class_select import ClassConfig
except Exception:
    ClassConfig = None


# ── Default spell catalogue (shape, element) → spell ─────────────────────────

_SPELL_MAP = {
    ('circle',   'WATER'):   'Arcane Ward',
    ('circle',   'FIRE'):    'Arcane Ward',
    ('circle',   'LIGHT'):   'Arcane Ward',
    ('circle',   'VOID'):    'Arcane Ward',
    ('circle',   'NEUTRAL'): 'Arcane Ward',
    ('triangle', 'FIRE'):    'Ignition Pulse',
    ('triangle', 'WATER'):   'Tide Blade',
    ('triangle', 'LIGHT'):   'Prism Lance',
    ('triangle', 'VOID'):    'Shade Rend',
    ('triangle', 'NEUTRAL'): 'Sigil Strike',
    ('square',   'LIGHT'):   'Arcane Construct',
    ('square',   'FIRE'):    'Arcane Construct',
    ('square',   'WATER'):   'Arcane Construct',
    ('square',   'VOID'):    'Arcane Construct',
    ('square',   'NEUTRAL'): 'Arcane Construct',
    ('spiral',   'VOID'):    'Void Collapse',
    ('spiral',   'WATER'):   'Maelstrom',
    ('spiral',   'FIRE'):    'Firestorm',
    ('spiral',   'LIGHT'):   'Radiant Pulse',
    ('spiral',   'NEUTRAL'): 'Ether Spiral',
    ('line',     'FIRE'):    'Reality Slice',
    ('line',     'VOID'):    'Reality Slice',
    ('line',     'WATER'):   'Torrent Slash',
    ('line',     'LIGHT'):   'Flash Strike',
    ('line',     'NEUTRAL'): 'Reality Slice',
    ('circle_dual', 'WATER'):   'Domain of Aether',
    ('circle_dual', 'FIRE'):    'Domain of Aether',
    ('circle_dual', 'LIGHT'):   'Domain of Aether',
    ('circle_dual', 'VOID'):    'Domain of Aether',
    ('circle_dual', 'NEUTRAL'): 'Domain of Aether',
}

# ── Naruto theme spell map (trail shapes → jutsu) ─────────────────────────────
_NARUTO_MAP = {}
for e in ['FIRE', 'WATER', 'LIGHT', 'VOID', 'NEUTRAL']:
    _NARUTO_MAP[('circle',   e)] = 'Rasengan'
    _NARUTO_MAP[('line',     e)] = 'Chidori'
    _NARUTO_MAP[('triangle', e)] = 'Katon: Goka Mekkyaku'
    _NARUTO_MAP[('spiral',   e)] = 'Kage Bunshin'
    _NARUTO_MAP[('square',   e)] = 'Kagemane no Jutsu'

# ── One Piece theme spell map ─────────────────────────────────────────────────
_ONE_PIECE_MAP = {}
for e in ['FIRE', 'WATER', 'LIGHT', 'VOID', 'NEUTRAL']:
    _ONE_PIECE_MAP[('circle',   e)] = 'Gomu Gomu no Cannon'
    _ONE_PIECE_MAP[('line',     e)] = 'Gomu Gomu no Pistol'
    _ONE_PIECE_MAP[('triangle', e)] = 'Gear Third: King Kong Gun'
    _ONE_PIECE_MAP[('spiral',   e)] = 'Gear Second'
    _ONE_PIECE_MAP[('square',   e)] = 'Elephant Gatling'

_THEME_SPELL_MAPS = {
    'naruto':    _NARUTO_MAP,
    'one_piece': _ONE_PIECE_MAP,
}

# ── Combo catalogue ───────────────────────────────────────────────────────────
_COMBOS = {
    ('circle',   'triangle'): 'Exploding Ward',
    ('spiral',   'line'):     'Void Beam',
    ('square',   'circle'):   'Barrier Field',
    ('triangle', 'circle'):   'Nova Burst',
    ('line',     'circle'):   'Convergence',
    ('line',     'spiral'):   'Void Beam',
}

_POWER_SPELLS = {
    'Domain of Aether', 'Void Collapse', 'Nova Burst', 'Exploding Ward',
    'Kage Bunshin', 'Rasengan', 'Katon: Goka Mekkyaku',
    'Gear Third: King Kong Gun', 'Gear Second',
}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _glow_circle(overlay, center, radius, color, thickness=2, blur=13):
    tmp = np.zeros_like(overlay)
    cv2.circle(tmp, center, max(1, radius), color, thickness)
    bl  = cv2.GaussianBlur(tmp, (blur | 1, blur | 1), 0)
    out = overlay.astype(np.float32) + tmp.astype(np.float32) + bl.astype(np.float32) * 0.6
    np.clip(out, 0, 255, out=out)
    overlay[:] = out.astype(np.uint8)


def _glow_line(overlay, p1, p2, color, thickness=3, blur=11):
    tmp = np.zeros_like(overlay)
    cv2.line(tmp, p1, p2, color, thickness)
    bl  = cv2.GaussianBlur(tmp, (blur | 1, blur | 1), 0)
    out = overlay.astype(np.float32) + tmp.astype(np.float32) + bl.astype(np.float32) * 0.7
    np.clip(out, 0, 255, out=out)
    overlay[:] = out.astype(np.uint8)


def _glow_polygon(overlay, pts_arr, color, thickness=2, blur=13):
    tmp = np.zeros_like(overlay)
    cv2.polylines(tmp, [pts_arr.astype(np.int32)], True, color, thickness)
    bl  = cv2.GaussianBlur(tmp, (blur | 1, blur | 1), 0)
    out = overlay.astype(np.float32) + tmp.astype(np.float32) + bl.astype(np.float32) * 0.6
    np.clip(out, 0, 255, out=out)
    overlay[:] = out.astype(np.uint8)


def _filled_glow(overlay, center, radius, color, blur=17):
    tmp = np.zeros_like(overlay)
    cv2.circle(tmp, center, max(1, radius), color, -1)
    bl  = cv2.GaussianBlur(tmp, (blur | 1, blur | 1), 0)
    out = overlay.astype(np.float32) + bl.astype(np.float32) * 1.1
    np.clip(out, 0, 255, out=out)
    overlay[:] = out.astype(np.uint8)


def _batch_draw(overlay, draw_fn, blur_s=11, blur_w=27, w_s=0.80, w_w=0.40):
    """
    Batch all geometry into one buffer then apply TWO blur passes.
    ~5-8× faster than calling _glow_* helpers individually.
    draw_fn(buf) — should draw everything into buf (np array, same shape as overlay).
    """
    buf = np.zeros_like(overlay)
    draw_fn(buf)
    bl_s = cv2.GaussianBlur(buf, (blur_s | 1, blur_s | 1), 0)
    bl_w = cv2.GaussianBlur(buf, (blur_w | 1, blur_w | 1), 0)
    out  = (overlay.astype(np.float32)
            + buf.astype(np.float32)
            + bl_s.astype(np.float32) * w_s
            + bl_w.astype(np.float32) * w_w)
    np.clip(out, 0, 255, out=out)
    overlay[:] = out.astype(np.uint8)


def _screen_tint(overlay, color, strength=0.10):
    tint = np.zeros_like(overlay)
    tint[:] = color
    out = overlay.astype(np.float32) + tint.astype(np.float32) * strength
    np.clip(out, 0, 255, out=out)
    overlay[:] = out.astype(np.uint8)


# ── Active spell effect ───────────────────────────────────────────────────────

class SpellEffect:
    def __init__(self, name, element, trail_pts, frame_w, frame_h):
        self.name      = name
        self.element   = element
        self.color     = config.ELEMENT_COLORS.get(element, config.GOLD)
        self.state     = 'CHARGING'
        self.t         = 0
        self.charge_t  = time.time()
        self.release_t = None
        self.w         = frame_w
        self.h         = frame_h
        self.trail_pts = trail_pts or []
        self._released_particles = False

        if self.trail_pts:
            arr = np.asarray(self.trail_pts, dtype=np.float64)
            cx, cy = arr.mean(axis=0)
            self.center = (int(cx), int(cy))
            dists = np.linalg.norm(arr - arr.mean(axis=0), axis=1)
            self.radius = max(30, int(dists.mean()))
        else:
            self.center = (frame_w // 2, frame_h // 2)
            self.radius = 80

    @property
    def charge_duration(self):
        return time.time() - self.charge_t

    @property
    def charge_frac(self):
        return min(1.0, self.charge_duration / config.OVERLOAD_SECS)

    @property
    def decay_progress(self):
        if self.release_t is None:
            return 0.0
        return min(1.0, (time.time() - self.release_t) / config.DECAY_DURATION)

    def trigger_release(self):
        if self.state == 'CHARGING':
            self.state     = 'RELEASE'
            self.release_t = time.time()
            self._released_particles = False

    def is_done(self):
        return self.state == 'DECAY' and self.decay_progress >= 1.0

    def update(self):
        self.t += 1
        if self.state == 'RELEASE':
            if time.time() - self.release_t > config.RELEASE_DURATION:
                self.state = 'DECAY'

    def render(self, overlay, psys):
        alpha = 1.0
        if self.state == 'DECAY':
            alpha = max(0.0, 1.0 - self.decay_progress)
        if alpha < 0.02:
            return
        c  = tuple(int(ch * alpha) for ch in self.color)
        fn = _RENDERERS.get(self.name, _render_generic)
        fn(self, overlay, psys, c, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# ── Per-spell render functions (cinematic) ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _render_arcane_ward(fx, overlay, psys, c, alpha):
    """
    Sacred-geometry alchemy circle — FMA-style transmutation array.
    Outer hexagon → inner counter-hex → 6-pointed star → rings → lens-flare core.
    One batch draw + two blur passes instead of 10+ separate glow calls.
    """
    cx, cy = fx.center
    r, t   = fx.radius, fx.t
    charge = fx.charge_frac

    gold  = tuple(int(ch * alpha) for ch in config.GOLD)
    blue  = tuple(int(ch * alpha) for ch in config.NEON_BLUE)
    white = tuple(int(ch * alpha) for ch in (255, 255, 255))
    cc    = tuple(int(ch * alpha) for ch in c)

    def _draw(buf):
        # ── Outer hexagon (CW) ────────────────────────────────────────────
        hr = int(r * 1.28)
        hex1 = np.array([[int(cx + hr * math.cos(t * 0.022 + i * math.pi / 3)),
                          int(cy + hr * math.sin(t * 0.022 + i * math.pi / 3))]
                         for i in range(6)], dtype=np.int32)
        cv2.polylines(buf, [hex1], True, gold, 2)

        # ── Outer circles (3 pulsing rings) ───────────────────────────────
        for k in range(3):
            pulse = 1.0 + 0.06 * math.sin(t * 0.12 + k * 1.1)
            rr    = max(5, int(r * pulse) + k * 14)
            fade  = 1.0 - k * 0.25
            rc    = tuple(int(ch * fade) for ch in cc)
            cv2.circle(buf, (cx, cy), rr, rc, max(1, 4 - k))

        # ── Inner hexagon (CCW, offset by 30°) ───────────────────────────
        hr2  = int(r * 0.82)
        hex2 = np.array([[int(cx + hr2 * math.cos(-t * 0.032 + math.pi/6 + i * math.pi/3)),
                          int(cy + hr2 * math.sin(-t * 0.032 + math.pi/6 + i * math.pi/3))]
                         for i in range(6)], dtype=np.int32)
        cv2.polylines(buf, [hex2], True, blue, 2)

        # ── 6-pointed star (two counter-rotating triangles) ───────────────
        star_r = int(r * 0.62)
        for tri_rot, col in [(0, cc), (math.pi / 3, gold)]:
            tri = np.array([[int(cx + star_r * math.cos(t * 0.028 + tri_rot + i * 2*math.pi/3)),
                             int(cy + star_r * math.sin(t * 0.028 + tri_rot + i * 2*math.pi/3))]
                            for i in range(3)], dtype=np.int32)
            cv2.polylines(buf, [tri], True, col, 2)

        # ── Inner ring ────────────────────────────────────────────────────
        cv2.circle(buf, (cx, cy), int(r * 0.58), blue, 1)
        cv2.circle(buf, (cx, cy), int(r * 0.32), gold, 1)

        # ── Rune dots — outer (12, CW) ────────────────────────────────────
        for i in range(12):
            ang = t * 0.045 + i * (2*math.pi/12)
            cv2.circle(buf, (int(cx + r * 1.1 * math.cos(ang)),
                             int(cy + r * 1.1 * math.sin(ang))),
                       max(2, 3 + int(charge * 2)), blue, -1)

        # ── Rune dots — inner (6, CCW) ────────────────────────────────────
        for i in range(6):
            ang = -t * 0.065 + i * (math.pi/3)
            cv2.circle(buf, (int(cx + r * 0.6 * math.cos(ang)),
                             int(cy + r * 0.6 * math.sin(ang))),
                       max(2, 4 + int(charge * 3)), gold, -1)

        # ── Connecting spokes hex→centre (faint) ──────────────────────────
        spoke_c = tuple(max(0, ch - 150) for ch in cc)
        for pt in hex1:
            cv2.line(buf, (cx, cy), tuple(pt), spoke_c, 1)

        # ── Lens-flare core + cross rays ──────────────────────────────────
        core_r = max(4, int(r * 0.18 * (0.4 + charge * 0.6)))
        cv2.circle(buf, (cx, cy), core_r, white, -1)
        ray = int(r * 0.30 * charge)
        if ray > 6:
            for ang in [0, math.pi/4, math.pi/2, 3*math.pi/4,
                        math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
                thick = max(1, 3 if ang % (math.pi/2) < 0.01 else 1)
                cv2.line(buf, (cx, cy),
                         (int(cx + ray * math.cos(ang)),
                          int(cy + ray * math.sin(ang))),
                         white, thick)

    _batch_draw(overlay, _draw, blur_s=9, blur_w=27, w_s=0.85, w_w=0.45)

    if fx.state == 'CHARGING' and t % 8 == 0:
        psys.orbit(cx, cy, r, c, count=6, lifetime=65, speed_ang=0.07, size=3)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.ring_expand(cx, cy, r,      c,             count=200, lifetime=70, size=6)
        psys.ring_expand(cx, cy, r // 2, (255,255,255), count=100, lifetime=50, size=4)
        psys.burst(cx, cy, (255,255,255), count=120, speed=8, lifetime=55, size=5, gravity=0.03)


def _render_ignition_pulse(fx, overlay, psys, c, alpha):
    """
    Demon Slayer-style plasma flame triangle.
    Each edge is a 4-layer plasma column; vertices are miniature suns.
    """
    pts    = fx.trail_pts
    t      = fx.t
    cx, cy = fx.center
    charge = fx.charge_frac
    tri_s  = None

    fire_c  = config.ELEMENT_COLORS['FIRE']
    amber_c = (0, 150, 255)
    deep_r  = (0, 10, 180)
    white_y = (80, 220, 255)

    if len(pts) >= 3:
        arr    = np.asarray(pts, dtype=np.float64)
        idx    = [0, len(pts) // 2, len(pts) - 1]
        tri    = arr[idx].astype(np.float64)
        expand = 1.0 + 0.07 * math.sin(t * 0.15) + charge * 0.28
        tri_s  = ((tri - [cx, cy]) * expand + [cx, cy]).astype(np.int32)

        def _draw(buf):
            # ── 4-layer plasma edges ──────────────────────────────────────
            for col, thick in [(deep_r, 18), (fire_c, 11), (amber_c, 6), (white_y, 3)]:
                lc = tuple(int(ch * alpha) for ch in col)
                cv2.polylines(buf, [tri_s], True, lc, thick)

            # ── Vertex "sun" halos ────────────────────────────────────────
            sun_r = max(8, int(16 + charge * 14))
            for pt in tri_s:
                pt_t = tuple(pt)
                cv2.circle(buf, pt_t, sun_r + 8, tuple(int(ch * alpha) for ch in deep_r),  -1)
                cv2.circle(buf, pt_t, sun_r,     tuple(int(ch * alpha) for ch in fire_c),   -1)
                cv2.circle(buf, pt_t, sun_r // 2, tuple(int(ch * alpha) for ch in white_y), -1)

            # ── Interior ambient glow (triangle center) ───────────────────
            cv2.circle(buf, (cx, cy), max(5, int(fx.radius * 0.35 * charge)),
                       tuple(int(ch * alpha * 0.25) for ch in fire_c), -1)

        _batch_draw(overlay, _draw, blur_s=11, blur_w=35, w_s=0.90, w_w=0.50)

        # Screen-wide orange ambient during charge
        if charge > 0.3:
            _screen_tint(overlay, fire_c, strength=alpha * charge * 0.08)

    # Embers rising upward
    if fx.state == 'CHARGING' and t % 2 == 0:
        psys.burst(cx, cy, fire_c, count=5, speed=4, lifetime=24, size=3,
                   gravity=-0.16, spread=math.pi * 0.5)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        burst_pts = tri_s.tolist() if tri_s is not None else [[cx, cy]]
        for pt in burst_pts:
            psys.burst(int(pt[0]), int(pt[1]), fire_c,  count=100, speed=14,
                       lifetime=60, size=7, gravity=0.08)
            psys.burst(int(pt[0]), int(pt[1]), amber_c, count=60,  speed=10,
                       lifetime=45, size=5, gravity=0.10)
        psys.burst(cx, cy, fire_c, count=140, speed=7, lifetime=50, size=4,
                   gravity=-0.07, spread=2*math.pi)
        psys.lightning(cx, cy, cx+fx.radius, cy-fx.radius, amber_c)
        psys.lightning(cx, cy, cx-fx.radius, cy-fx.radius, amber_c)


def _render_arcane_construct(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    r, t   = fx.radius, fx.t
    charge = fx.charge_frac

    ang  = t * (0.04 + charge * 0.03)
    half = r + int(charge * r * 0.3)

    outer = np.array([
        [cx + half * math.cos(ang + i * math.pi / 2),
         cy + half * math.sin(ang + i * math.pi / 2)]
        for i in range(4)
    ])
    inner = np.array([
        [cx + (half * 0.5) * math.cos(-ang + 0.3 + i * math.pi / 2),
         cy + (half * 0.5) * math.sin(-ang + 0.3 + i * math.pi / 2)]
        for i in range(4)
    ])
    _glow_polygon(overlay, outer, c, thickness=2, blur=21)
    _glow_polygon(overlay, inner, c, thickness=1, blur=13)
    for i in range(4):
        _glow_line(overlay, tuple(outer[i].astype(int)),
                   tuple(inner[i].astype(int)), c, thickness=1, blur=9)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        for pt in outer:
            psys.burst(int(pt[0]), int(pt[1]), c, count=40, speed=7,
                       lifetime=45, size=5, gravity=0.05)


def _render_void_collapse(fx, overlay, psys, c, alpha):
    """
    Cosmic event horizon — JJK-inspired black void with bright violet ring edge.
    Darkens the frame OUTSIDE the void (light-absorption effect).
    """
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    H, W   = overlay.shape[:2]

    void_r   = max(8, int(fx.radius * (0.25 + charge * 0.35)))
    pulse_r  = max(10, int(fx.radius * (1.0 + 0.12 * math.sin(t * 0.18))))
    bright_c = tuple(int(ch * alpha) for ch in c)
    dim_c    = tuple(int(ch * alpha * 0.4) for ch in c)

    # ── Frame darkening outside the void (absorbs light) ─────────────────
    if charge > 0.2:
        Y, X   = np.ogrid[:H, :W]
        dist   = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
        mask   = np.clip((dist - pulse_r) / (pulse_r * 1.5), 0, 1)
        dark   = alpha * charge * 0.45 * mask
        overlay[:] = np.clip(
            overlay.astype(np.float32) * (1 - dark[:, :, None]), 0, 255
        ).astype(np.uint8)

    def _draw(buf):
        # ── Spiral arms (3 rotating bands) ───────────────────────────────
        for arm in range(3):
            base_ang = t * 0.055 + arm * (2*math.pi/3)
            for step in range(20):
                frac = step / 20
                ang  = base_ang + frac * math.pi * 1.5
                dist_frac = 0.35 + frac * 0.65
                sx   = int(cx + fx.radius * dist_frac * math.cos(ang))
                sy   = int(cy + fx.radius * dist_frac * math.sin(ang))
                sr   = max(2, int(5 * (1 - frac)))
                col  = tuple(int(ch * alpha * (1 - frac * 0.5)) for ch in c)
                cv2.circle(buf, (sx, sy), sr, col, -1)

        # ── Event-horizon ring (very bright) ──────────────────────────────
        cv2.circle(buf, (cx, cy), pulse_r, bright_c, 5)
        cv2.circle(buf, (cx, cy), max(4, pulse_r - 14), dim_c, 2)

        # ── Black void core ───────────────────────────────────────────────
        cv2.circle(buf, (cx, cy), void_r, (0, 0, 0), -1)

        # ── Debris shards orbiting ────────────────────────────────────────
        for i in range(8):
            ang  = t * 0.07 + i * (math.pi / 4)
            dist = fx.radius * 0.75
            sx   = int(cx + dist * math.cos(ang))
            sy   = int(cy + dist * math.sin(ang))
            ex   = int(cx + dist * 0.55 * math.cos(ang + 0.5))
            ey   = int(cy + dist * 0.55 * math.sin(ang + 0.5))
            cv2.line(buf, (sx, sy), (ex, ey), bright_c, 2)

    _batch_draw(overlay, _draw, blur_s=9, blur_w=33, w_s=0.85, w_w=0.55)

    if t % 2 == 0:
        psys.vortex_inward(cx, cy, c, count=14, radius=fx.radius + 40,
                           lifetime=45, size=4.0)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.vortex_inward(cx, cy, c, count=260, radius=fx.radius * 2.5,
                           lifetime=70, size=5)
        psys.burst(cx, cy, (40, 0, 80), count=120, speed=5, lifetime=60,
                   size=7, gravity=0.0)
        psys.ring_expand(cx, cy, fx.radius, c, count=150, lifetime=55, size=6)


def _render_reality_slice(fx, overlay, psys, c, alpha):
    """
    Dimensional cut — Bleach/Dragon Ball-style iridescent energy blade.
    5 colour layers from black-void outer to white-hot core.
    Ghost afterimage + perpendicular lightning arcs.
    """
    pts = fx.trail_pts
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.float64)
    p1  = tuple(arr[0].astype(int))
    p2  = tuple(arr[-1].astype(int))
    mx  = (p1[0] + p2[0]) // 2
    my  = (p1[1] + p2[1]) // 2

    dx     = p2[0] - p1[0]; dy = p2[1] - p1[1]
    length = max(math.sqrt(dx**2 + dy**2), 1)
    nx, ny = -dy / length, dx / length   # perpendicular

    def _draw(buf):
        # ── 5-layer blade (narrow void → white core) ──────────────────────
        for col, thick in [
            ((0, 0, 0),       24),   # black void gap
            ((80, 0, 120),    18),   # deep violet
            ((255, 30, 80),   12),   # electric blue
            ((255, 200, 100),  6),   # cyan-white
            ((255, 255, 255),  2),   # white core
        ]:
            lc = tuple(int(ch * alpha) for ch in col)
            cv2.line(buf, p1, p2, lc, thick)

        # ── Ghost afterimage (offset 18px perpendicular) ──────────────────
        for off in [18, -18, 32, -32]:
            p1g = (int(p1[0] + nx*off), int(p1[1] + ny*off))
            p2g = (int(p2[0] + nx*off), int(p2[1] + ny*off))
            gc  = tuple(int(ch * alpha * 0.28) for ch in c)
            cv2.line(buf, p1g, p2g, gc, 3)

        # ── Perpendicular impact flares at 5 points along slash ───────────
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            lx = int(p1[0] + (p2[0]-p1[0]) * frac)
            ly = int(p1[1] + (p2[1]-p1[1]) * frac)
            fl = int(20 + 30 * math.sin(fx.t * 0.3 + frac * math.pi))
            cv2.line(buf, (lx - int(nx*fl), ly - int(ny*fl)),
                          (lx + int(nx*fl), ly + int(ny*fl)),
                     tuple(int(ch * alpha * 0.55) for ch in (255,255,255)), 2)

        # ── Impact burst at midpoint ──────────────────────────────────────
        cv2.circle(buf, (mx, my), 12, tuple(int(ch*alpha) for ch in (255,255,255)), -1)

    _batch_draw(overlay, _draw, blur_s=7, blur_w=23, w_s=0.9, w_w=0.45)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.beam(p1[0], p1[1], p2[0], p2[1], c, count=200, spread=25, lifetime=50, size=5)
        for frac in [0.2, 0.5, 0.8]:
            lx = int(p1[0] + (p2[0]-p1[0]) * frac)
            ly = int(p1[1] + (p2[1]-p1[1]) * frac)
            psys.lightning(lx, ly-50, lx, ly+50, (255,255,255))
        psys.lightning(p1[0], p1[1], p2[0], p2[1], (180,180,255), segments=20)
        psys.burst(mx, my, (255,255,255), count=120, speed=10, lifetime=40, size=6, gravity=0.03)


def _render_domain_of_aether(fx, overlay, psys, c, alpha):
    """
    Domain Expansion — JJK-inspired cosmic domain.
    Full-screen golden sacred-geometry mandala + spiral arms + portal rings.
    """
    h, w = overlay.shape[:2]
    t    = fx.t
    charge = fx.charge_frac
    cx, cy = w // 2, h // 2

    # ── Deepening golden screen tint (pulses like a heartbeat) ───────────
    beat = 0.15 + 0.08 * abs(math.sin(t * 0.08))
    _screen_tint(overlay, config.GOLD, strength=alpha * beat)

    def _draw(buf):
        cc  = tuple(int(ch * alpha) for ch in c)
        dim = tuple(int(ch * alpha * 0.4) for ch in c)

        # ── Large outer mandala hexagon ────────────────────────────────────
        outer_r = int(min(w, h) * 0.42)
        hex_big = np.array([[int(cx + outer_r * math.cos(t*0.008 + i*math.pi/3)),
                              int(cy + outer_r * math.sin(t*0.008 + i*math.pi/3))]
                             for i in range(6)], dtype=np.int32)
        cv2.polylines(buf, [hex_big], True, cc, 2)

        # ── Middle hexagon (CCW) ───────────────────────────────────────────
        mid_r = int(min(w, h) * 0.28)
        hex_mid = np.array([[int(cx + mid_r * math.cos(-t*0.012 + math.pi/6 + i*math.pi/3)),
                              int(cy + mid_r * math.sin(-t*0.012 + math.pi/6 + i*math.pi/3))]
                             for i in range(6)], dtype=np.int32)
        cv2.polylines(buf, [hex_mid], True, cc, 2)

        # ── 12 radial spokes from center to outer hex ─────────────────────
        for i in range(12):
            ang = t * 0.01 + i * math.pi / 6
            ex  = int(cx + outer_r * math.cos(ang))
            ey  = int(cy + outer_r * math.sin(ang))
            cv2.line(buf, (cx, cy), (ex, ey), dim, 1)

        # ── 3 rotating spiral arms ────────────────────────────────────────
        for arm in range(3):
            base = t * 0.035 + arm * (2*math.pi/3)
            prev = (cx, cy)
            for step in range(30):
                frac  = step / 30
                ang   = base + frac * math.pi * 2.5
                dist  = outer_r * frac
                px    = int(cx + dist * math.cos(ang))
                py    = int(cy + dist * math.sin(ang))
                sc    = tuple(int(ch * alpha * (0.8 - frac * 0.6)) for ch in c)
                cv2.line(buf, prev, (px, py), sc, max(1, int(3*(1-frac))))
                prev  = (px, py)

        # ── Expanding portal rings (3-phase) ──────────────────────────────
        for offset in [0, 30, 60]:
            rt  = (t + offset) % 90
            rr  = int((rt / 90) * max(w, h) * 1.1)
            ra  = alpha * (1.0 - rt / 90)
            rc  = tuple(int(ch * ra) for ch in c)
            if rr > 2:
                cv2.circle(buf, (cx, cy), rr, rc, 2)

        # ── Golden core star ──────────────────────────────────────────────
        core_r = int(60 + 20 * math.sin(t * 0.1))
        for i in range(8):
            ang = t * 0.04 + i * math.pi / 4
            sx  = int(cx + core_r * math.cos(ang))
            sy  = int(cy + core_r * math.sin(ang))
            cv2.line(buf, (cx, cy), (sx, sy), cc, max(1, 3-i//3))
        cv2.circle(buf, (cx, cy), 20, tuple(int(ch*alpha) for ch in (255,255,255)), -1)

    _batch_draw(overlay, _draw, blur_s=9, blur_w=31, w_s=0.80, w_w=0.45)

    if t % 3 == 0:
        psys.screen_storm(w, h, c, count=20, lifetime=65)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.screen_storm(w, h, c, count=350, lifetime=100)
        psys.burst(cx, cy, config.GOLD, count=220, speed=16, lifetime=80,
                   size=7, gravity=0.0, spread=2*math.pi)


def _render_maelstrom(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    t      = fx.t
    if t % 4 == 0:
        psys.orbit(cx, cy, fx.radius, c, count=8, lifetime=65,
                   speed_ang=0.10 + fx.charge_frac * 0.08, size=4.5)
    _glow_circle(overlay, (cx, cy), fx.radius, c, thickness=3, blur=23)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.arc_shower(cx, cy, fx.radius, c, count=160, lifetime=60, size=5)


def _render_radiant_pulse(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    for ring in range(4):
        r  = max(5, int(fx.radius * (0.4 + ring * 0.2) * (1 + charge * 0.3)))
        a  = alpha * (1.0 - ring * 0.22)
        rc = tuple(int(ch * a) for ch in c)
        _glow_circle(overlay, (cx, cy), r, rc, thickness=1, blur=13)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        for i in range(8):
            ang = i * math.pi / 4
            x2  = int(cx + fx.radius * 2.5 * math.cos(ang))
            y2  = int(cy + fx.radius * 2.5 * math.sin(ang))
            psys.beam(cx, cy, x2, y2, c, count=35, spread=6, lifetime=45, size=4)


# ── Combo spell renderers ─────────────────────────────────────────────────────

def _render_exploding_ward(fx, overlay, psys, c, alpha):
    _render_arcane_ward(fx, overlay, psys, c, alpha)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        cx, cy = fx.center
        psys.ring_expand(cx, cy, fx.radius,      c,           count=180, lifetime=60, size=6)
        psys.ring_expand(cx, cy, fx.radius // 2, config.GOLD, count=90,  lifetime=50, size=5)
        psys.burst(cx, cy, (255,255,255), count=50, speed=16, lifetime=25, size=4, gravity=0.0)


def _render_void_beam(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    t      = fx.t
    h, w   = overlay.shape[:2]
    ang    = -math.pi / 4 + math.sin(t * 0.05) * 0.1
    blen   = int(max(w, h) * 0.9)
    bx     = int(cx + blen * math.cos(ang))
    by     = int(cy + blen * math.sin(ang))
    for width in [10, 6, 3]:
        _glow_line(overlay, (cx, cy), (bx, by), c, thickness=width, blur=25)
    _glow_line(overlay, (cx, cy), (bx, by), (220, 220, 255), thickness=1, blur=7)
    if t % 4 == 0:
        psys.beam(cx, cy, bx, by, c, count=18, spread=7, lifetime=28, size=3)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.beam(cx, cy, bx, by, c, count=220, spread=22, lifetime=55, size=6)
        psys.lightning(cx, cy, bx, by, (255, 255, 255), segments=20)


def _render_barrier_field(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    max_r  = int(fx.radius * (1 + charge * 1.5))
    for ring in range(3):
        rr     = max_r - ring * 14
        if rr < 5:
            continue
        ring_a = alpha * (1.0 - ring * 0.3)
        rc     = tuple(int(ch * ring_a) for ch in c)
        cv2.ellipse(overlay, (cx, cy), (rr, rr), 0, 180, 360, rc, 2)
    if t % 6 == 0:
        psys.arc_shower(cx, cy, max_r, c, count=14, a_start=math.pi,
                        a_end=2 * math.pi, lifetime=45, size=4)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.arc_shower(cx, cy, max_r,      c,             count=220, a_start=math.pi, a_end=2*math.pi, lifetime=65, size=6)
        psys.arc_shower(cx, cy, max_r // 2, config.NEON_BLUE, count=90, a_start=math.pi, a_end=2*math.pi, lifetime=50, size=4)


def _render_nova_burst(fx, overlay, psys, c, alpha):
    _render_arcane_ward(fx, overlay, psys, c, alpha)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        cx, cy = fx.center
        psys.burst(cx, cy, c, count=220, speed=12, lifetime=60, size=6,
                   gravity=0.04, spread=2 * math.pi)
        psys.ring_expand(cx, cy, fx.radius, config.GOLD, count=140, lifetime=55, size=5)
        psys.lightning(cx - fx.radius, cy, cx + fx.radius, cy, (255, 255, 180))
        psys.lightning(cx, cy - fx.radius, cx, cy + fx.radius, (255, 255, 180))


def _render_convergence(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    t      = fx.t
    h, w   = overlay.shape[:2]
    if t % 6 == 0:
        ang  = np.random.uniform(0, 2 * math.pi)
        dist = max(w, h) * 0.6
        ox   = int(cx + dist * math.cos(ang))
        oy   = int(cy + dist * math.sin(ang))
        _glow_line(overlay, (ox, oy), (cx, cy), c, thickness=2, blur=15)
    _glow_circle(overlay, (cx, cy), max(5, int(20 + t * 0.5)), c, thickness=2, blur=15)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        for i in range(8):
            ang  = i * math.pi / 4
            dist = max(w, h) * 0.5
            ox   = int(cx + dist * math.cos(ang))
            oy   = int(cy + dist * math.sin(ang))
            psys.beam(ox, oy, cx, cy, c, count=45, spread=12, lifetime=45, size=5)
        psys.burst(cx, cy, c, count=140, speed=9, lifetime=50, size=6, gravity=0.05)


def _render_prism_lance(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    w      = fx.w
    light_c = config.ELEMENT_COLORS['LIGHT']
    for off, col in [(0, c), (10, (255, 255, 200)), (-10, config.NEON_BLUE)]:
        _glow_line(overlay, (cx, cy), (w, cy + off), col, thickness=4, blur=19)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.beam(cx, cy, w, cy, light_c, count=180, spread=18, lifetime=50, size=5)
        psys.lightning(cx, cy, w, cy, (255, 255, 255))


# ══════════════════════════════════════════════════════════════════════════════
# ── Naruto jutsu renderers ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_CHAKRA_BLUE  = (255, 180, 40)     # BGR: blue chakra
_CHAKRA_WHITE = (255, 240, 200)


def _render_kage_bunshin(fx, overlay, psys, c, alpha):
    """Blue chakra rings + four ghost clone silhouettes spreading outward."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    h, w   = overlay.shape[:2]

    # Screen blue-tint aura building with charge
    _screen_tint(overlay, _CHAKRA_BLUE, strength=alpha * charge * 0.08)

    # Concentric charging rings
    for ring in range(4):
        pulse = 1.0 + 0.10 * math.sin(t * 0.18 + ring * 0.9)
        rr    = max(5, int(fx.radius * (0.35 + ring * 0.30) * pulse))
        rc    = tuple(int(ch * alpha * (1.0 - ring * 0.20)) for ch in _CHAKRA_BLUE)
        _glow_circle(overlay, (cx, cy), rr, rc, 2, blur=21)

    # Four ghost-clone silhouettes (simple cross figures at 4 compass points)
    n_clones = min(4, 1 + int(charge * 4))
    for i in range(n_clones):
        ang   = i * (math.pi / 2) + t * 0.04
        dist  = int(fx.radius * 0.85)
        gx    = int(cx + dist * math.cos(ang))
        gy    = int(cy + dist * math.sin(ang))
        g_a   = alpha * (0.28 + 0.18 * math.sin(t * 0.14 + i))
        gc    = tuple(int(ch * g_a) for ch in _CHAKRA_BLUE)
        # Head
        cv2.circle(overlay, (gx, gy - 12), 8, gc, 1)
        # Body & arms
        cv2.line(overlay, (gx, gy - 4), (gx, gy + 14), gc, 2)
        cv2.line(overlay, (gx - 10, gy + 4), (gx + 10, gy + 4), gc, 2)
        # Legs
        cv2.line(overlay, (gx, gy + 14), (gx - 7, gy + 26), gc, 2)
        cv2.line(overlay, (gx, gy + 14), (gx + 7, gy + 26), gc, 2)

    if t % 3 == 0:
        psys.vortex_inward(cx, cy, _CHAKRA_BLUE, count=10, radius=fx.radius + 35,
                           lifetime=38, size=3.5)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        # Clone explosion — burst from each ghost position
        for i in range(4):
            ang  = i * math.pi / 2
            dist = int(fx.radius * 0.85)
            gx   = int(cx + dist * math.cos(ang))
            gy   = int(cy + dist * math.sin(ang))
            psys.burst(gx, gy, _CHAKRA_BLUE, count=70, speed=9, lifetime=50, size=5, gravity=0.04)
        psys.burst(cx, cy, _CHAKRA_WHITE, count=180, speed=13, lifetime=55, size=6, gravity=0.0)
        psys.screen_storm(w, h, _CHAKRA_BLUE, count=120, lifetime=65)


def _render_chidori(fx, overlay, psys, c, alpha):
    """Blue lightning converging into a crackling concentrated core."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    h, w   = overlay.shape[:2]

    # Converging lightning bolts from screen edges
    n_bolts = 6 + int(charge * 8)
    for i in range(n_bolts):
        ang  = i * (2 * math.pi / n_bolts) + t * 0.04
        dist = int(max(w, h) * 0.52)
        ox   = int(cx + dist * math.cos(ang))
        oy   = int(cy + dist * math.sin(ang))
        b_a  = alpha * (0.35 + 0.30 * math.sin(t * 0.28 + i))
        bc   = tuple(int(ch * b_a) for ch in _CHAKRA_BLUE)
        _glow_line(overlay, (ox, oy), (cx, cy), bc, 2, blur=13)

    # Concentrated crackling core
    core_r = max(5, int(20 + charge * 35))
    _filled_glow(overlay, (cx, cy), core_r,      _CHAKRA_BLUE,  blur=29)
    _filled_glow(overlay, (cx, cy), core_r // 2, _CHAKRA_WHITE, blur=17)
    _filled_glow(overlay, (cx, cy), core_r // 4, (255,255,255), blur=9)

    if t % 2 == 0:
        psys.burst(cx, cy, _CHAKRA_BLUE, count=10, speed=7, lifetime=16, size=3, gravity=0.0)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        # Full lightning storm emanating outward
        for i in range(10):
            ang  = i * math.pi / 5
            dist = int(max(w, h) * 0.65)
            ox   = int(cx + dist * math.cos(ang))
            oy   = int(cy + dist * math.sin(ang))
            psys.lightning(cx, cy, ox, oy, _CHAKRA_WHITE, segments=22)
        psys.burst(cx, cy, _CHAKRA_BLUE,  count=220, speed=11, lifetime=55, size=6, gravity=0.0)
        psys.burst(cx, cy, _CHAKRA_WHITE, count=110, speed=16, lifetime=30, size=7, gravity=0.0)


def _render_rasengan(fx, overlay, psys, c, alpha):
    """Spinning concentric sphere of compressed chakra → massive explosion."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac

    r_base = max(10, int(28 + charge * 45))

    # Outer equator ring — spinning CW
    for i in range(18):
        ang = i * (2 * math.pi / 18) + t * 0.11
        dx  = int(cx + r_base * math.cos(ang))
        dy  = int(cy + r_base * 0.55 * math.sin(ang))   # squish = 3D perspective
        cv2.circle(overlay, (dx, dy), max(2, 3), tuple(int(ch*alpha) for ch in _CHAKRA_BLUE), -1)

    # Inner ring — spinning CCW
    r_in = max(5, r_base // 2)
    for i in range(12):
        ang = i * (2 * math.pi / 12) - t * 0.16
        dx  = int(cx + r_in * math.cos(ang))
        dy  = int(cy + r_in * 0.55 * math.sin(ang))
        cv2.circle(overlay, (dx, dy), 2, tuple(int(ch*alpha) for ch in _CHAKRA_WHITE), -1)

    # Glowing core
    _filled_glow(overlay, (cx, cy), r_base // 3, _CHAKRA_BLUE,  blur=23)
    _filled_glow(overlay, (cx, cy), r_base // 6, _CHAKRA_WHITE, blur=13)

    if t % 2 == 0:
        psys.orbit(cx, cy, r_base, _CHAKRA_BLUE, count=5, lifetime=22, speed_ang=0.18, size=3.5)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.burst(cx, cy, _CHAKRA_BLUE,  count=240, speed=14, lifetime=60, size=7, gravity=0.0, spread=2*math.pi)
        psys.ring_expand(cx, cy, r_base, _CHAKRA_WHITE, count=140, lifetime=50, size=6)
        psys.burst(cx, cy, _CHAKRA_WHITE, count=120, speed=20, lifetime=35, size=5, gravity=0.0)


def _render_katon_goka(fx, overlay, psys, c, alpha):
    """Expanding fireball that erupts into a full-screen fire wall."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    h, w   = overlay.shape[:2]

    fire_c  = config.ELEMENT_COLORS['FIRE']
    amber_c = (0, 140, 255)

    # Growing fireball sphere
    r = max(10, int(fx.radius * (0.25 + charge * 1.0)))
    _filled_glow(overlay, (cx, cy), r,           (0, 30, 180), blur=41)
    _filled_glow(overlay, (cx, cy), int(r * 0.7), fire_c,      blur=27)
    _filled_glow(overlay, (cx, cy), int(r * 0.4), amber_c,     blur=17)
    _filled_glow(overlay, (cx, cy), int(r * 0.2), (200,230,255), blur=9)
    _glow_circle(overlay, (cx, cy), r + 12, fire_c, thickness=5, blur=27)

    # Fire particles floating upward
    if t % 2 == 0:
        for _ in range(4):
            ox = cx + int(np.random.uniform(-r, r))
            oy = cy + int(np.random.uniform(-r // 2, r // 2))
            psys.burst(ox, oy, fire_c, count=4, speed=3,
                       lifetime=28, size=5, gravity=-0.10, spread=0.7)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.burst(cx, cy, fire_c, count=300, speed=14, lifetime=70,
                   size=8, gravity=0.04, spread=2*math.pi)
        psys.burst(cx, cy, amber_c, count=160, speed=10, lifetime=55, size=6, gravity=0.06)
        # Fire wall: particles across full screen width
        for x in range(0, w, 55):
            psys.burst(x, cy, fire_c, count=18, speed=6,
                       lifetime=45, size=5, gravity=-0.08)


def _render_kagemane(fx, overlay, psys, c, alpha):
    """Dark shadow tendrils creeping outward from the caster's position."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    h, w   = overlay.shape[:2]

    shadow_c = (0, 210, 120)    # dark purple (BGR)

    # Tendrils extending outward
    n_tendrils = 8
    for i in range(n_tendrils):
        ang    = i * (2 * math.pi / n_tendrils) + math.sin(t * 0.04 + i) * 0.12
        length = int(max(w, h) * 0.40 * charge)
        ex     = int(cx + length * math.cos(ang))
        ey     = int(cy + length * math.sin(ang))
        lc     = tuple(int(ch * alpha) for ch in c)
        _glow_line(overlay, (cx, cy), (ex, ey), lc, thickness=5, blur=17)
        # Branching feeler
        if charge > 0.35:
            mid_x = (cx + ex) // 2 + int(np.random.uniform(-18, 18))
            mid_y = (cy + ey) // 2 + int(np.random.uniform(-18, 18))
            lc2   = tuple(int(ch * alpha * 0.55) for ch in c)
            _glow_line(overlay, (cx, cy), (mid_x, mid_y), lc2, thickness=2, blur=9)

    # Dark void core
    cv2.circle(overlay, (cx, cy), max(5, int(22 * charge)), (0, 0, 0), -1)
    _glow_circle(overlay, (cx, cy), max(5, int(28 * charge)), c, thickness=3, blur=23)

    if t % 4 == 0:
        psys.vortex_inward(cx, cy, c, count=7, radius=110, lifetime=45, size=3.5)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        for i in range(n_tendrils):
            ang  = i * (2 * math.pi / n_tendrils)
            dist = int(max(w, h) * 0.48)
            ex   = int(cx + dist * math.cos(ang))
            ey   = int(cy + dist * math.sin(ang))
            psys.beam(cx, cy, ex, ey, c, count=50, spread=10, lifetime=45, size=5)
        psys.burst(cx, cy, c, count=140, speed=6, lifetime=55, size=6, gravity=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# ── One Piece technique renderers ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_LUFFY_RED  = (30,  30, 220)    # deep red
_LUFFY_ORG  = (0,  100, 255)    # orange
_LUFFY_WHT  = (200, 220, 255)


def _render_gomu_pistol(fx, overlay, psys, c, alpha):
    """Stretching elastic punch beam with shockwave rings at impact."""
    pts = fx.trail_pts
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.float64)
    p1  = tuple(arr[0].astype(int))
    p2  = tuple(arr[-1].astype(int))
    t   = fx.t
    charge = fx.charge_frac

    # Stretch effect — tip oscillates further
    stretch = 1.0 + 0.18 * math.sin(t * 0.32) * charge
    ex = int(p1[0] + (p2[0] - p1[0]) * stretch)
    ey = int(p1[1] + (p2[1] - p1[1]) * stretch)

    # Thick rubber arm — three colour layers
    _glow_line(overlay, p1, (ex, ey), _LUFFY_RED, 14, 21)
    _glow_line(overlay, p1, (ex, ey), _LUFFY_ORG,  8, 13)
    _glow_line(overlay, p1, (ex, ey), _LUFFY_WHT,  3,  7)

    # Big impact glow at tip
    tip_r = max(8, int(28 + charge * 20))
    _filled_glow(overlay, (ex, ey), tip_r,           _LUFFY_ORG, blur=25)
    _filled_glow(overlay, (ex, ey), tip_r // 2,      _LUFFY_WHT, blur=13)
    _glow_circle(overlay, (ex, ey), tip_r + 8, _LUFFY_ORG, thickness=3, blur=15)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.burst(ex, ey, _LUFFY_ORG, count=220, speed=13, lifetime=55, size=7, gravity=0.05)
        psys.burst(ex, ey, _LUFFY_WHT, count=100, speed=19, lifetime=28, size=6, gravity=0.0)
        for r in [20, 50, 90]:
            psys.ring_expand(ex, ey, r, _LUFFY_ORG, count=50, lifetime=32, size=4)
        psys.beam(p1[0], p1[1], ex, ey, _LUFFY_ORG, count=90, spread=12, lifetime=35, size=4)


def _render_gomu_cannon(fx, overlay, psys, c, alpha):
    """Giant compressed rubber ball — grows then erupts."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac

    r     = max(12, int(fx.radius * (0.22 + charge * 1.0)))
    pulse = 1.0 + 0.09 * math.sin(t * 0.22)

    _filled_glow(overlay, (cx, cy), int(r * pulse * 0.9), _LUFFY_RED, blur=41)
    _filled_glow(overlay, (cx, cy), int(r * 0.68),        _LUFFY_ORG, blur=27)
    _filled_glow(overlay, (cx, cy), int(r * 0.38),        _LUFFY_WHT, blur=15)
    _glow_circle(overlay, (cx, cy), int(r * pulse) + 12, _LUFFY_ORG, thickness=4, blur=25)

    if t % 3 == 0:
        psys.orbit(cx, cy, r + 8, _LUFFY_ORG, count=10, lifetime=32, speed_ang=0.14, size=4.5)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.burst(cx, cy, _LUFFY_ORG, count=280, speed=15, lifetime=65, size=8, gravity=0.03, spread=2*math.pi)
        psys.ring_expand(cx, cy, r,     _LUFFY_WHT, count=220, lifetime=60, size=6)
        psys.ring_expand(cx, cy, r * 2, _LUFFY_ORG, count=120, lifetime=45, size=4)
        psys.burst(cx, cy, _LUFFY_WHT, count=110, speed=22, lifetime=38, size=7, gravity=0.0)


def _render_gear_second(fx, overlay, psys, c, alpha):
    """Red steam eruption — rising heat aura across entire screen."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac
    h, w   = overlay.shape[:2]

    steam_c = (200, 200, 210)

    # Red screen heat aura builds with charge
    _screen_tint(overlay, _LUFFY_RED, strength=alpha * charge * 0.14)

    # Expanding steam rings
    for ring in range(5):
        ring_t = (t + ring * 12) % 65
        r      = int((ring_t / 65) * max(w, h) * 0.65)
        ring_a = alpha * (1.0 - ring_t / 65)
        rc     = tuple(int(ch * ring_a) for ch in _LUFFY_RED)
        if r > 5:
            _glow_circle(overlay, (cx, cy), r, rc, thickness=3, blur=23)

    # Steam particles floating upward
    if t % 2 == 0:
        for _ in range(6):
            px = int(np.random.uniform(cx - fx.radius, cx + fx.radius))
            py = int(np.random.uniform(cy - fx.radius // 2, cy + fx.radius // 2))
            psys.burst(px, py, steam_c, count=3, speed=2,
                       lifetime=45, size=7, gravity=-0.14)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.screen_storm(w, h, _LUFFY_RED, count=300, lifetime=90)
        psys.burst(cx, cy, _LUFFY_RED, count=220, speed=11, lifetime=65, size=7, gravity=-0.06)
        psys.burst(cx, cy, steam_c,    count=90,  speed=8,  lifetime=55, size=8, gravity=-0.18)


def _render_gear_third(fx, overlay, psys, c, alpha):
    """Enormous inflating fist sphere — biggest hit in One Piece."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac

    # ENORMOUS inflating ball
    r     = max(20, int(fx.radius * (0.55 + charge * 2.8)))
    pulse = 1.0 + 0.07 * math.sin(t * 0.10)

    _filled_glow(overlay, (cx, cy), int(r * pulse * 0.92), (0, 15, 160), blur=51)
    _filled_glow(overlay, (cx, cy), int(r * 0.68),         _LUFFY_RED,   blur=35)
    _filled_glow(overlay, (cx, cy), int(r * 0.38),         _LUFFY_ORG,   blur=21)
    _filled_glow(overlay, (cx, cy), int(r * 0.16),         _LUFFY_WHT,   blur=11)
    _glow_circle(overlay, (cx, cy), int(r * pulse), (180, 140, 255), thickness=5, blur=29)

    if t % 4 == 0:
        psys.ring_expand(cx, cy, r, _LUFFY_ORG, count=22, lifetime=28, size=6)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.burst(cx, cy, _LUFFY_ORG, count=360, speed=17, lifetime=75, size=9, gravity=0.06, spread=2*math.pi)
        psys.ring_expand(cx, cy, r,      _LUFFY_WHT, count=280, lifetime=70, size=7)
        psys.ring_expand(cx, cy, r * 2,  _LUFFY_ORG, count=160, lifetime=55, size=5)
        psys.burst(cx, cy, _LUFFY_WHT, count=140, speed=24, lifetime=45, size=8, gravity=0.0)


def _render_elephant_gatling(fx, overlay, psys, c, alpha):
    """Rotating barrage of multiple rubber-arm impacts."""
    cx, cy = fx.center
    t      = fx.t
    charge = fx.charge_frac

    arm_r  = max(30, int(fx.radius * (0.55 + charge * 0.55)))
    n_arms = 8

    # Spinning impact circles
    for i in range(n_arms):
        ang = i * (2 * math.pi / n_arms) + t * 0.09
        ax  = int(cx + arm_r * math.cos(ang))
        ay  = int(cy + arm_r * math.sin(ang))
        i_a = alpha * (0.65 + 0.35 * math.sin(t * 0.32 + i))
        ic  = tuple(int(ch * i_a) for ch in _LUFFY_ORG)
        cv2.circle(overlay, (ax, ay), max(6, int(14 * charge)), ic, -1)

    _glow_circle(overlay, (cx, cy), arm_r, _LUFFY_ORG, thickness=3, blur=23)

    if t % 2 == 0:
        ang  = np.random.uniform(0, 2 * math.pi)
        dist = int(np.random.uniform(0.4, 0.95) * arm_r)
        px   = int(cx + dist * math.cos(ang))
        py   = int(cy + dist * math.sin(ang))
        psys.burst(px, py, _LUFFY_ORG, count=12, speed=6, lifetime=22, size=5, gravity=0.05)

    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        for i in range(n_arms):
            ang = i * math.pi / 4
            for dist in [arm_r * 0.5, arm_r, arm_r * 1.6]:
                px = int(cx + int(dist) * math.cos(ang))
                py = int(cy + int(dist) * math.sin(ang))
                psys.burst(px, py, _LUFFY_ORG, count=45, speed=9, lifetime=45, size=6, gravity=0.06)
        psys.ring_expand(cx, cy, arm_r, _LUFFY_ORG, count=220, lifetime=60, size=6)


def _render_generic(fx, overlay, psys, c, alpha):
    cx, cy = fx.center
    pulse  = 1.0 + 0.07 * math.sin(fx.t * 0.14)
    r      = max(5, int(fx.radius * pulse))
    _glow_circle(overlay, (cx, cy), r, c, thickness=2, blur=17)
    if fx.state == 'RELEASE' and not fx._released_particles:
        fx._released_particles = True
        psys.burst(cx, cy, c, count=120, speed=7, lifetime=45, size=5, gravity=0.08)


# ── Renderer lookup table ─────────────────────────────────────────────────────

_RENDERERS = {
    # Aether Magic
    'Arcane Ward':       _render_arcane_ward,
    'Ignition Pulse':    _render_ignition_pulse,
    'Firestorm':         _render_ignition_pulse,
    'Arcane Construct':  _render_arcane_construct,
    'Void Collapse':     _render_void_collapse,
    'Reality Slice':     _render_reality_slice,
    'Flash Strike':      _render_reality_slice,
    'Torrent Slash':     _render_reality_slice,
    'Domain of Aether':  _render_domain_of_aether,
    'Maelstrom':         _render_maelstrom,
    'Radiant Pulse':     _render_radiant_pulse,
    # Combos
    'Exploding Ward':    _render_exploding_ward,
    'Void Beam':         _render_void_beam,
    'Barrier Field':     _render_barrier_field,
    'Nova Burst':        _render_nova_burst,
    'Convergence':       _render_convergence,
    'Prism Lance':       _render_prism_lance,
    # Naruto
    'Kage Bunshin':           _render_kage_bunshin,
    'Chidori':                _render_chidori,
    'Rasengan':               _render_rasengan,
    'Katon: Goka Mekkyaku':   _render_katon_goka,
    'Kagemane no Jutsu':      _render_kagemane,
    # One Piece
    'Gomu Gomu no Pistol':    _render_gomu_pistol,
    'Gomu Gomu no Cannon':    _render_gomu_cannon,
    'Gear Second':            _render_gear_second,
    'Gear Third: King Kong Gun': _render_gear_third,
    'Elephant Gatling':       _render_elephant_gatling,
}


# ── Spell System ──────────────────────────────────────────────────────────────

class SpellSystem:
    def __init__(self, frame_w, frame_h, theme=None):
        self.fw    = frame_w
        self.fh    = frame_h
        self.theme = theme
        self._class_cfg = None

        self._buffer:  list = []
        self.active:   SpellEffect | None = None
        self.history   = deque(maxlen=5)
        self.mastery   = {e: 0 for e in config.ELEMENT_LIST}
        self._combo    = deque(maxlen=2)
        self.overloading = False
        self._release_event: tuple | None = None

    # ── Element remapping ─────────────────────────────────────────────────────

    def _remap_element(self, element: str) -> str:
        # If a class has been set via set_class(), prefer that mapping
        if hasattr(self, '_class_cfg') and self._class_cfg is not None:
            allowed = getattr(self._class_cfg, 'allowed_elements', None)
            primary = getattr(self._class_cfg, 'primary_element', None)
            if allowed and element in allowed:
                return element
            return primary or element
        # Fallback to theme-based mapping
        if self.theme is None:
            return element
        if element in self.theme.allowed_elements:
            return element
        return self.theme.primary_element

    # ── Spell resolution ──────────────────────────────────────────────────────

    def _resolve_spell(self, shape: str, element: str) -> str:
        # Theme-specific map overrides default
        if self.theme and self.theme.id in _THEME_SPELL_MAPS:
            tmap = _THEME_SPELL_MAPS[self.theme.id]
            name = tmap.get((shape, element)) or tmap.get((shape, 'NEUTRAL'))
            if name:
                return name
        # Default map
        name = _SPELL_MAP.get((shape, element))
        if name is None:
            name = _SPELL_MAP.get((shape, 'NEUTRAL'), shape.title() + ' Cast')
        return name

    # ── Public API ────────────────────────────────────────────────────────────

    def on_shape(self, shape: str, element: str, trail_pts: list):
        element = self._remap_element(element)
        now     = time.time()
        self._buffer.append({'shape': shape, 'element': element, 't': now})
        self._buffer = [e for e in self._buffer
                        if now - e['t'] < config.SPELL_TIMEOUT]

        self._combo.append(shape)
        combo_key  = tuple(self._combo)
        spell_name = _COMBOS.get(combo_key) or self._resolve_spell(shape, element)

        print(f'[AETHER] Shape={shape:14s}  Element={element:8s}  → {spell_name}')

        if self.active is None or self.active.state == 'DECAY':
            self.active = SpellEffect(spell_name, element, trail_pts,
                                      self.fw, self.fh)

    def on_hand_sign(self, jutsu_name: str, element: str, trail_pts: list):
        """Called when a Naruto hand sign triggers a jutsu."""
        print(f'[AETHER] ✦ JUTSU: {jutsu_name}')
        if self.active is None or self.active.state == 'DECAY':
            self.active = SpellEffect(jutsu_name, element, trail_pts,
                                      self.fw, self.fh)
            self.active.trigger_release()
            self._do_release_record(jutsu_name, self.active.color, element)

    # ── Class switching API ───────────────────────────────────────────────
    def set_class(self, cfg):
        """Switch to a new class configuration at runtime.

        cfg: ClassConfig-like object with attributes `allowed_elements` and
        `primary_element`.
        """
        self._class_cfg = cfg
        # Reset active spell + combo buffer
        self.active = None
        self._buffer.clear()
        self._combo.clear()

    def on_op_move(self, move_name: str, center_xy: tuple):
        """Called when a One Piece physical move is detected."""
        print(f'[AETHER] ✦ OP MOVE: {move_name}')
        if self.active is None or self.active.state == 'DECAY':
            # Build a minimal trail centred on the hand position
            cx, cy = center_xy
            fake_trail = [(cx + dx, cy) for dx in range(-40, 41, 10)]
            self.active = SpellEffect(move_name, 'NEUTRAL', fake_trail,
                                      self.fw, self.fh)
            self.active.trigger_release()
            self._do_release_record(move_name, self.active.color, 'NEUTRAL')

    def on_release(self):
        if self.active and self.active.state == 'CHARGING':
            self._do_release()

    def update(self):
        self._release_event = None

        if self.active is None:
            self.overloading = False
            return

        self.active.update()

        if (self.active.state == 'CHARGING' and
                self.active.charge_duration >= config.OVERLOAD_SECS):
            self.overloading = True
            self._do_release()
        else:
            self.overloading = False

        if self.active.is_done():
            self.active = None

    def render(self, overlay, psys):
        if self.active:
            self.active.render(overlay, psys)

    def reset(self):
        self.active = None
        self._buffer.clear()
        self._combo.clear()
        self.overloading = False
        self._release_event = None

    # ── State queries ─────────────────────────────────────────────────────────

    @property
    def release_event(self):
        return self._release_event

    @property
    def charge_secs(self):
        if self.active and self.active.state == 'CHARGING':
            return self.active.charge_duration
        return 0.0

    @property
    def charge_frac(self):
        return min(1.0, self.charge_secs / config.OVERLOAD_SECS)

    @property
    def combo_progress(self):
        return list(self._combo)

    @property
    def is_power_spell(self):
        return self.active is not None and self.active.name in _POWER_SPELLS

    # ── Internals ─────────────────────────────────────────────────────────────

    def _do_release(self):
        if self.active is None:
            return
        self.active.trigger_release()
        self._do_release_record(self.active.name, self.active.color,
                                self.active.element)

    def _do_release_record(self, name, color, elem):
        self.history.append(name)
        if elem in self.mastery:
            self.mastery[elem] = min(100, self.mastery[elem] + 5)
        self._release_event = (name, color, elem)
        print(f'[AETHER] ✦ RELEASED: {name}  ({elem})')
