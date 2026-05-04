"""
Vectorised particle system.

All particle data lives in fixed-size numpy arrays; dead slots are recycled.
Each particle has:  position, velocity, color, lifetime, max_lifetime, size.
"""
import cv2
import numpy as np
import config


class ParticleSystem:
    def __init__(self, max_n=config.MAX_PARTICLES):
        self._n    = max_n
        self.pos   = np.zeros((max_n, 2), np.float32)   # (x, y)
        self.vel   = np.zeros((max_n, 2), np.float32)   # (vx, vy)
        self.col   = np.zeros((max_n, 3), np.uint8)     # BGR
        self.life  = np.zeros(max_n, np.float32)        # remaining frames
        self.mlife = np.zeros(max_n, np.float32)        # original lifetime
        self.size  = np.zeros(max_n, np.float32)        # radius pixels
        # per-particle gravity scale (1.0 = normal down, -1.0 = float up, 0 = none)
        self.grav  = np.zeros(max_n, np.float32)

    # ── Spawn helpers ─────────────────────────────────────────────────────────
    def _alloc(self, count):
        """Return an array of `count` free slot indices (oldest first)."""
        dead = np.where(self.life <= 0)[0]
        if len(dead) >= count:
            return dead[:count]
        # Overwrite oldest living particles
        oldest = np.argsort(self.life)[:count]
        return oldest

    def spawn(self, positions, velocities, colors, lifetimes, sizes, gravity=0.0):
        """
        All array args should be (N, …) numpy arrays or broadcastable.
        gravity: scalar applied per-slot.
        """
        n    = len(positions)
        if n == 0:
            return
        slots = self._alloc(n)
        k     = len(slots)
        self.pos  [slots] = positions[:k]
        self.vel  [slots] = velocities[:k]
        self.col  [slots] = np.clip(colors[:k], 0, 255).astype(np.uint8)
        self.life [slots] = lifetimes[:k]
        self.mlife[slots] = lifetimes[:k]
        self.size [slots] = sizes[:k]
        self.grav [slots] = gravity

    # ── Convenience factory methods ───────────────────────────────────────────
    def burst(self, cx, cy, color_bgr, count=120, speed=5.0,
              lifetime=45, size=4.0, gravity=0.06, spread=2 * np.pi):
        """Radial explosion burst centred at (cx, cy)."""
        angles = np.random.uniform(0, spread, count)
        spds   = np.random.uniform(speed * 0.4, speed, count).astype(np.float32)
        vx     = np.cos(angles) * spds
        vy     = np.sin(angles) * spds
        vel    = np.stack([vx, vy], axis=1).astype(np.float32)
        pos    = np.tile([cx, cy], (count, 1)).astype(np.float32)
        col    = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life   = np.random.uniform(lifetime * 0.6, lifetime, count).astype(np.float32)
        sz     = np.random.uniform(size * 0.5, size, count).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity)

    def orbit(self, cx, cy, radius, color_bgr, count=60, lifetime=80,
              speed_ang=0.07, size=3.0):
        """Spawn particles orbiting (cx, cy) at given radius."""
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        px     = cx + np.cos(angles) * radius
        py     = cy + np.sin(angles) * radius
        vx     = -np.sin(angles) * speed_ang * radius
        vy     =  np.cos(angles) * speed_ang * radius
        pos    = np.stack([px, py], axis=1).astype(np.float32)
        vel    = np.stack([vx, vy], axis=1).astype(np.float32)
        col    = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life   = np.full(count, float(lifetime), np.float32)
        sz     = np.full(count, size, np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.0)

    def vortex_inward(self, cx, cy, color_bgr, count=120, radius=120,
                      lifetime=50, size=3.5):
        """Particles spiral inward toward (cx, cy)."""
        angles = np.random.uniform(0, 2 * np.pi, count)
        radii  = np.random.uniform(radius * 0.4, radius, count)
        px     = cx + np.cos(angles) * radii
        py     = cy + np.sin(angles) * radii
        # velocity pointing inward with a slight tangent
        to_cx  = cx - px;  to_cy = cy - py
        norms  = np.maximum(np.sqrt(to_cx**2 + to_cy**2), 1)
        speed  = np.random.uniform(2.0, 5.0, count)
        vx     = (to_cx / norms + np.sin(angles) * 0.3) * speed
        vy     = (to_cy / norms - np.cos(angles) * 0.3) * speed
        pos    = np.stack([px, py], axis=1).astype(np.float32)
        vel    = np.stack([vx, vy], axis=1).astype(np.float32)
        col    = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life   = np.random.uniform(lifetime * 0.5, lifetime, count).astype(np.float32)
        sz     = np.random.uniform(2.0, size, count).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.0)

    def screen_storm(self, width, height, color_bgr, count=200, lifetime=60):
        """Random screen-filling particle storm (Domain of Aether)."""
        px   = np.random.uniform(0, width,  count).astype(np.float32)
        py   = np.random.uniform(0, height, count).astype(np.float32)
        vx   = np.random.uniform(-3, 3, count).astype(np.float32)
        vy   = np.random.uniform(-3, 3, count).astype(np.float32)
        pos  = np.stack([px, py], axis=1)
        vel  = np.stack([vx, vy], axis=1)
        col  = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life = np.random.uniform(lifetime * 0.5, lifetime, count).astype(np.float32)
        sz   = np.random.uniform(1.5, 5.0, count).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.0)

    # ── Update & render ───────────────────────────────────────────────────────
    def update(self):
        alive = self.life > 0
        self.pos [alive] += self.vel[alive]
        self.vel [alive, 1] += self.grav[alive]   # per-particle gravity
        self.life[alive] -= 1.0

    def render(self, overlay):
        """Draw all living particles onto overlay (uint8 BGR)."""
        idx = np.where(self.life > 0)[0]
        if idx.size == 0:
            return

        h, w  = overlay.shape[:2]
        px    = self.pos[idx, 0].astype(np.int32)
        py    = self.pos[idx, 1].astype(np.int32)
        alphas= np.clip(self.life[idx] / np.maximum(self.mlife[idx], 1), 0, 1)
        szs   = np.maximum(1, (self.size[idx] * alphas)).astype(np.int32)

        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        for i in np.where(valid)[0]:
            a = float(alphas[i])
            c = tuple(int(self.col[idx[i], ch] * a) for ch in range(3))
            cv2.circle(overlay, (int(px[i]), int(py[i])), int(szs[i]), c, -1)

    def beam(self, x1, y1, x2, y2, color_bgr, count=80,
             spread=8, lifetime=35, size=3.0):
        """Particles scattered along a line segment — slash / beam effect."""
        ts   = np.random.uniform(0, 1, count).astype(np.float32)
        px   = x1 + ts * (x2 - x1) + np.random.uniform(-spread, spread, count)
        py   = y1 + ts * (y2 - y1) + np.random.uniform(-spread, spread, count)
        # Perpendicular outward velocity
        dx, dy = x2 - x1, y2 - y1
        length = max(np.sqrt(dx**2 + dy**2), 1)
        nx, ny = -dy / length, dx / length
        speeds = np.random.uniform(-3, 3, count).astype(np.float32)
        vx = nx * speeds
        vy = ny * speeds
        pos  = np.stack([px, py], axis=1).astype(np.float32)
        vel  = np.stack([vx, vy], axis=1).astype(np.float32)
        col  = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life = np.random.uniform(lifetime * 0.6, lifetime, count).astype(np.float32)
        sz   = np.random.uniform(size * 0.5, size, count).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.03)

    def arc_shower(self, cx, cy, radius, color_bgr, count=80,
                   a_start=0.0, a_end=2 * np.pi, lifetime=50, size=3.5):
        """Particles along an arc that then drift outward."""
        angles = np.random.uniform(a_start, a_end, count).astype(np.float32)
        px     = cx + np.cos(angles) * radius
        py     = cy + np.sin(angles) * radius
        speeds = np.random.uniform(1.5, 4.5, count).astype(np.float32)
        vx     = np.cos(angles) * speeds
        vy     = np.sin(angles) * speeds
        pos    = np.stack([px, py], axis=1).astype(np.float32)
        vel    = np.stack([vx, vy], axis=1).astype(np.float32)
        col    = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life   = np.random.uniform(lifetime * 0.6, lifetime, count).astype(np.float32)
        sz     = np.random.uniform(size * 0.4, size, count).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.05)

    def lightning(self, x1, y1, x2, y2, color_bgr, segments=14, count_per=6):
        """Jagged bolt of bright short-lived particles between two points."""
        pts = [(x1, y1)]
        for i in range(1, segments):
            t     = i / segments
            mx    = x1 + (x2 - x1) * t + np.random.uniform(-20, 20)
            my    = y1 + (y2 - y1) * t + np.random.uniform(-20, 20)
            pts.append((mx, my))
        pts.append((x2, y2))

        all_pos, all_vel = [], []
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            for _ in range(count_per):
                t   = np.random.random()
                px  = ax + (bx - ax) * t + np.random.uniform(-4, 4)
                py  = ay + (by - ay) * t + np.random.uniform(-4, 4)
                all_pos.append([px, py])
                all_vel.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

        n    = len(all_pos)
        pos  = np.array(all_pos, np.float32)
        vel  = np.array(all_vel, np.float32)
        col  = np.tile(color_bgr, (n, 1)).astype(np.uint8)
        life = np.random.uniform(8, 18, n).astype(np.float32)
        sz   = np.random.uniform(1.5, 3.5, n).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.0)

    def ring_expand(self, cx, cy, radius, color_bgr, count=100, lifetime=40, size=3.0):
        """Particles spawned on a ring and launched outward — ring-burst."""
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        px     = cx + np.cos(angles) * radius
        py     = cy + np.sin(angles) * radius
        spds   = np.random.uniform(3.5, 8.0, count).astype(np.float32)
        vx     = np.cos(angles) * spds
        vy     = np.sin(angles) * spds
        pos    = np.stack([px, py], axis=1).astype(np.float32)
        vel    = np.stack([vx, vy], axis=1).astype(np.float32)
        col    = np.tile(color_bgr, (count, 1)).astype(np.uint8)
        life   = np.random.uniform(lifetime * 0.6, lifetime, count).astype(np.float32)
        sz     = np.random.uniform(size * 0.5, size, count).astype(np.float32)
        self.spawn(pos, vel, col, life, sz, gravity=0.04)

    def clear(self):
        self.life[:] = 0
