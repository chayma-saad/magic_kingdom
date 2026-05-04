"""
Fruit Basket — hand-slicing arcade game.

Fruits arc up from the bottom of the frame.  The player slashes them
by moving their index fingertip fast through the fruit's hitbox.
Hitting a bomb costs a life.  Missing a fruit costs a life.
Reach WIN_SCORE to win; drop to 0 lives to lose.

Usage
-----
    game = FruitGame(frame_w, frame_h)

    # Each frame:
    game.update(tip=(x, y))   # tip = index-fingertip pixel (or None)
    game.render(frame)         # draws on top of the camera image
    if game.state != 'playing':
        ...  # 'win' | 'lose'

    game.reset()               # restart
"""
import cv2
import numpy as np
import math
import random
import time
from collections import deque

WIN_SCORE  = 150
START_LIVES = 3
GRAVITY    = 0.42

# ── Fruit catalogue ───────────────────────────────────────────────────────────
_FRUITS = [
    dict(name='apple',      color=(30,  30,  210), hi=(130, 100, 240), r=28, pts=10),
    dict(name='orange',     color=(0,   140, 255), hi=(80,  200, 255), r=26, pts=10),
    dict(name='watermelon', color=(30,  160, 40),  hi=(120, 220, 100), r=34, pts=15),
    dict(name='banana',     color=(0,   220, 255), hi=(100, 240, 255), r=22, pts=10),
    dict(name='strawberry', color=(40,  40,  220), hi=(140, 110, 240), r=22, pts=20),
    dict(name='grape',      color=(160, 20,  160), hi=(210, 100, 210), r=20, pts=15),
]
_BOMB = dict(name='bomb', color=(30, 30, 30), hi=(80, 80, 80), r=26, pts=0)


# ── Fruit object ──────────────────────────────────────────────────────────────

class _Fruit:
    def __init__(self, x, y, vx, vy, info):
        self.x, self.y    = float(x), float(y)
        self.vx, self.vy  = float(vx), float(vy)
        self.info         = info
        self.r            = info['r']
        self.color        = info['color']
        self.highlight    = info['hi']
        self.is_bomb      = info['name'] == 'bomb'
        self.pts          = info['pts']
        self.alive        = True
        self._angle       = random.uniform(0, 2 * math.pi)
        self._spin        = random.uniform(-0.08, 0.08)

    def update(self):
        self.x      += self.vx
        self.y      += self.vy
        self.vy     += GRAVITY
        self._angle += self._spin

    def draw(self, img, t):
        cx, cy = int(self.x), int(self.y)
        r      = self.r

        if self.is_bomb:
            # Dark sphere with fuse
            cv2.circle(img, (cx, cy), r, self.color, -1)
            cv2.circle(img, (cx, cy), r, (80, 80, 80), 2)
            # Fuse line
            fx, fy = cx + int(r * 0.3), cy - r
            cv2.line(img, (cx, cy - r), (fx, fy - 14), (60, 60, 60), 2)
            # Sparkling tip
            spark = (0, 200, 255) if int(t * 10) % 2 == 0 else (0, 100, 200)
            cv2.circle(img, (fx, fy - 14), 5, spark, -1)
            cv2.putText(img, '!', (cx - 5, cy + 7),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (100, 100, 100), 1)
        else:
            # Fruit body
            cv2.circle(img, (cx, cy), r, self.color, -1)
            # Highlight dot (makes it look 3D)
            hx = int(cx - r * 0.30 * math.cos(self._angle))
            hy = int(cy - r * 0.30 * math.sin(self._angle))
            cv2.circle(img, (hx, hy), max(3, r // 4), self.highlight, -1)
            # Border
            cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)
            # Stem (not on bomb)
            sx  = int(cx + r * 0.15 * math.cos(self._angle - 1.2))
            sy  = int(cy - r + int(r * 0.1 * math.sin(self._angle)))
            sx2 = int(sx + 5 * math.cos(self._angle - 0.5))
            sy2 = int(sy - 8)
            cv2.line(img, (sx, sy), (sx2, sy2), (30, 100, 30), 2)


# ── Sliced half ───────────────────────────────────────────────────────────────

class _Half:
    def __init__(self, x, y, vx, vy, color, hi, r, upper):
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.color, self.hi = color, hi
        self.r           = r
        self.upper       = upper
        self.life        = 35
        self.mlife       = 35

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.55
        self.life -= 1

    def draw(self, img):
        if self.life <= 0:
            return
        a   = self.life / self.mlife
        cx, cy = int(self.x), int(self.y)
        start  = 0   if self.upper else 180
        end    = 180 if self.upper else 360
        col    = tuple(int(c * a) for c in self.color)
        hi_c   = tuple(int(c * a) for c in self.hi)
        cv2.ellipse(img, (cx, cy), (self.r, self.r), 0, start, end, col, -1)
        cv2.ellipse(img, (cx, cy), (self.r, self.r), 0, start, end, (255,255,255), 1)
        # Juice inner
        cv2.ellipse(img, (cx, cy), (self.r//2, self.r//2), 0, start, end, hi_c, -1)


# ── Splash particle ───────────────────────────────────────────────────────────

class _Spark:
    def __init__(self, x, y, color):
        ang       = random.uniform(0, 2 * math.pi)
        spd       = random.uniform(3, 9)
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = math.cos(ang)*spd, math.sin(ang)*spd
        self.color = color
        self.life  = random.randint(15, 28)
        self.mlife = self.life
        self.r     = random.randint(3, 6)

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.4
        self.life -= 1

    def draw(self, img):
        if self.life <= 0:
            return
        a = self.life / self.mlife
        c = tuple(int(ch * a) for ch in self.color)
        cv2.circle(img, (int(self.x), int(self.y)), self.r, c, -1)


# ── Main game class ───────────────────────────────────────────────────────────

class FruitGame:
    def __init__(self, w, h):
        self.w, self.h    = w, h
        self._fruits: list[_Fruit] = []
        self._halves: list[_Half]  = []
        self._sparks: list[_Spark] = []

        self.score    = 0
        self.lives    = START_LIVES
        self.state    = 'playing'   # 'playing' | 'win' | 'lose'

        self._tip_trail  = deque(maxlen=10)
        self._prev_tip   = None

        self._spawn_next = time.time() + 1.0
        self._spawn_int  = 1.4

        self._combo      = 0
        self._combo_t    = 0.0
        self._combo_pts  = 0        # pts added this combo flash
        self._slash_trail: list = []  # [(x,y,age)] for rendering
        self._t          = 0

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, tip=None):
        if self.state != 'playing':
            return

        self._t += 1
        now      = time.time()

        # Spawn
        if now >= self._spawn_next:
            self._spawn()
            self._spawn_next = now + self._spawn_int
            self._spawn_int  = max(0.7, 1.4 - self.score * 0.004)

        # Track tip
        tip_vel = 0.0
        if tip is not None:
            self._tip_trail.append(tip)
            self._slash_trail.append([tip[0], tip[1], 12])
            if self._prev_tip is not None:
                tip_vel = math.sqrt((tip[0] - self._prev_tip[0])**2 +
                                    (tip[1] - self._prev_tip[1])**2)
            self._prev_tip = tip

        # Decay slash trail
        self._slash_trail = [[x, y, a-1] for x, y, a in self._slash_trail if a > 0]

        # Slice detection — only when moving fast
        sliced_now = 0
        if tip_vel > 15 and len(self._tip_trail) >= 2:
            p1 = self._tip_trail[-2]
            p2 = self._tip_trail[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg2 = dx*dx + dy*dy

            for fruit in self._fruits:
                if not fruit.alive:
                    continue
                fx, fy = fruit.x, fruit.y
                if seg2 < 1:
                    continue
                t_ = max(0.0, min(1.0, ((fx-p1[0])*dx + (fy-p1[1])*dy) / seg2))
                clx = p1[0] + t_*dx - fx
                cly = p1[1] + t_*dy - fy
                if math.sqrt(clx*clx + cly*cly) < fruit.r:
                    fruit.alive = False
                    if fruit.is_bomb:
                        self.lives = max(0, self.lives - 1)
                        if self.lives == 0:
                            self.state = 'lose'
                    else:
                        sliced_now += 1
                        self._slice_fruit(fruit, dx, dy, seg2)

        # Combo tracking
        if sliced_now > 0:
            if now - self._combo_t < 0.9:
                self._combo += sliced_now
            else:
                self._combo = sliced_now
            self._combo_t = now
            pts = sliced_now * 10 * max(1, self._combo // 2 + 1)
            self.score     += pts
            self._combo_pts = pts
            if self.score >= WIN_SCORE:
                self.state = 'win'

        # Physics
        for f in self._fruits:
            if f.alive:
                f.update()
        for h in self._halves:
            h.update()
        for s in self._sparks:
            s.update()

        # Off-screen fruits = missed
        for f in self._fruits:
            if f.alive and f.y > self.h + 80:
                f.alive = False
                if not f.is_bomb:
                    self.lives = max(0, self.lives - 1)
                    if self.lives == 0:
                        self.state = 'lose'

        self._fruits = [f for f in self._fruits if f.alive]
        self._halves = [h for h in self._halves if h.life > 0]
        self._sparks = [s for s in self._sparks if s.life > 0]

    def _spawn(self):
        n        = random.randint(1, min(3, 1 + self.score // 40))
        has_bomb = random.random() < 0.13

        for i in range(n):
            x  = random.uniform(self.w * 0.1, self.w * 0.9)
            vx = random.uniform(-2.5, 2.5)
            vy = random.uniform(-17, -12)
            info = _BOMB if (has_bomb and i == 0) else random.choice(_FRUITS)
            self._fruits.append(_Fruit(x, self.h + 30, vx, vy, info))

    def _slice_fruit(self, fruit, dx, dy, seg2):
        cx, cy = int(fruit.x), int(fruit.y)
        norm   = max(1.0, math.sqrt(seg2))
        # Halves fly perpendicular to slash + upward
        for upper in (True, False):
            sign = 1 if upper else -1
            self._halves.append(_Half(
                cx, cy,
                -dy/norm * 3 * sign + dx/norm * 2,
                -abs(dy/norm) * 5 - 2,
                fruit.color, fruit.highlight, fruit.r, upper
            ))
        # Sparks
        for _ in range(14):
            self._sparks.append(_Spark(cx, cy, fruit.highlight))

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, frame):
        t = self._t / 30.0

        # Fruits
        for f in self._fruits:
            f.draw(frame, t)

        # Sliced halves + sparks
        for h in self._halves:
            h.draw(frame)
        for s in self._sparks:
            s.draw(frame)

        # Slash trail
        st = self._slash_trail
        for i in range(1, len(st)):
            a1 = st[i-1][2] / 12
            a2 = st[i][2]   / 12
            col = (int(180*a2), int(220*a2), 255)
            cv2.line(frame, (int(st[i-1][0]), int(st[i-1][1])),
                     (int(st[i][0]),   int(st[i][1])), col,
                     max(1, int(5 * a2)), cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────
        H, W = frame.shape[:2]
        font = cv2.FONT_HERSHEY_COMPLEX

        # Score
        cv2.putText(frame, f'{self.score}', (W//2 - 50, 55),
                    font, 1.6, (0, 220, 80), 3, cv2.LINE_AA)
        cv2.putText(frame, 'SCORE', (W//2 - 40, 22),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.55, (120, 200, 120), 1)

        # Lives (hearts)
        for i in range(START_LIVES):
            col = (0, 50, 220) if i < self.lives else (50, 50, 50)
            cv2.circle(frame, (W - 40 - i * 42, 35), 14, col, -1)
        cv2.putText(frame, 'LIVES', (W - 180, 55),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.52, (180, 180, 180), 1)

        # Win target hint
        cv2.putText(frame, f'GOAL: {WIN_SCORE} pts', (16, 22),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.50, (80, 160, 80), 1)

        # Combo flash
        now = time.time()
        if self._combo > 1 and now - self._combo_t < 0.9:
            fade = 1.0 - (now - self._combo_t) / 0.9
            col  = tuple(int(c * fade) for c in (0, 230, 255))
            cv2.putText(frame, f'x{self._combo}  COMBO!  +{self._combo_pts}',
                        (W//2 - 160, H//2 - 80),
                        font, 1.5, col, 3, cv2.LINE_AA)

        # End screens
        if self.state == 'win':
            self._draw_end(frame, 'YOU WIN!',  (0, 220, 80))
        elif self.state == 'lose':
            self._draw_end(frame, 'GAME OVER', (0, 50, 220))

    def _draw_end(self, frame, text, color):
        H, W  = frame.shape[:2]
        dark  = frame.copy()
        dark[:] = (0, 0, 0)
        cv2.addWeighted(frame, 0.35, dark, 0.65, 0, dst=frame)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2.2, 3)
        tx, ty = (W - tw)//2, H//2 - 20
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_COMPLEX, 2.2, color, 3, cv2.LINE_AA)
        cv2.putText(frame, f'Score: {self.score}',
                    ((W - 180)//2, ty + th + 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (200, 200, 200), 1)
        cv2.putText(frame, 'Press  R  to play again',
                    ((W - 250)//2, ty + th + 60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (140, 140, 140), 1)

    def reset(self):
        self._fruits.clear()
        self._halves.clear()
        self._sparks.clear()
        self._slash_trail.clear()
        self._tip_trail.clear()
        self._prev_tip   = None
        self.score       = 0
        self.lives       = START_LIVES
        self.state       = 'playing'
        self._spawn_next = time.time() + 1.0
        self._spawn_int  = 1.4
        self._combo      = 0
        self._combo_t    = 0.0
        self._combo_pts  = 0
        self._t          = 0
