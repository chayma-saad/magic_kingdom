"""
Class selection screen for Arcane Academy.

Provides `ClassConfig` dataclass and `ClassSelectScreen` which renders an
in-game class selection UI on a supplied OpenCV frame size. The `run()` method
blocks until the user confirms a class and returns the chosen `ClassConfig`.

All rendering uses numpy + cv2 so it can be composited into the main game frame.
"""
from dataclasses import dataclass
import cv2
import numpy as np
import time
import math
from selection_screen import _Particles  # reuse particle logic


@dataclass
class ClassConfig:
    id: str
    name: str
    primary_element: str
    allowed_elements: tuple
    accent_color: tuple  # BGR


class ClassSelectScreen:
    """Render a horizontal class chooser and return the selected ClassConfig.

    run(frame_w, frame_h, mastery_dict) -> ClassConfig
    """

    def __init__(self):
        # Card definitions — order matters and matches number keys 1..5
        self._cards = [
            ClassConfig('pyro', 'Pyromancer', 'FIRE', ('FIRE', 'NEUTRAL'), (0, 69, 255)),
            ClassConfig('tide', 'Tide Caller', 'WATER', ('WATER', 'NEUTRAL'), (255, 255, 0)),
            ClassConfig('void', 'Void Walker', 'VOID', ('VOID', 'NEUTRAL'), (255, 0, 123)),
            ClassConfig('sage', 'Sage', 'LIGHT', ('LIGHT', 'NEUTRAL'), (205, 250, 255)),
            ClassConfig('free', 'Free Caster', 'NEUTRAL', ('FIRE','WATER','LIGHT','VOID','NEUTRAL'), (240,240,240)),
        ]

    def run(self, frame_w: int, frame_h: int, mastery_dict: dict) -> ClassConfig:
        # Create a window and run a local event loop (blocks until selection)
        win = 'CLASS SELECT'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, frame_w, frame_h)

        selected = 0
        particles = [_Particles('gold_dust') for _ in self._cards]
        t0 = time.time()

        while True:
            t = time.time() - t0
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            # Background vignette
            frame[:] = (8, 4, 18)

            # Draw centered title
            title = 'Arcane Academy — Choose your Class'
            font = cv2.FONT_HERSHEY_COMPLEX
            (tw, th), _ = cv2.getTextSize(title, font, 1.1, 2)
            tx = (frame_w - tw) // 2
            cv2.putText(frame, title, (tx, 60), font, 1.1, (220,220,220), 2, cv2.LINE_AA)

            # Layout: horizontal strip
            card_w = int(frame_w * 0.18)
            card_h = int(frame_h * 0.42)
            gap = int((frame_w - card_w * len(self._cards)) / (len(self._cards) + 1))
            y = (frame_h - card_h) // 2

            for i, cfg in enumerate(self._cards):
                x = gap + i * (card_w + gap)
                # selected card scales up slightly
                scale = 1.08 if i == selected else 1.0
                cw = int(card_w * scale)
                ch = int(card_h * scale)
                cx = x - (cw - card_w) // 2
                cy = y - (ch - card_h) // 2

                # Card background
                cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (18, 8, 26), -1)
                # Accent glow (blurred)
                glow = np.zeros_like(frame)
                cv2.rectangle(glow, (cx + 6, cy + 6), (cx + cw - 6, cy + ch - 6), cfg.accent_color, -1)
                bl = cv2.GaussianBlur(glow, (51,51), 0)
                frame = np.clip(frame.astype(np.float32) + bl.astype(np.float32)*0.18, 0, 255).astype(np.uint8)

                # Title
                f = cv2.FONT_HERSHEY_COMPLEX
                s = 0.7 * scale
                (twc, _), _ = cv2.getTextSize(cfg.name, f, s, 1)
                cv2.putText(frame, cfg.name, (cx + 12, cy + 34), f, s, cfg.accent_color, 1, cv2.LINE_AA)

                # Element badge (circle)
                cv2.circle(frame, (cx + cw - 34, cy + 34), 18, cfg.accent_color, -1)
                cv2.putText(frame, cfg.primary_element[0], (cx + cw - 40, cy + 40),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (10,10,10), 1, cv2.LINE_AA)

                # Signature spells list
                spells = []
                if cfg.id == 'pyro': spells = ['Ignition Pulse', 'Firestorm', 'Exploding Ward']
                elif cfg.id == 'tide': spells = ['Tide Blade', 'Maelstrom', 'Barrier Field']
                elif cfg.id == 'void': spells = ['Void Collapse', 'Reality Slice', 'Void Beam']
                elif cfg.id == 'sage': spells = ['Prism Lance', 'Radiant Pulse', 'Convergence']
                else: spells = ['Arcane Ward', 'Domain of Aether', 'Nova Burst']

                for si, sname in enumerate(spells):
                    cv2.putText(frame, u'• ' + sname, (cx + 14, cy + 70 + si * 22),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.45 * scale, (200,200,200), 1, cv2.LINE_AA)

                # Mastery bar
                mastery_pct = 0
                # masteries are stored per element name
                if isinstance(cfg.allowed_elements, (list, tuple)):
                    # for single-element classes, pick the primary
                    if cfg.primary_element in mastery_dict:
                        mastery_pct = mastery_dict.get(cfg.primary_element, 0)
                    else:
                        # free caster: average all
                        vals = [mastery_dict.get(e,0) for e in cfg.allowed_elements]
                        mastery_pct = int(sum(vals)/len(vals)) if vals else 0

                bar_x = cx + 12
                bar_y = cy + ch - 20
                bar_w = cw - 24
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (30,20,40), -1)
                fill = int(bar_w * max(0, min(1, mastery_pct/100)))
                if fill > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 8), cfg.accent_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (80,70,110), 1)

                # Local particle effect — draw into card area using particle system
                # map particle coords into card local coordinates
                buf = np.zeros((ch, cw, 3), dtype=np.uint8)
                # temporarily set global particle system to draw in a moved canvas
                particles[i]._init()
                particles[i].update_draw(buf, t)
                # paste buf into frame with alpha blend
                alpha = 0.28 if i == selected else 0.12
                sub = frame[cy:cy+ch, cx:cx+cw].astype(np.float32)
                frame[cy:cy+ch, cx:cx+cw] = np.clip(sub * (1-alpha) + buf.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

            cv2.imshow(win, frame)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(win)
                return None
            elif key in (ord('a'), 81):  # left
                selected = max(0, selected - 1)
            elif key in (ord('d'), 83):  # right
                selected = min(len(self._cards) - 1, selected + 1)
            elif key in (10, 13, 32):  # ENTER / SPACE
                cv2.destroyWindow(win)
                return self._cards[selected]
            elif key in map(lambda n: ord(str(n)), range(1,6)):
                idx = int(chr(key)) - 1
                if 0 <= idx < len(self._cards):
                    cv2.destroyWindow(win)
                    return self._cards[idx]
