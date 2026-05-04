"""
Compact class switch overlay rendered on top of the game frame.

Renders small circular icons for the five Arcane classes and provides a
`handle_input()` helper which maps numeric keys to index-based class switching.
"""
import cv2
import numpy as np
from class_select import ClassConfig


class ClassSwitchOverlay:
    def __init__(self, classes):
        """classes: list[ClassConfig]"""
        self.classes = classes
        self.visible = False

    def render(self, frame, highlight_idx=0):
        H, W = frame.shape[:2]
        # Compact horizontal strip at top center
        strip_w = min(W - 40, 420)
        strip_h = 56
        x = (W - strip_w) // 2
        y = 8
        # Background
        overlay = frame[y:y+strip_h, x:x+strip_w].astype(np.float32)
        frame[y:y+strip_h, x:x+strip_w] = (overlay * 0.35).astype(np.uint8)

        # Draw icons
        n = len(self.classes)
        gap = strip_w // (n + 1)
        for i, cfg in enumerate(self.classes):
            cx = x + gap * (i + 1)
            cy = y + strip_h // 2
            r = 16
            col = cfg.accent_color
            cv2.circle(frame, (cx, cy), r, col, -1)
            # Initial inside
            init = cfg.name[0] if cfg.name else '?'
            cv2.putText(frame, init, (cx - 6, cy + 6), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7, (10,10,10), 1, cv2.LINE_AA)
            if i == highlight_idx:
                cv2.circle(frame, (cx, cy), r+3, (230,230,230), 1)

    def handle_input(self, key):
        """Return index (0-based) if numeric 1..5 pressed, otherwise None."""
        if key in map(lambda n: ord(str(n)), range(1, len(self.classes)+1)):
            return int(chr(key)) - 1
        return None
