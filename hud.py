"""
Grimoire HUD — semi-transparent sidebar panel rendered onto the live frame.
Tries PIL/Pillow TTF fonts; falls back to cv2 built-in fonts.
"""
import cv2
import numpy as np
import math
import time
import config

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False


class HUDRenderer:
    def __init__(self):
        self._fonts = {}
        if _PIL:
            self._load_fonts()

    def _load_fonts(self):
        paths = [
            'C:/Windows/Fonts/consola.ttf',
            'C:/Windows/Fonts/lucon.ttf',
            'C:/Windows/Fonts/cour.ttf',
            'C:/Windows/Fonts/arial.ttf',
        ]
        for p in paths:
            try:
                self._fonts = {
                    'lg': ImageFont.truetype(p, 16),
                    'md': ImageFont.truetype(p, 13),
                    'sm': ImageFont.truetype(p, 11),
                }
                break
            except (IOError, OSError):
                continue

    # ── Main entry ────────────────────────────────────────────────────────────

    def render(self, frame, element_engine, spell_system, theme=None):
        h, w  = frame.shape[:2]
        hw    = config.HUD_WIDTH
        panel = np.full((h, hw, 3), config.HUD_BG_BGR, dtype=np.uint8)

        # Animated border when charging
        border_c = config.NEON_BLUE
        if spell_system.active and spell_system.active.state == 'CHARGING':
            pulse    = int(127 + 127 * math.sin(time.time() * 6))
            border_c = (pulse, int(pulse * 0.5), 0)
        cv2.rectangle(panel, (0, 0), (hw - 1, h - 1), border_c, 1)

        y = self._header(panel, y=16, theme=theme)
        y = self._active_spell(panel, y, spell_system)
        y = self._element_block(panel, y, element_engine.current, spell_system)
        y = self._spell_history(panel, y, spell_system)
        y = self._mastery_bars(panel, y, spell_system)
        y = self._combo_block(panel, y, spell_system)
        self._overload_warning(panel, spell_system)

        # Blend panel onto left strip of frame
        roi     = frame[:, :hw]
        blended = cv2.addWeighted(roi, 1.0 - config.HUD_ALPHA,
                                  panel, config.HUD_ALPHA, 0)
        frame[:, :hw] = blended

    # ── Sections ──────────────────────────────────────────────────────────────

    def _header(self, panel, y, theme=None):
        self._text(panel, 'A E T H E R', (10, y), config.GOLD, scale=0.55, bold=True)
        y += 18
        self._text(panel, 'S C R I P T', (10, y), config.NEON_BLUE, scale=0.42)
        y += 13
        cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), config.GOLD, 1)
        y += 7
        if theme is not None:
            label_col = theme.accent
            self._text(panel, theme.element_label + '  ' + theme.name,
                       (10, y), label_col, scale=0.38)
            y += 13
            cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), (40, 20, 35), 1)
            y += 4
        return y

    def _active_spell(self, panel, y, spell_system):
        sp = spell_system.active
        if sp is None:
            self._text(panel, 'IDLE', (10, y), (60, 60, 60), scale=0.45)
            y += 16
            cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), (40, 20, 30), 1)
            return y + 6

        color  = sp.color
        state  = sp.state

        # Spell name (may be wide — clip it)
        name   = sp.name
        if len(name) > 16:
            name = name[:15] + '~'
        self._text(panel, name, (10, y), color, scale=0.48, bold=True)
        y += 16

        # State label + charge bar
        state_labels = {
            'CHARGING': 'CHARGING',
            'RELEASE':  'RELEASE!',
            'DECAY':    'FADING..',
        }
        label_colors = {
            'CHARGING': config.NEON_BLUE,
            'RELEASE':  config.GOLD,
            'DECAY':    (80, 80, 80),
        }
        lbl_c = label_colors.get(state, (100, 100, 100))
        self._text(panel, state_labels.get(state, state), (10, y), lbl_c, scale=0.38)
        y += 13

        # Charge bar
        charge = spell_system.charge_frac
        if charge > 0:
            bar_w = config.HUD_WIDTH - 20
            fill  = int(bar_w * charge)
            bar_c = (0, 0, 200) if charge > 0.85 else lbl_c
            cv2.rectangle(panel, (10, y), (10 + bar_w, y + 6), (40, 25, 35), -1)
            if fill > 0:
                cv2.rectangle(panel, (10, y), (10 + fill, y + 6), bar_c, -1)
            y += 10

        cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), (50, 20, 35), 1)
        return y + 7

    def _element_block(self, panel, y, element, spell_system):
        color = config.ELEMENT_COLORS.get(element, config.GOLD)
        icon  = config.ELEMENT_ICONS.get(element, '?')
        self._text(panel, 'ELEMENT', (10, y), (100, 100, 100), scale=0.36)
        y += 13
        # Pulsing dot indicator
        pulse_r = int(5 + 2 * math.sin(time.time() * 4))
        cv2.circle(panel, (14, y - 4), pulse_r, color, -1)
        self._text(panel, icon, (28, y), color, scale=0.45, bold=True)
        y += 15
        cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), (50, 20, 35), 1)
        return y + 7

    def _spell_history(self, panel, y, spell_system):
        self._text(panel, 'RECENT SPELLS', (10, y), (100, 100, 100), scale=0.36)
        y += 13
        spells = list(spell_system.history)
        if not spells:
            self._text(panel, '-- none --', (14, y), (55, 55, 55), scale=0.36)
            y += 13
        else:
            for name in reversed(spells[-5:]):
                display = name[:18] + '..' if len(name) > 18 else name
                self._text(panel, f'> {display}', (12, y), config.GOLD, scale=0.38)
                y += 13
        cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), (50, 20, 35), 1)
        return y + 7

    def _mastery_bars(self, panel, y, spell_system):
        self._text(panel, 'MASTERY', (10, y), (100, 100, 100), scale=0.36)
        y += 13
        bar_w = config.HUD_WIDTH - 72
        for elem in config.ELEMENT_LIST:
            val   = spell_system.mastery.get(elem, 0)
            color = config.ELEMENT_COLORS.get(elem, config.GOLD)
            self._text(panel, elem[:3], (10, y), color, scale=0.36)
            bx = 46
            # Background bar
            cv2.rectangle(panel, (bx, y - 7), (bx + bar_w, y + 2), (38, 20, 28), -1)
            # Filled portion
            fill = int(bar_w * val / 100)
            if fill > 0:
                cv2.rectangle(panel, (bx, y - 7), (bx + fill, y + 2), color, -1)
            # Value text
            self._text(panel, str(val), (bx + bar_w + 4, y), color, scale=0.33)
            y += 13
        cv2.line(panel, (8, y), (config.HUD_WIDTH - 8, y), (50, 20, 35), 1)
        return y + 7

    def _combo_block(self, panel, y, spell_system):
        self._text(panel, 'COMBO', (10, y), (100, 100, 100), scale=0.36)
        y += 13
        combo = spell_system.combo_progress
        if combo:
            parts = ' → '.join(s.upper() for s in combo)
            self._text(panel, parts, (10, y), config.NEON_BLUE, scale=0.38)
            y += 13
        else:
            self._text(panel, '-- --', (14, y), (55, 55, 55), scale=0.36)
            y += 13
        return y + 5

    def _overload_warning(self, panel, spell_system):
        if not spell_system.overloading:
            return
        h = panel.shape[0]
        t = time.time()
        pulse = int(100 + 155 * abs(math.sin(t * 8)))
        wc    = (0, 0, pulse)
        cv2.rectangle(panel, (0, 0), (config.HUD_WIDTH - 1, h - 1), wc, 3)
        self._text(panel, '!! OVERLOAD !!', (8, h - 22), wc, scale=0.48, bold=True)

    # ── Text helper ───────────────────────────────────────────────────────────

    def _text(self, img, text, pos, color, scale=0.42, bold=False):
        if _PIL and self._fonts:
            self._pil_text(img, text, pos, color, scale)
        else:
            thick = 2 if bold else 1
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        scale, color, thick, cv2.LINE_AA)

    def _pil_text(self, img, text, pos, color, scale):
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = (self._fonts.get('lg') if scale > 0.5
                else self._fonts.get('md') if scale > 0.38
                else self._fonts.get('sm'))
        if font is None:
            return
        r, g, b = color[2], color[1], color[0]
        draw.text(pos, text, font=font, fill=(r, g, b))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
