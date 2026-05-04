"""
AETHER SCRIPT — Real-time computer vision spellcasting system.

Detection modes (chosen at startup)
------------------------------------
Aether Magic  [1-5] — draw shapes with finger trail, pinch to release
Naruto        [6]   — hold hand-sign poses (~1.2 s) to cast jutsu
One Piece     [7]   — full body skeleton; physical Luffy-style moves

Controls
--------
ESC    quit
r      reset
g      toggle Grimoire HUD
d      toggle debug skeleton
b      toggle Spell Book panel
SPACE  force-release (magic only)
"""
import os
os.environ['GLOG_minloglevel']      = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['ABSL_LOGGING_LEVEL']    = '3'

import cv2
import numpy as np
import argparse
import math
import time
from collections import deque

import config
from hand_tracker        import HandTracker
from trail               import Trail, TrailRenderer
from shape_detector      import ShapeDetector, draw_debug_corners
from element_engine      import ElementEngine
from spell_system        import SpellSystem
from particles           import ParticleSystem
from hud                 import HUDRenderer
from effects             import ScreenEffects
from audio               import AudioEngine
from selection_screen    import show_selection_screen
from action_guide        import ActionGuide
from naruto_signs        import NarutoSignDetector
from chakra_effects      import CloneEffect, play_jutsu, draw_sign_hud
from fruit_game          import FruitGame
from class_select        import ClassSelectScreen, ClassConfig
from class_switch_overlay import ClassSwitchOverlay

# One Piece body-detection imports (loaded lazily — only if theme selected)
_pose_tracker_mod   = None
_body_detector_mod  = None


def _load_one_piece_modules():
    global _pose_tracker_mod, _body_detector_mod
    if _pose_tracker_mod is None:
        import pose_tracker
        import body_action_detector
        _pose_tracker_mod  = pose_tracker
        _body_detector_mod = body_action_detector


# ── Constants ─────────────────────────────────────────────────────────────────
_VEL_HISTORY   = 6
_SPIKE_MULT    = 3.2
_SPIKE_ABS_MIN = 22
_DUAL_SPREAD   = 0.32

_OP_RED = (30, 30, 220)    # Luffy red (BGR)


# ── Utility ───────────────────────────────────────────────────────────────────

def desaturate(frame, factor=0.70):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    np.clip(hsv[:, :, 1], 0, 255, out=hsv[:, :, 1])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def is_pinching(thumb, finger, hand_size):
    return np.linalg.norm(np.array(thumb) - np.array(finger)) < hand_size * config.PINCH_RATIO


def draw_fps(frame, fps):
    cv2.putText(frame, f'{fps:.0f} fps',
                (config.HUD_WIDTH + 8, 22),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.52, (60, 200, 60), 1, cv2.LINE_AA)


def draw_exit_button(frame, bx, by, bw, bh):
    """Render a glowing EXIT button at the given position."""
    # Dark pill background
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (20, 8, 30), -1)
    # Red border — pulses gently
    pulse = int(160 + 60 * math.sin(time.time() * 3))
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 30, pulse), 2)
    # Label
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(frame, '[ EXIT ]', (bx + 6, by + 21),
                font, 0.55, (80, 80, 220), 1, cv2.LINE_AA)


def draw_debug_elem(frame, element, x, y):
    color = config.ELEMENT_COLORS.get(element, config.GOLD)
    cv2.putText(frame, f'[{element}]', (x, y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.45, color, 1, cv2.LINE_AA)


def draw_naruto_sign_hud(frame, sign_detectors, hands):
    """Live hand-sign name + charging bar near each hand."""
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for idx, hand in enumerate(hands[:2]):
        det      = sign_detectors[idx]
        sign     = det.current_sign
        progress = det.hold_progress
        if not sign or progress <= 0:
            continue
        tip   = hand['fingertip']
        color = SIGN_COLORS.get(sign, (200, 200, 200))
        hx, hy = tip[0] - 80, max(50, tip[1] - 60)
        cv2.rectangle(frame, (hx - 4, hy - 18), (hx + 170, hy + 22), (12, 4, 20), -1)
        cv2.rectangle(frame, (hx - 4, hy - 18), (hx + 170, hy + 22), color, 1)
        cv2.putText(frame, SIGN_LABELS.get(sign, sign), (hx, hy - 2),
                    font, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(frame, det.current_jutsu, (hx, hy + 14),
                    font, 0.42, tuple(int(c * 0.65) for c in color), 1, cv2.LINE_AA)
        bar_w = 166
        fill  = int(bar_w * progress)
        cv2.rectangle(frame, (hx - 2, hy + 18), (hx + bar_w, hy + 22), (35, 20, 45), -1)
        if fill > 0:
            cv2.rectangle(frame, (hx - 2, hy + 18), (hx - 2 + fill, hy + 22), color, -1)
        glow_r = int(8 + 4 * math.sin(time.time() * 8))
        cv2.circle(frame, tip, glow_r, color, 2)
        cv2.circle(frame, tip, glow_r // 2, (255, 255, 255), -1)


def draw_op_move_hint(frame, move_name, effect_xy):
    """Brief flash of the move name near the impact point."""
    if not move_name:
        return
    x, y = effect_xy
    font  = cv2.FONT_HERSHEY_COMPLEX_SMALL
    (tw, _), _ = cv2.getTextSize(move_name, font, 0.65, 1)
    tx = max(10, x - tw // 2)
    ty = max(30, y - 30)
    cv2.putText(frame, move_name, (tx + 2, ty + 2), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, move_name, (tx, ty),         font, 0.65, _OP_RED,   1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AETHER SCRIPT')
    parser.add_argument('--debug',     action='store_true')
    parser.add_argument('--no-audio',  action='store_true')
    parser.add_argument('--no-select', action='store_true',
                        help='Skip theme selection (uses Arcane Master)')
    args = parser.parse_args()

    # ── Theme selection ───────────────────────────────────────────────────────
    if args.no_select:
        from theme import DEFAULT_THEME
        chosen_theme = DEFAULT_THEME
    else:
        chosen_theme = show_selection_screen()
    print(f'[AETHER] Theme: {chosen_theme.name}')

    is_naruto    = chosen_theme.id == 'naruto'
    is_one_piece = chosen_theme.id == 'one_piece'
    is_game      = chosen_theme.id == 'game'
    is_magic     = chosen_theme.category == 'magic'

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('[AETHER] Could not open camera 0.')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or config.FRAME_WIDTH
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
    print(f'[AETHER] Camera: {W}×{H}')

    # ── Subsystems (always needed) ────────────────────────────────────────────
    spells    = SpellSystem(W, H, theme=chosen_theme)
    particles = ParticleSystem()
    hud       = HUDRenderer()
    fx        = ScreenEffects(W, H)
    audio     = AudioEngine() if not args.no_audio else None
    guide = ActionGuide()
    guide.setup(chosen_theme)

    # If Arcane Academy was chosen, run class selection UI before play
    class_overlay = None
    class_selector = None
    if chosen_theme.id == 'academy':
        class_selector = ClassSelectScreen()
        cfg = class_selector.run(W, H, spells.mastery)
        if cfg is None:
            # Player pressed ESC in class select — return to door selector
            chosen_theme = show_selection_screen()
            print(f'[AETHER] Theme: {chosen_theme.name}')
            # Recreate spells with new theme
            spells = SpellSystem(W, H, theme=chosen_theme)
        else:
            spells.set_class(cfg)
            # Build overlay with the same cards used by selector
            class_overlay = ClassSwitchOverlay(class_selector._cards)
            current_class_idx = 0

    # ── Subsystems per detection mode ─────────────────────────────────────────
    if is_one_piece:
        _load_one_piece_modules()
        pose_tracker  = _pose_tracker_mod.PoseTracker()
        pose_smoother = _pose_tracker_mod.PoseSmoother(alpha=0.50)
        body_detector = _body_detector_mod.BodyActionDetector()
        # Dummy hand tracker not needed; create lightweight trail for wrist
        tracker    = None
        trails     = [Trail(), Trail()]
        trail_rdr  = TrailRenderer()
        engines    = [ElementEngine(), ElementEngine()]
        sign_dets  = []
        detectors  = []
        vel_hist   = []
        prev_tip   = []
        print('[AETHER] One Piece mode — body skeleton tracking active!')
        print('[AETHER] Moves: Punch sideways=Pistol | Slam down=Cannon | '
              'Pump body=Gear2 | Hold arm out=Gear3 | Rapid alternating=Gatling')

    elif is_naruto:
        tracker       = HandTracker()
        trails        = [Trail(), Trail()]
        trail_rdr     = TrailRenderer()
        detectors     = []
        engines       = [ElementEngine(), ElementEngine()]
        sign_detector = NarutoSignDetector()   # real two-hand sign detector
        clone_fx      = CloneEffect()           # ghost clone visual
        vel_hist      = [deque(maxlen=_VEL_HISTORY), deque(maxlen=_VEL_HISTORY)]
        prev_tip      = [None, None]
        pose_tracker  = None
        body_detector = None
        print('[AETHER] Naruto mode — make the Shadow Clone cross sign with BOTH hands!')

    elif is_game:
        tracker    = HandTracker()
        trails     = [Trail(), Trail()]
        trail_rdr  = TrailRenderer()
        detectors  = []
        engines    = [ElementEngine(), ElementEngine()]
        vel_hist   = [deque(maxlen=_VEL_HISTORY), deque(maxlen=_VEL_HISTORY)]
        prev_tip   = [None, None]
        pose_tracker  = None
        body_detector = None
        sign_detector = NarutoSignDetector()
        clone_fx      = CloneEffect()
        fruit_game    = FruitGame(W, H)
        print('[AETHER] Fruit Basket — slash the fruits with your hand!')

    else:   # magic themes
        tracker    = HandTracker()
        trails     = [Trail(), Trail()]
        trail_rdr  = TrailRenderer()
        detectors  = [ShapeDetector(), ShapeDetector()]
        engines    = [ElementEngine(), ElementEngine()]
        vel_hist   = [deque(maxlen=_VEL_HISTORY), deque(maxlen=_VEL_HISTORY)]
        prev_tip   = [None, None]
        pose_tracker  = None
        body_detector = None
        sign_detector = NarutoSignDetector()
        clone_fx      = CloneEffect()
        fruit_game    = FruitGame(W, H)   # dummy

    show_hud   = True
    debug_mode = args.debug
    frame_dt   = 1.0 / config.TARGET_FPS
    fps_avg    = float(config.TARGET_FPS)
    _hum_on    = False
    _quit      = False

    # For displaying One Piece move name briefly on screen
    _last_op_move    = None
    _last_op_xy      = (W // 2, H // 2)
    _last_op_display = 0.0

    # ── EXIT button (top-right corner) ────────────────────────────────────────
    _BTN_W, _BTN_H = 80, 32
    _BTN_X = W - _BTN_W - 10
    _BTN_Y = 8

    def _on_mouse(event, x, y, flags, param):
        nonlocal _quit
        if event == cv2.EVENT_LBUTTONDOWN:
            if _BTN_X <= x <= _BTN_X + _BTN_W and _BTN_Y <= y <= _BTN_Y + _BTN_H:
                _quit = True

    cv2.namedWindow('AETHER SCRIPT')
    cv2.setMouseCallback('AETHER SCRIPT', _on_mouse)

    print('[AETHER] READY — ESC=quit  r=reset  g=HUD  d=debug  b=SpellBook')

    while True:
        t0 = time.perf_counter()

        ret, raw = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame   = cv2.flip(raw, 1)
        frame   = desaturate(frame, 0.70)
        overlay = np.zeros((H, W, 3), np.uint8)

        # ══════════════════════════════════════════════════════════════════════
        # ONE PIECE — rubber body visual only, no spells, no triggers
        # ══════════════════════════════════════════════════════════════════════
        if is_one_piece:
            pose_data = pose_tracker.process(frame)

            pose_data = pose_smoother.smooth(pose_data)   # remove jitter

            if pose_data is not None:
                # Green pulsing skeleton
                pulse = 0.78 + 0.22 * math.sin(time.time() * 3)
                pose_tracker.draw_skeleton(frame, pose_data, pulse=pulse)

                # Rubber stretch on BOTH arms — always follows naturally
                r_ratio, l_ratio = body_detector.arm_extensions(pose_data)
                pose_tracker.draw_rubber_stretch(frame, pose_data, 'right', r_ratio)
                pose_tracker.draw_rubber_stretch(frame, pose_data, 'left',  l_ratio)

                if audio and not _hum_on:
                    audio.start_hum()
                    _hum_on = True
            else:
                body_detector.reset()
                pose_smoother.reset()
                if audio and _hum_on:
                    audio.stop_hum()
                    _hum_on = False

        # ══════════════════════════════════════════════════════════════════════
        # NARUTO — hand sign pose detection (no trail shapes)
        # ══════════════════════════════════════════════════════════════════════
        # NARUTO — real two-hand sign detection, clone ghost effect, TTS voice
        # ══════════════════════════════════════════════════════════════════════
        elif is_naruto:
            hands        = tracker.process(frame)
            hand_present = len(hands) > 0

            # ── Draw hand skeleton in debug mode ──────────────────────────────
            if debug_mode and hands:
                tracker.draw_debug(overlay, hands)

            # ── Detect the sign using BOTH hands ─────────────────────────────
            triggered = sign_detector.update(hands)

            if triggered == 'shadow_clone':
                # 1. Play the MP3 from chakra folder
                play_jutsu('Kage Bushin No Jutsu.mp3')
                # 2. Start clone ghost visual
                clone_fx.trigger(frame)
                # 3. Screen flash
                fx.flash((255, 200, 100), duration=8)
                fx.chromatic_aberration(strength=5, duration=10)

            # ── Render clone ghosts onto the blended frame (done later) ──────
            # (clone_fx.render is called after composite below)

            # ── Show reference card + progress bar while sign is detected ────
            if sign_detector.active_sign:
                draw_sign_hud(
                    frame,
                    sign_detector.active_label,
                    sign_detector.hold_progress,
                    sign_detector.active_ref_img,
                    accent=(255, 180, 40),
                )

            if audio:
                if hand_present and not _hum_on:
                    audio.start_hum();  _hum_on = True
                elif not hand_present and _hum_on:
                    audio.stop_hum();   _hum_on = False

        # GAME — simple hand input for Fruit Basket
        # (must not fall through to the magic branch where detectors are required)
        elif is_game:
            hands        = tracker.process(frame)
            hand_present = len(hands) > 0
            if debug_mode and hands:
                tracker.draw_debug(overlay, hands)
            if audio:
                if hand_present and not _hum_on:
                    audio.start_hum();  _hum_on = True
                elif not hand_present and _hum_on:
                    audio.stop_hum();   _hum_on = False

        # ══════════════════════════════════════════════════════════════════════
        # MAGIC — draw trail shapes (original system)
        # ══════════════════════════════════════════════════════════════════════
        else:
            hands        = tracker.process(frame)
            hand_present = len(hands) > 0

            for idx, hand in enumerate(hands[:2]):
                tip   = hand['fingertip']
                thumb = hand['thumb_tip']
                h_sz  = hand['hand_size']

                if prev_tip[idx] is not None:
                    vel = float(np.linalg.norm(np.array(tip) - np.array(prev_tip[idx])))
                    vel_hist[idx].append(vel)
                    if len(vel_hist[idx]) >= 3:
                        hist    = list(vel_hist[idx])
                        avg_old = np.mean(hist[:-1])
                        if (avg_old > 0 and
                                hist[-1] > avg_old * _SPIKE_MULT and
                                hist[-1] > _SPIKE_ABS_MIN):
                            spells.on_release()
                prev_tip[idx] = tip

                trails[idx].add(tip)
                pts     = trails[idx].pts
                element = engines[idx].update(pts)

                shape = detectors[idx].detect(pts)
                if shape:
                    spells.on_shape(shape, element, pts)
                    if audio:
                        audio.play_charge()

                if is_pinching(thumb, tip, h_sz):
                    spells.on_release()

                trail_rdr.render(overlay, trails[idx], element,
                                 charge=spells.charge_frac)

                if debug_mode:
                    tracker.draw_debug(overlay, [hand])
                    draw_debug_corners(overlay, pts)
                    draw_debug_elem(frame, element, tip[0] + 12, max(20, tip[1] - 12))

            # Dual-hand Domain of Aether
            if len(hands) >= 2:
                t0h    = np.array(hands[0]['fingertip'])
                t1h    = np.array(hands[1]['fingertip'])
                spread = float(np.linalg.norm(t0h - t1h))
                if spread > W * _DUAL_SPREAD:
                    for idx in range(2):
                        if detectors[idx]._last_shape == 'circle':
                            spells.on_shape('circle_dual',
                                            engines[idx].current,
                                            trails[idx].pts)
                            fx.vortex(duration=100)
                            break

            for idx in range(len(hands), 2):
                trails[idx].on_lost()
                if len(trails[idx]) > 1:
                    trail_rdr.render(overlay, trails[idx],
                                     engines[idx].current, charge=0.0)
                vel_hist[idx].clear()
                prev_tip[idx] = None

            if audio:
                if hand_present and not _hum_on:
                    audio.start_hum();  _hum_on = True
                elif not hand_present and _hum_on:
                    audio.stop_hum();   _hum_on = False

        # ── Spell + particles (magic & naruto only) ───────────────────────────
        if not is_game:
            spells.update()
            particles.update()
            spells.render(overlay, particles)
            particles.render(overlay)

            ev = spells.release_event
            if ev:
                name, color, elem = ev
                fx.show_spell_name(name, color, element=elem, duration=85)
                fx.flash(color, duration=10)
                fx.chromatic_aberration(strength=4, duration=7)
                if spells.is_power_spell:
                    fx.shake(magnitude=14, duration=10)
                    fx.tint(color, duration=40)
                if elem == 'VOID':
                    fx.vortex(duration=75)
                if audio:
                    audio.play_release(elem)

        # ── Composite + effects ───────────────────────────────────────────────
        blended = cv2.add(frame, overlay)
        blended = fx.apply(blended)

        # Clone ghost overlay
        if is_naruto:
            clone_fx.render(blended)

        # ── GAME: fruit basket ────────────────────────────────────────────────
        if is_game:
            _game_tip = None
            if 'hands' in locals() and hands:
                _game_tip = hands[0]['fingertip']
            fruit_game.update(tip=_game_tip)
            fruit_game.render(blended)

        # ── HUD (magic sections only — not game) ─────────────────────────────
        if show_hud and not is_game:
            hud.render(blended, engines[0], spells, theme=chosen_theme)

        # ── Action guide (all sections) ───────────────────────────────────────
        guide.render(blended)
        if not guide.visible:
            cv2.putText(blended, '[B] Guide', (config.HUD_WIDTH + 10, H - 12),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.46, (80, 65, 95), 1, cv2.LINE_AA)

        draw_fps(blended, fps_avg)
        draw_exit_button(blended, _BTN_X, _BTN_Y, _BTN_W, _BTN_H)

        # Render class-switch overlay if present
        if class_overlay and class_overlay.visible:
            class_overlay.render(blended, highlight_idx=current_class_idx)

        cv2.imshow('AETHER SCRIPT', blended)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or _quit:
            break
        elif key == ord('r'):
            spells.reset()
            particles.clear()
            for t_ in trails:
                t_.clear()
            for e in engines:
                e.reset()
            for dtr in detectors:
                dtr.reset()
            if is_naruto:
                sign_detector.reset()
                clone_fx.active = False
            if is_game:
                fruit_game.reset()
            if body_detector:
                body_detector.reset()
            print('[AETHER] Reset.')
        elif key == ord('g'):
            show_hud = not show_hud
        elif key == ord('d'):
            debug_mode = not debug_mode
        elif key == ord('b'):
            guide.toggle()
        elif key == ord('c'):
            if class_overlay:
                class_overlay.visible = not class_overlay.visible
        elif class_overlay:
            idx = class_overlay.handle_input(key)
            if idx is not None and 0 <= idx < len(class_overlay.classes):
                new_cfg = class_overlay.classes[idx]
                spells.set_class(new_cfg)
                current_class_idx = idx
        elif key == 32 and not is_naruto and not is_one_piece:
            spells.on_release()

        # ── Frame pacing ──────────────────────────────────────────────────────
        elapsed   = time.perf_counter() - t0
        fps_avg   = 0.92 * fps_avg + 0.08 * (1.0 / max(elapsed, 1e-6))
        remaining = frame_dt - elapsed
        if remaining > 0.003:
            time.sleep(remaining)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if tracker:
        tracker.release()
    if pose_tracker:
        pose_tracker.release()
    cv2.destroyAllWindows()
    if audio:
        audio.cleanup()
    print('[AETHER] Closed.')


if __name__ == '__main__':
    main()
