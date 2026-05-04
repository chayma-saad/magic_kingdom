"""
Audio Engine — procedurally generated tones via NumPy + pygame.mixer.
Gracefully disabled if pygame is unavailable.
"""
import numpy as np
import config

try:
    import pygame
    import pygame.mixer as _mixer
    _mixer.pre_init(config.SAMPLE_RATE, -16, 1, 512)
    pygame.init()
    _mixer.init()
    _PYGAME = True
except Exception:
    _PYGAME = False


def _to_sound(wave_f32):
    """Convert a float32 mono wave (values in [-1, 1]) to a pygame Sound."""
    if not _PYGAME:
        return None
    pcm = np.clip(wave_f32 * 32767, -32768, 32767).astype(np.int16)
    # Duplicate to stereo if the mixer opened in stereo mode
    if pygame.mixer.get_init()[2] == 2:
        pcm = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(pcm)


def _sine(freq, duration, amp=0.35):
    t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _sweep(f0, f1, duration, amp=0.4):
    n     = int(config.SAMPLE_RATE * duration)
    t     = np.linspace(0, duration, n, False)
    freqs = np.linspace(f0, f1, n)
    phase = np.cumsum(2 * np.pi * freqs / config.SAMPLE_RATE)
    return (np.sin(phase) * amp).astype(np.float32)


def _impact(duration=0.4, amp=0.55):
    n        = int(config.SAMPLE_RATE * duration)
    t        = np.linspace(0, duration, n, False)
    noise    = np.random.randn(n).astype(np.float32)
    envelope = np.exp(-t * 12).astype(np.float32)
    return (noise * envelope * amp).astype(np.float32)


class AudioEngine:
    def __init__(self):
        self.enabled    = _PYGAME
        self._hum_ch    = None    # looping channel
        self._sounds    = {}
        if self.enabled:
            self._build_sounds()

    def _build_sounds(self):
        self._sounds['hum']     = _to_sound(_sine(config.HUM_FREQ,     1.5, 0.20))
        self._sounds['charge']  = _to_sound(_sweep(config.HUM_FREQ,
                                                   config.SWEEP_END_FREQ, 2.5, 0.35))
        self._sounds['release'] = _to_sound(_impact(0.45, 0.55))
        self._sounds['void']    = _to_sound(_sweep(440, 110, 1.0, 0.3))

    # ── Public API ────────────────────────────────────────────────────────────
    def start_hum(self):
        if not self.enabled:
            return
        snd = self._sounds.get('hum')
        if snd and self._hum_ch is None:
            self._hum_ch = snd.play(-1)   # loop indefinitely

    def stop_hum(self):
        if not self.enabled or self._hum_ch is None:
            return
        snd = self._sounds.get('hum')
        if snd:
            snd.stop()
        self._hum_ch = None

    def play_charge(self):
        self._play('charge')

    def play_release(self, element='NEUTRAL'):
        key = 'void' if element == 'VOID' else 'release'
        self._play(key)

    def _play(self, key):
        if not self.enabled:
            return
        snd = self._sounds.get(key)
        if snd:
            snd.play()

    def cleanup(self):
        if self.enabled:
            _mixer.quit()
            pygame.quit()
