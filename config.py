"""AETHER SCRIPT — Central configuration for all modules."""

# ── Display ───────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30

# ── Colors (BGR for OpenCV) ───────────────────────────────────────────────────
GOLD      = (0, 215, 255)      # #FFD700
NEON_BLUE = (255, 191, 0)      # #00BFFF
VOID_DARK = (21, 0, 10)        # #0A0015
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)

ELEMENT_COLORS = {
    'WATER':   (255, 255,   0),   # cyan    #00FFFF
    'FIRE':    (  0,  69, 255),   # orange  #FF4500
    'LIGHT':   (205, 250, 255),   # ivory   #FFFACD
    'VOID':    (255,   0, 123),   # purple  #7B00FF
    'NEUTRAL': (  0, 215, 255),   # gold
}

ELEMENT_ICONS = {
    'WATER':   '~ WATER',
    'FIRE':    '^ FIRE',
    'LIGHT':   '* LIGHT',
    'VOID':    'O VOID',
    'NEUTRAL': '? NEUTRAL',
}

ELEMENT_LIST = ['WATER', 'FIRE', 'LIGHT', 'VOID']

# ── Trail ─────────────────────────────────────────────────────────────────────
TRAIL_MAX_LEN     = 50
TRAIL_MAX_WIDTH   = 10
TRAIL_FADE_FRAMES = 12

# ── Shape detection ───────────────────────────────────────────────────────────
# Intent guards — finger must have actually moved before any shape is tested
MIN_GESTURE_DISTANCE = 160  # total path length in pixels (filters idle tremors)
MIN_GESTURE_BBOX     = 80   # bounding-box max dimension in pixels

CIRCLE_STD_RATIO  = 0.28
CIRCLE_MIN_PTS    = 40
CIRCLE_CLOSE_PX   = 80
CIRCLE_MIN_RADIUS = 35      # raised: avoid tiny accidental circles

SPIRAL_MIN_PTS  = 40
SPIRAL_MIN_REVS = 0.70

LINE_MIN_PTS    = 20        # raised: require more points for a line
LINE_PERP_RATIO = 0.15      # tighter: only very straight lines qualify

CORNER_WINDOW  = 6
CORNER_MIN_ANG = 0.60       # raised slightly so gentle curves aren't corners
CORNER_MIN_GAP = 18
CORNER_SEG_MIN = 30         # raised: each triangle/square side must be longer

# ── Element inference ─────────────────────────────────────────────────────────
ELEM_WINDOW  = 25           # recent frames to analyse
VEL_SMOOTH   = 2.5
VEL_FAST     = 10.0
ACCEL_SPIKE  = 4.0
CURV_LOW     = 0.06
CURV_HIGH    = 0.32
VEL_STD_LOW  = 1.5
VEL_STD_HIGH = 5.0

# ── Spell system ──────────────────────────────────────────────────────────────
SHAPE_COOLDOWN   = 2.8      # seconds before a new shape can fire
SPELL_TIMEOUT    = 3.5      # seconds before spell buffer is cleared
OVERLOAD_SECS    = 4.0      # auto-release after this many charging seconds
DECAY_DURATION   = 1.5      # seconds for DECAY phase
RELEASE_DURATION = 0.35     # seconds for RELEASE flash phase
PINCH_RATIO      = 0.14     # thumb-index distance / hand_size threshold

# ── Particles ─────────────────────────────────────────────────────────────────
MAX_PARTICLES       = 600
PARTICLES_PER_SPELL = 120

# ── HUD ───────────────────────────────────────────────────────────────────────
HUD_WIDTH  = 270
HUD_ALPHA  = 0.82
HUD_BG_BGR = (18, 3, 10)

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100
HUM_FREQ       = 220.0
SWEEP_END_FREQ = 880.0
