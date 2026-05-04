"""
Theme / loadout definitions.

Each theme locks the player into a set of elements so spell variety
matches a chosen class fantasy.  The 'id' field drives theme-specific
spell map lookups in SpellSystem.
"""
from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class Theme:
    id: str                             # internal key used by spell system
    key: str                            # keyboard shortcut '1'–'7'
    name: str
    tagline: str
    element_label: str                  # short ASCII label shown in UI
    primary_element: str                # fallback when detected element is blocked
    allowed_elements: FrozenSet[str]
    accent: Tuple[int, int, int]        # BGR accent colour
    bg_dark: Tuple[int, int, int]       # BGR dark background for selection card
    spells: Tuple[str, ...]             # display list of available spells
    category: str = 'magic'            # 'magic' | 'anime'


THEMES = (
    # ── Arcane Academy ───────────────────────────────────────────────────────
    Theme(
        id='academy',
        key='1',
        name='Arcane Academy',
        tagline='Choose your elemental class',
        element_label='CLASS',
        primary_element='NEUTRAL',
        allowed_elements=frozenset({'FIRE', 'WATER', 'LIGHT', 'VOID', 'NEUTRAL'}),
        accent=(200, 180, 255),
        bg_dark=(15, 5, 25),
        category='magic',
        spells=(
            'Arcane Ward', 'Ignition Pulse', 'Tide Blade', 'Prism Lance',
            'Shade Rend', 'Arcane Construct', 'Void Collapse', 'Maelstrom',
            'Radiant Pulse', 'Reality Slice', 'Flash Strike', 'Domain of Aether',
        ),
    ),

    # ── Anime Worlds ──────────────────────────────────────────────────────────
    Theme(
        id='naruto',
        key='6',
        name='Konoha Ninja',
        tagline='Show them your ninja way.',
        element_label='CHAKRA',
        primary_element='NEUTRAL',
        allowed_elements=frozenset({'FIRE', 'WATER', 'LIGHT', 'VOID', 'NEUTRAL'}),
        accent=(80, 120, 255),          # chakra blue-orange
        bg_dark=(8, 4, 20),
        category='anime',
        spells=(
            '-- HAND SIGNS (hold 1.2s) --',
            'Ram  (Ne)   → Kage Bunshin',
            'Bird (Tori) → Chidori',
            'Tiger(Tora) → Fire Jutsu',
            'Boar (I)    → Rasengan',
            'Snake(Mi)   → Shadow Bind',
            '-- DRAW SHAPES (trail) --',
            'Circle  → Rasengan',
            'Line    → Chidori',
            'Triangle→ Fire Jutsu',
            'Spiral  → Kage Bunshin',
        ),
    ),
    Theme(
        id='one_piece',
        key='7',
        name='Devil Fruit User',
        tagline="I'm gonna be King of the Pirates!",
        element_label='RUBBER',
        primary_element='NEUTRAL',
        allowed_elements=frozenset({'FIRE', 'WATER', 'LIGHT', 'VOID', 'NEUTRAL'}),
        accent=(0, 30, 220),            # Luffy red
        bg_dark=(5, 2, 20),
        category='anime',
        spells=(
            '-- DRAW SHAPES (trail) --',
            'Circle   → Gomu Gomu no Cannon',
            'Line     → Gomu Gomu no Pistol',
            'Triangle → Gear 3rd: King Kong Gun',
            'Spiral   → Gear Second',
            'Square   → Elephant Gatling',
        ),
    ),
    Theme(
        id='game',
        key='8',
        name='Fruit Basket',
        tagline='Slice everything. Miss nothing.',
        element_label='GAME',
        primary_element='NEUTRAL',
        allowed_elements=frozenset({'NEUTRAL'}),
        accent=(0, 220, 80),            # green
        bg_dark=(5, 20, 5),
        category='game',
        spells=(
            'Slash fruits with your hand!',
            'Miss a fruit → lose a life',
            'Hit a BOMB → lose a life',
            'Reach 150 pts to WIN',
            'Chain slashes for COMBO bonus',
        ),
    ),
)

THEME_MAP    = {t.key: t for t in THEMES}
THEME_BY_ID  = {t.id:  t for t in THEMES}
DEFAULT_THEME = THEMES[0]
