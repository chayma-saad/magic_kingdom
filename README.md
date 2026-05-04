# AETHER SCRIPT
### Real-Time Computer Vision Spellcasting System

> Draw magic spells in the air with your bare hands — your webcam sees your gestures, classifies them into shapes, reads the energy of your motion, and renders cinematic anime-style spell effects live on screen.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [How to Run](#how-to-run)
4. [Controls](#controls)
5. [How to Cast Spells](#how-to-cast-spells)
6. [All Spells & Effects](#all-spells--effects)
7. [Combo Spells](#combo-spells)
8. [The Four Elements](#the-four-elements)
9. [The Grimoire HUD](#the-grimoire-hud)
10. [Project File Structure](#project-file-structure)
11. [Tools & Technologies Explained](#tools--technologies-explained)

---

## Requirements

- **Windows 10 / 11** (64-bit)
- **Python 3.10 or higher**
- **Webcam** (built-in or USB)
- **~500 MB disk space** for packages

---

## Installation

### Step 1 — Install Python (if not already installed)

Download Python 3.11 from [python.org](https://www.python.org/downloads/) and run the installer.
**Important:** check the box that says **"Add Python to PATH"** before clicking Install.

Or install silently via Windows Package Manager:
```
winget install Python.Python.3.11
```

### Step 2 — Install project dependencies

Open a terminal, navigate to the project folder, and run:

```bash
cd "c:\Users\2026\Desktop\AI Magic system\aether_script"
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥ 4.8 | Webcam capture and all image rendering |
| mediapipe | ≥ 0.10 | Hand landmark detection (21 points per hand) |
| numpy | ≥ 1.24 | Math for trails, particles, shape detection |
| pygame | ≥ 2.5 | Audio engine for procedural spell sounds |
| Pillow | ≥ 10.0 | Optional high-quality TTF font rendering in HUD |

### One-click alternative

Double-click **`install_and_run.bat`** — it finds Python automatically, installs all packages, and launches the app.

---

## How to Run

### Normal mode
```bash
cd "c:\Users\2026\Desktop\AI Magic system\aether_script"
python main.py
```

### Debug mode — shows hand skeleton + shape detection corners
```bash
python main.py --debug
```

### Silent mode — disables all audio
```bash
python main.py --no-audio
```

### Using the batch file
```
Double-click  install_and_run.bat
```

---

## Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit the application |
| `r` | Reset — clear all spells, trails, and particles |
| `g` | Toggle the Grimoire HUD sidebar on/off |
| `d` | Toggle debug overlay (skeleton + corner markers) |
| `SPACE` | Force-release the currently charging spell instantly |

---

## How to Cast Spells

Spellcasting happens in four stages:

```
DRAW A SHAPE  →  CHARGING  →  RELEASE  →  DECAY
```

### Stage 1 — Draw a Shape
Hold your hand in front of the webcam and move your **index finger** to draw a shape in the air. A glowing trail follows your fingertip. Draw slowly and deliberately for best recognition.

### Stage 2 — Charging
When the system recognizes your shape, the spell enters the **CHARGING** state. You will see:
- The spell's geometric symbol appear glowing on screen
- The HUD shows the spell name and a charging bar
- The fingertip orb grows larger and pulses faster
- Ambient sound begins building

### Stage 3 — Release
Trigger the release using any of these methods:
- **Pinch** your thumb and index finger together
- **Flick** your wrist fast (velocity spike detection)
- Press **SPACE** on the keyboard
- Wait for **4 seconds** — the spell auto-releases at overload

On release: particle burst, screen flash, spell name appears center-screen.

### Stage 4 — Decay
The spell effect fades out over 1.5 seconds, then you are ready to cast again.

## Stages of Casting (Full Details)

This section expands the basic flow with the internal stages and triggers the system uses.

1. Idle / Ready
	- No current trail or charge. HUD shows '[Ready]'. Audio hum is idle. System monitors MediaPipe hands for input.

2. Draw / Trace
	- Fingertip landmark (index tip) creates a `Trail` of points. The trail renderer displays a glowing line.
	- The shape detector continuously evaluates the trail for geometric patterns (circle, line, triangle, square, spiral).
	- The element engine analyzes motion (velocity, curvature, variance) but does not commit until the shape is stable.

3. Shape Confirmed → Enter Charging
	- When a shape is recognized with sufficient confidence, the detector sets a `charging` state for that hand.
	- Visuals: symbol appears, charge bar starts, fingertip orb pulses. Audio transitions to a charging tone.
	- The system applies a short hysteresis window (several frames) to prevent flicker between shapes.

4. Charging (Hold) — Build Power
	- Charge increases over time while the hand remains present and the shape is maintained.
	- The HUD shows the spell name, current element, and a numeric/visual charge fraction. Overload threshold at ~4s.
	- The user may abort by releasing the hand or by clearing the trail (moving hand out of view).

5. Release (Multiple Triggers)
	- Manual triggers:
	  - Pinch (thumb + index within normalized distance)
	  - Velocity spike (fast flick detected by comparing recent fingertip velocity history)
	  - Keyboard `SPACE` (force-release)
	- Automatic triggers:
	  - Overload timeout (auto-release when charge exceeds max)
	  - Dual-hand combos (certain dual shapes trigger special combined releases)

6. Spell Execution
	- On release the `SpellSystem` spawns particles, screen effects (flash, shake, tint), and plays release audio.
	- Spell-specific visuals are rendered for a short duration; combo logic may spawn additional effects.

7. Decay / Cooldown
	- Visuals fade (particles and overlays decay). The spell state machine resets to Idle for that hand.
	- Some spells impose a short cooldown before the next cast to avoid accidental re-triggering.

8. Post-Processing & Global Effects
	- After the spell finishes, `ScreenEffects` may play extra long-duration overlays (vortex, tint), and mastery XP is awarded.

Notes:
- The system tracks up to two hands independently; each hand has its own `Trail`, `ShapeDetector`, `ElementEngine`, and charge state.
- Dual-hand mechanics (Domain of Aether, combos) are detected when both hands' trails meet specific spatial criteria.
- Tuning parameters (hysteresis frames, pinch ratio, overload timeout) are in `config.py` for easy adjustment.

---

## All Spells & Effects

### CIRCLE → Arcane Ward
**Element:** Any  
**How to draw:** Trace a complete circle in the air  
**Visual effect:**  
- Three concentric rings pulse outward from the circle center  
- Six rotating rune dots orbit the perimeter, spinning faster as the spell charges  
- A bright glowing core grows at the center  
- On release: the rings shatter outward in a ring-expand burst + radial particle explosion  
**Best element:** LIGHT for a white-gold radiant ward

---

### TRIANGLE → Ignition Pulse
**Element:** FIRE (strongest), any element changes the color  
**How to draw:** Three sharp directional strokes forming a triangle  
**Visual effect:**  
- A glowing triangle shape drawn from your trail's three corners  
- The triangle breathes outward as charge builds  
- Fire particles constantly spray upward during charging  
- On release: each vertex explodes outward with a burst of particles + lightning bolt  
**Variants:** Tide Blade (WATER), Prism Lance (LIGHT), Shade Rend (VOID), Sigil Strike (NEUTRAL)

---

### SQUARE → Arcane Construct
**Element:** Any  
**How to draw:** Four strokes forming a square or rectangle  
**Visual effect:**  
- A rotating outer square and inner counter-rotating square materialize  
- Four connecting struts link the two squares, creating a wireframe construct  
- The whole structure grows and rotates faster as charge builds  
- On release: each corner fires an outward particle burst  

---

### SPIRAL → Void Collapse
**Element:** VOID (strongest)  
**How to draw:** Slowly spiral inward or outward, making at least 3/4 of a full rotation  
**Visual effect:**  
- Particles continuously spiral inward toward the center point  
- A filled dark void core grows at the center  
- A deep purple pulsing ring surrounds the void  
- On release: a massive inward vortex pulls 180 particles toward the center, the screen flashes with void tint  
**Variants:** Maelstrom (WATER), Firestorm (FIRE), Radiant Pulse (LIGHT), Ether Spiral (NEUTRAL)

---

### STRAIGHT LINE → Reality Slice
**Element:** Any  
**How to draw:** A fast straight slash in any direction  
**Visual effect:**  
- A bright energy blade rendered along your trail line  
- White hot core with element-colored outer glow  
- On release: particles scatter from the blade with perpendicular spread + lightning bolt along the blade  
**Variants:** Flash Strike (LIGHT), Torrent Slash (WATER)

---

### CIRCLE (both hands spread wide) → Domain of Aether
**Element:** Any  
**How to activate:** Hold both hands visible to the camera spread far apart, then draw a circle with one hand  
**Visual effect:**  
- The entire screen fills with a golden tint  
- Two expanding ring pulses radiate from the screen center  
- A continuous particle storm fills the screen  
- A rotating vortex of spokes and rings overlays the whole frame  
- On release: 250 screen-filling particles + 150-particle radial burst from center  
**Note:** The most powerful spell — triggers maximum screen shake and vortex effect

---

### SPIRAL → Maelstrom
**Element:** WATER  
**Visual effect:**  
- Particles continuously orbit the spiral center  
- A large circular glow ring expands and contracts  
- On release: 150 particles arc outward in a full circle shower

---

### SPIRAL → Radiant Pulse
**Element:** LIGHT  
**Visual effect:**  
- Four concentric rings of light radiate outward  
- On release: eight beams of particles fire outward like spokes of a wheel

---

### SPIRAL → Firestorm
**Element:** FIRE  
**Visual effect:**  
- Same as Ignition Pulse but centered at the spiral  
- On release: vertex explosions from three triangle points

---

### LINE → Prism Lance
**Element:** LIGHT  
**Visual effect:**  
- Three parallel beams of light (element color, white, neon blue) fire toward the right edge  
- On release: full beam of particles across the screen + lightning bolt

---

## Combo Spells

Draw two shapes **within 3.5 seconds** of each other to unlock combo spells. The HUD combo tracker shows your current chain.

| Shape Sequence | Combo Spell | Effect |
|---------------|-------------|--------|
| Circle → Triangle | **Exploding Ward** | A ward that charges then fully shatters — double ring expansion + white flash burst |
| Spiral → Line | **Void Beam** | A sustained dark energy beam fires in a fixed direction with lightning strike and beam particles |
| Square → Circle | **Barrier Field** | A semi-circular dome expands upward, arc particles rain along its edge |
| Triangle → Circle | **Nova Burst** | Combined ring + cross explosion — 200 particles + two lightning bolts at right angles |
| Line → Circle | **Convergence** | Lines stream in from all directions converging at the center, then a radial explosion |

---

## The Four Elements

The system reads **how** you draw — not just what shape — to determine the element automatically. No button presses needed.

| Element | Color | Motion Pattern | Particles |
|---------|-------|---------------|-----------|
| **WATER** | Cyan `#00FFFF` | Smooth, flowing curves. Low velocity variance, moderate speed | Soft ripples, orbiting drops |
| **FIRE** | Orange-Red `#FF4500` | Sharp accelerations, angular direction changes | Ember bursts, upward spray |
| **LIGHT** | White-Gold `#FFFACD` | Slow, deliberate, precise strokes. Very low curvature variance | Radial rays, prismatic beams |
| **VOID** | Deep Purple `#7B00FF` | Erratic, chaotic, high variance in both speed and direction | Inward vortex, dark distortion |
| **NEUTRAL** | Gold `#FFD700` | Does not match any of the above | Default golden particles |

The element classifier uses a **hysteresis gate** — the detected element must hold for 5 consecutive frames before switching, preventing flickering between elements during a single gesture.

---

## The Grimoire HUD

Press `g` to toggle the sidebar panel. It displays:

```
┌──────────────────────┐
│  A E T H E R         │
│  S C R I P T         │
├──────────────────────┤
│  [Active Spell Name] │
│  CHARGING ████░░░░░  │  ← charge bar
├──────────────────────┤
│  ELEMENT             │
│  ● ~ WATER           │  ← pulsing dot
├──────────────────────┤
│  RECENT SPELLS       │
│  > Arcane Ward       │
│  > Reality Slice     │
├──────────────────────┤
│  MASTERY             │
│  WAT ████████ 40     │
│  FIR ██░░░░░░ 10     │
│  LIG ░░░░░░░░  0     │
│  VOI ████░░░░ 20     │
├──────────────────────┤
│  COMBO               │
│  CIRCLE → TRIANGLE   │
└──────────────────────┘
```

**Mastery** increases by 5 points every time you successfully cast a spell of that element (max 100).

**The border pulses** in the active element's color while a spell is charging.

**OVERLOAD** warning (red pulsing border + text) appears if you hold a charge for more than 4 seconds without releasing.

---

## Spells by Theme (All Doors / Programs)

This lists the spells and abilities available for each selection "door" (theme) shown on the selection screen.

- **Arcane Master (All elements)**
	- Arcane Ward, Ignition Pulse, Tide Blade, Prism Lance, Shade Rend,
		Arcane Construct, Void Collapse, Maelstrom, Radiant Pulse, Reality Slice,
		Flash Strike, Domain of Aether
	- Combo/Advanced: Exploding Ward, Void Beam, Barrier Field, Nova Burst, Convergence

- **Pyromancer (FIRE)**
	- Ignition Pulse, Firestorm, Reality Slice, Arcane Construct, Arcane Ward
	- Combo: Exploding Ward, Nova Burst

- **Tide Caller (WATER)**
	- Tide Blade, Torrent Slash, Maelstrom, Arcane Ward, Arcane Construct
	- Combo: Barrier Field, Convergence

- **Void Walker (VOID)**
	- Void Collapse, Shade Rend, Reality Slice, Ether Spiral, Arcane Ward
	- Combo: Void Beam, Exploding Ward

- **Sage (LIGHT)**
	- Prism Lance, Flash Strike, Radiant Pulse, Arcane Ward, Arcane Construct
	- Combo: Barrier Field, Convergence

- **Konoha Ninja (Naruto world)**
	- Hand signs (hold ~1.2s): Ram → Kage Bunshin, Bird → Chidori, Tiger → Fire Jutsu,
		Boar → Rasengan, Snake → Shadow Bind
	- Draw shapes (trail): Circle → Rasengan, Line → Chidori, Triangle → Fire Jutsu,
		Spiral → Kage Bunshin

- **Devil Fruit User (One Piece world)**
	- Draw shapes map to Luffy moves: Circle → Cannon, Line → Pistol,
		Triangle → Gear 3rd (King Kong Gun), Spiral → Gear Second, Square → Gatling

- **Fruit Basket (Game mode)**
	- Gameplay rules rather than spells: slice fruits, avoid bombs, score and combos

---

## Project File Structure

```
aether_script/
│
├── main.py              Main loop — camera, input handling, compositing
├── config.py            All constants: colors, thresholds, timings
│
├── hand_tracker.py      MediaPipe wrapper — extracts fingertip + thumb positions
├── trail.py             Trail deque + glow renderer + fingertip orb + sparkles
├── shape_detector.py    Geometric classifiers for 5 shapes using linear algebra
├── element_engine.py    Motion analysis → element inference with hysteresis
│
├── spell_system.py      Spell state machine + 15 spell visual renderers
├── particles.py         Vectorised NumPy particle pool (burst/orbit/vortex/beam/lightning)
├── effects.py           Screen-level post-processing (flash/shake/name overlay/vortex)
├── hud.py               Grimoire sidebar UI renderer
├── audio.py             Procedural tone synthesis via NumPy + pygame.mixer
│
├── requirements.txt     pip dependencies
└── install_and_run.bat  One-click installer and launcher for Windows
```

---

## Tools & Technologies Explained

### Python 3.11
The core language. Python 3.10+ is required for the `X | Y` union type syntax used in type hints throughout the codebase.

---

### OpenCV (`opencv-python`)
**What it is:** An open-source computer vision library originally developed by Intel, now maintained by the OpenCV Foundation.

**How we use it:**
- `cv2.VideoCapture(0)` — opens the default webcam and reads frames at up to 30fps
- `cv2.flip(frame, 1)` — mirrors the frame horizontally so it acts like a mirror, making gesture drawing intuitive
- `cv2.cvtColor()` — converts between BGR, RGB, and HSV color spaces (MediaPipe needs RGB; OpenCV works in BGR)
- `cv2.line()`, `cv2.circle()`, `cv2.polylines()` — draw the trail, orb, spell shapes, and HUD elements
- `cv2.GaussianBlur()` — the key to the bloom/glow effect: draw an element, blur it, then blend it back
- `cv2.addWeighted()` — alpha blending for the HUD panel and flash effects
- `cv2.imshow()` — renders the final composited frame in the display window
- `cv2.warpAffine()` — applies the screen-shake transform by translating the whole frame

**The glow effect recipe:**
```
1. Draw shape onto blank layer A
2. Gaussian blur layer A → layer B  
3. overlay = overlay + layer_A + (layer_B × 0.5)
```
This creates a soft light bloom around every drawn element.

---

### MediaPipe (`mediapipe`)
**What it is:** A cross-platform ML framework by Google for real-time perception pipelines. The Hands solution uses a two-step neural network: a palm detector followed by a landmark model.

**How we use it:**
- Detects up to **2 hands** simultaneously each frame
- Returns **21 landmarks per hand** — each is an `(x, y, z)` normalized coordinate
- We use **landmark 8** (index fingertip) as the "drawing point" for the trail
- We use **landmark 4** (thumb tip) to detect the pinch gesture
- **Landmark 0** (wrist) and **landmark 9** (middle finger MCP) give us `hand_size` — the distance between them is used to normalize the pinch threshold so it works regardless of how close the hand is to the camera

The hand tracker runs on the CPU in real-time at 30fps with minimal lag.

---

### NumPy (`numpy`)
**What it is:** The fundamental package for numerical computing in Python. Provides N-dimensional arrays and hundreds of math operations that run in compiled C code — far faster than Python loops.

**How we use it — everywhere:**

**Trail math:**
```python
# Compute distances from centroid for circle detection
centroid = arr.mean(axis=0)
distances = np.linalg.norm(arr - centroid, axis=1)
```

**Shape detection (PCA for line detection):**
```python
_, s, _ = np.linalg.svd(centered, full_matrices=False)
# If s[1]/s[0] is tiny → points lie on a line
```

**Spiral detection (unwrapped angles):**
```python
angles = np.arctan2(delta[:, 1], delta[:, 0])
angles_unwrapped = np.unwrap(angles)   # handles 2π jumps
total_rotation = abs(angles_unwrapped[-1] - angles_unwrapped[0])
```

**Particle system (vectorised, no Python loops for physics):**
```python
self.pos[alive] += self.vel[alive]          # move all alive particles at once
self.vel[alive, 1] += self.grav[alive]      # gravity in one operation
self.life[alive] -= 1.0                     # age all particles at once
```

**Audio synthesis:**
```python
t = np.linspace(0, duration, n_samples)
wave = np.sin(2 * np.pi * freq * t) * amplitude
```

---

### pygame (`pygame`)
**What it is:** A Python library for writing games and multimedia applications, built on SDL 2.

**How we use it:**
- `pygame.mixer` — the audio subsystem. We use it to play procedurally generated sound clips
- `pygame.sndarray.make_sound(array)` — converts a NumPy int16 array directly into a playable Sound object
- `sound.play(-1)` — loops a sound indefinitely (used for the ambient hand-detected hum)
- `sound.stop()` — stops the looping hum when the hand disappears

**Procedural audio generation** (no audio files required):
```
Ambient hum   →  220 Hz sine wave, loops while hand is visible
Charge build  →  220 Hz → 880 Hz frequency sweep over 2.5 seconds
Spell release →  Decaying noise burst (impact sound)
Void release  →  880 Hz → 110 Hz downward sweep (sinister tone)
```

---

### Pillow / PIL (`Pillow`)
**What it is:** The Python Imaging Library fork — the standard image processing library for Python.

**How we use it:**
- Renders HUD text using **TrueType fonts** (`.ttf`) from the Windows system font directory
- Falls back to OpenCV's built-in bitmap fonts if no TTF is found
- Pillow produces much sharper, anti-aliased text compared to `cv2.putText()`

Font search order: `consola.ttf` → `lucon.ttf` → `cour.ttf` → `arial.ttf`

---

### Collections (`deque`)
**What it is:** Part of Python's standard library. A `deque` (double-ended queue) with a `maxlen` argument automatically discards the oldest element when a new one is added — perfect for a fixed-length trail buffer.

**How we use it:**
```python
self._pts = deque(maxlen=60)   # always keeps only the last 60 fingertip positions
```
This gives O(1) append and automatic tail-dropping — no manual list slicing needed.

---

### The Rendering Pipeline (how a frame is built)

Every frame goes through this exact sequence:

```
1.  cap.read()              Raw BGR frame from webcam
2.  cv2.flip(frame, 1)      Mirror horizontally
3.  desaturate(frame, 0.70) Reduce color saturation 30% so magic pops
4.  tracker.process(frame)  MediaPipe detects hands → landmark positions
5.  overlay = zeros(H,W,3)  Black canvas for all magic rendering
6.    trail_rdr.render()     Draw glowing trail + orb onto overlay
7.    spells.render()        Draw spell geometry + spawn particles onto overlay
8.    particles.render()     Draw all living particles onto overlay
9.  cv2.add(frame, overlay)  Additive blend: magic light adds to webcam image
10. effects.apply(blended)   Screen flash, shake, spell name text, vortex
11. hud.render(blended)      Draw Grimoire sidebar panel on top
12. cv2.imshow(...)          Display final composed frame
```

The **additive blend** at step 9 (`cv2.add`) is what makes the magic look like real light — bright spell colors add to the webcam image rather than replacing it, so the magic literally illuminates the scene.

---

## Troubleshooting

**Black screen / camera not opening**
- Make sure no other application (Teams, Zoom, OBS) is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py` if you have multiple cameras

**Hands not detected**
- Ensure good lighting — MediaPipe needs to see your hand clearly
- Keep your hand 40–80 cm from the camera
- A plain background (not cluttered) improves detection confidence

**Low FPS**
- Close other applications to free CPU
- MediaPipe runs on CPU; a modern laptop should hold 25–30 fps

**No sound**
- Run with `--no-audio` flag if pygame audio causes errors
- Some systems need audio drivers updated for SDL to initialize

**Shape detection not triggering**
- Draw shapes slowly and deliberately — fast scribbles confuse the classifier
- Make shapes large (at least 100px radius) to exceed the minimum point thresholds
- After a shape is detected there is a 1.8-second cooldown before the next one fires

---

*AETHER SCRIPT — built with Python, OpenCV, MediaPipe, NumPy, pygame, and Pillow*
