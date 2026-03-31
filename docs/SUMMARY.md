# CV_to_StreetFighter - Technical Summary

A real-time gesture-controlled fighting game that uses computer vision to translate player body movements into game controls for a Street Fighter-style fighting game.

## Project Overview

**Purpose:** Capture player poses via webcam, classify movements into fighting game actions, and control a PyGame-based fighting game engine in real-time.

**Core Technologies:**
- **MediaPipe** - Real-time pose estimation (33 body landmarks)
- **PyTorch** - LSTM/FC neural networks for action classification
- **PyGame + PyOpenGL** - 2D fighting game with 3D rendering effects
- **OpenCV** - Webcam capture and video processing

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────┐     ┌──────────────┐     ┌─────────────────┐
    │  Webcam  │────▶│  OpenCV      │────▶│  MediaPipe Pose │
    │          │     │  Capture     │     │  Detection      │
    └──────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  33 Body Points │
                                          │  (x, y, z, vis) │
                                          └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────┐
                    │                              │                      │
                    ▼                              ▼                      ▼
           ┌─────────────────┐           ┌─────────────────┐       ┌──────────┐
           │ 21 Filtered     │           │ Movement        │       │ PoseViewer│
           │ Landmarks       │           │ Detection       │       │ (Debug)   │
           └────────┬────────┘           └────────┬────────┘       └──────────┘
                    │                             │
                    ▼                             ▼
           ┌─────────────────┐           ┌─────────────────┐
           │ LSTM/FC Model   │           │ Edge Zone       │
           │ (PyTorch)       │           │ Detection       │
           └────────┬────────┘           └────────┬────────┘
                    │                             │
                    └──────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Action Label      │
                    │  (14 classes)      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  InputDevice       │
                    │  (external_mode_2) │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Game      │     │   OpenGL    │     │    Audio    │
   │   Logic     │────▶│   Renderer  │────▶│   System    │
   └─────────────┘     └─────────────┘     └─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Game Display     │
                    │   (60 FPS)         │
                    └─────────────────────┘
```

---

## Computer Vision Pipeline

### PoseWorker (`ComputerVision/pose_worker.py`)

Threaded webcam capture with pose detection for non-blocking game performance.

**Key Responsibilities:**
- Runs in separate thread to maintain 60 FPS game loop
- Captures frames via OpenCV (default: 30 FPS)
- Runs MediaPipe Pose detection on each frame
- Filters 33 landmarks → 21 relevant body points (removes face)
- Computes movement direction from body center position
- Thread-safe snapshot for game consumption

**Threading Model:**
```
Main Thread (Game Loop)
    │
    └──▶ PoseWorker Thread (Background)
              │
              ├──▶ OpenCV capture
              ├──▶ MediaPipe inference
              └──▶ Shared state (snapshot)
```

**Filtered Landmarks (21 points):**
- Shoulders, elbows, wrists, hips, knees, ankles
- Excludes: face, hands, feet (visibility issues)

### PoseViewer (`ComputerVision/pose_viewer.py`)

Debug window showing:
- Skeleton overlay on camera feed
- Probability bars per action class
- Movement direction indicator
- Real-time prediction display

### Data Labeling (`ComputerVision/data_labeling.py`)

Manual labeling tool for collecting training data.

**Key Mappings:**
| Key | Action | Key | Action |
|-----|--------|-----|--------|
| a | Jab | q | Rear Low Kick |
| s | Cross | w | Side Kick |
| d | Lead Hook | e | Spinning Back High Kick |
| f | Rear Hook | r | Crouching Low Sweep |
| g | Uppercut | z | Hadouken |
| h | Jumping Cross | x | Shoryuken |
| c | Grab | l/i | Idle |

---

## Machine Learning Models

### LSTM Model (`Models/LSTM_v1/`)

**Architecture:**
```
Input (84 features) → LSTM (128 hidden) → FC (128→14) → Output (14 classes)
```

**Configuration:**
- Input: 84 features (21 landmarks × 4 values: x, y, z, visibility)
- Sequence length: 5 frames (temporal window)
- Hidden size: 128 units
- Output: 14 action classes

**Inference (`lstm_live_predictions.py`):**
- Maintains sliding window buffer of last 5 frames
- Applies softmax for probability distribution
- Returns top prediction + confidence scores

### FC Model (`Models/FC/`)

**Architecture:**
```
Input (84 features) → FC (84→128) → FC (128→14) → Output (14 classes)
```

**Configuration:**
- Input: 84 features (single frame)
- Hidden layers: 2 FC layers (128 units each)
- Output: 14 action classes

**Note:** Faster inference but lacks temporal context.

### Action Classes (14 total)

| ID | Action | Game Input |
|----|--------|------------|
| 0 | idle | - |
| 1 | jab | Light Punch |
| 2 | cross | Heavy Punch |
| 3 | lead_hook | Medium Punch |
| 4 | rear_hook | Medium Punch |
| 5 | uppercut | Special (DP) |
| 6 | jumping_cross | Jump + Heavy |
| 7 | rear_low_kick | Light Kick |
| 8 | side_kick | Medium Kick |
| 9 | spinning_back_high_kick | Heavy Kick |
| 10 | crouching_low_sweep | Crouch + Light Kick |
| 11 | grab | Throw |
| 12 | hadouken | QCF + Punch |
| 13 | shoryuken | DP + Punch |

---

## Game Engine

### InputDevice (`Util/Input_device.py`)

Abstracts input from multiple sources into unified game input.

**Input Modes:**
| Mode | Source | Use Case |
|------|--------|----------|
| `keyboard_mode` | Keyboard | Standard play |
| `joystick_mode` | Gamepad | Console-style |
| `external_mode_2` | Computer Vision | Body gestures |
| `random_mode` | Random | Testing |

**Key Method:** `external_mode_2()`
- Reads latest CV prediction
- Maps action → keyboard event tuple
- Handles combo detection (QCF, DP motions)

### Game Screens (`Util/Game_Screens.py`)

| Screen | Purpose |
|--------|---------|
| TitleScreen | Game title, start prompt |
| ModeSelectionScreen | Mode selection menu |
| VersusScreen | Main fighting game mode |
| TrainingScreen | Practice mode |
| ComboTrialScreen | Combo challenges |
| EditScreen | Character/stage editor |
| DebuggingScreen | Debug view with hitboxes |

### Active Objects (`Util/Active_Objects.py`)

Base class for game entities handling:
- Position, velocity, acceleration
- State machine (standing, crouching, jumping, attacking)
- Hitboxes/hurtboxes/pushboxes
- Animation frame management

### Common Functions (`Util/Common_functions.py`)

~1000 lines of game logic:
- Collision detection (hitbox/hurtbox intersection)
- State transitions and command parsing
- Damage and knockback calculation
- Hitstun and blockstun frames
- Combo system implementation

### OpenGL Renderer (`Util/OpenGL_Renderer.py`)

3D rendering via PyOpenGL for:
- 2D sprite billboard rendering
- Camera following with smooth interpolation
- Visual effects (zoom, shake, tint)
- Lighting calculations

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RUNTIME DATA FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

Frame Capture (30 FPS)
        │
        ▼
MediaPipe Pose Detection
        │
        ├──▶ PoseViewer (debug display)
        │
        ▼
21 Landmarks Extraction
        │
        ├──▶ Movement Detection
        │         │
        │         └──▶ Edge Zone Check
        │
        └──▶ Feature Vector (84 values)
                 │
                 ▼
         LSTM Model Inference
         (5-frame sequence)
                 │
                 ▼
         Action Prediction
         (softmax, 14 classes)
                 │
                 ▼
         Confidence Threshold
         (default: 80%)
                 │
                 ├──▶ < threshold: idle
                 │
                 └──▶ > threshold: action
                          │
                          ▼
                   InputDevice
                   .external_mode_2()
                          │
                          ▼
                   Keyboard Event
                   (PyGame tuple)
                          │
                          ▼
                   Game Logic
                   (state machine)
                          │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Character           Opponent           Effects
   State Update      AI/State Update     (particles, etc.)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                    OpenGL Render
                    (60 FPS display)
```

---

## Key Files Reference

### Computer Vision
| File | Purpose |
|------|---------|
| `pose_worker.py` | Threaded webcam + pose detection |
| `pose_viewer.py` | Debug skeleton overlay |
| `data_labeling.py` | Training data collection tool |
| `detect_track_pose.py` | Record poses to CSV |
| `trainModel.py` | Model training script |

### Models
| File | Purpose |
|------|---------|
| `lstm_model.py` | LSTM architecture + training |
| `lstm_live_predictions.py` | Real-time LSTM inference |
| `fc_model.py` | FC architecture + training |
| `fc_model_live.py` | Real-time FC inference |

### Game Engine
| File | Purpose |
|------|---------|
| `main.py` | Entry point, initialization |
| `Input_device.py` | Input abstraction layer |
| `Game_Screens.py` | Screen/menu management |
| `Active_Objects.py` | Entity base class |
| `Common_functions.py` | Game logic functions |
| `OpenGL_Renderer.py` | 3D rendering |
| `Interface_objects.py` | UI components |

### Data
| Directory | Purpose |
|-----------|---------|
| `Data/Phase1/` | Initial training dataset |
| `Data/Phase2/` | Extended dataset |
| `DataAugmentation/` | Data augmentation utils |
| `Assets/images/` | Character sprites |
| `Assets/objects/` | Character/stage JSON configs |

---

## Character Configuration Format

Characters defined in JSON (`Assets/objects/SF3/Ryu.json`):

```json
{
  "type": "character",
  "name": "Character Name",
  "gravity": -2.16,
  "gauges": {
    "health": {"max": 10000, "rate": 0},
    "super": {"max": 10000, "rate": 10}
  },
  "boxes": {
    "hurtbox": {"boxes": [...]},
    "hitbox": {"boxes": [...]},
    "pushbox": {"boxes": [...]}
  },
  "states": {
    "Stand Block": {...},
    "Crouch Light Hit": {...},
    "Shoryuken Fierce": {...},
    "Hadouken": {...}
  },
  "combo_trails": [...]
}
```

---

## Setup Instructions

### Prerequisites
```bash
conda create -n StreetFighter python=3.12
conda activate StreetFighter
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)
```
mediapipe==0.10.32
opencv-python==4.13.0.92
torch>=2.0
pygame==2.6.1
PyOpenGL==3.1.10
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Running
```python
# main.py
COMPUTER_VISION = True  # Enable gesture control
python main.py
```

### With CV Mode Disabled (keyboard only)
```python
# main.py
COMPUTER_VISION = False
python main.py
```

---

## Performance Considerations

| Component | Target | Strategy |
|-----------|--------|----------|
| Game Loop | 60 FPS | Separate CV thread |
| Pose Detection | 30 FPS | Background thread |
| Model Inference | <10ms | Single forward pass |
| Total Latency | ~50ms | Pipeline optimization |

**Thread Safety:** PoseWorker uses locks for snapshot access to prevent race conditions with the main game loop.

---

## Extending the Project

### Adding New Actions
1. Record new poses with `data_labeling.py`
2. Add class to training labels
3. Retrain model
4. Update action mappings in `Input_device.py`
5. Add game state in character JSON

### Adding New Characters
1. Create JSON config in `Assets/objects/[Character]/`
2. Add sprites to `Assets/images/`
3. Define states and hitboxes
4. Register in game

### Improving Model Accuracy
1. Collect more training data
2. Apply data augmentation (`DataAugmentation/`)
3. Increase sequence length (LSTM)
4. Try larger hidden dimensions
5. Ensemble LSTM + FC models
