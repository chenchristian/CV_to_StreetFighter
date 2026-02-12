# Network Multiplayer Implementation - Changes Documentation

This document explains what changes were made to implement network multiplayer and how the system works.

## Overview

The game now supports two-player network multiplayer where each player uses their own device and camera. Players connect over a local network, with one device acting as the host and the other as the client.

## Changes Made

### 1. Removed Single-Camera Two-Player Code

**File: `ComputerVision/pose_worker.py`**
- Removed `MultiPlayerPoseWorker` class that split camera frames into left/right halves
- Kept only the original `PoseWorker` class for single-player pose detection

### 2. Added Network Support to InputDevice

**File: `Util/Input_device.py`**

**Added:**
- Import for `NetworkPeer`: `from Util.network_peer import NetworkPeer`
- `network_peer` parameter to `InputDevice.__init__()`
- Two new input modes:
  - `"network"`: Receives inputs from the network peer
  - `"network_sender"`: Sends local inputs to network and uses them locally

**New Methods:**
```python
def network_mode(self):
    """Receive inputs from network peer"""
    # Gets latest input from network_peer and processes it

def network_sender_mode(self):
    """Send local inputs to network and use them locally"""
    # Gets input from pose_worker, sends to network, and uses locally
```

### 3. Updated GameObject for Network Support

**File: `main.py`**

**Added to `GameObject.__init__()`:**
- `self.network_mode = False` - Flag indicating if network mode is active
- `self.is_host = False` - Whether this instance is the host
- `self.network_peer = None` - NetworkPeer instance for communication
- `self.remote_ip = "127.0.0.1"` - IP address of host (for client)
- `self.port = 5555` - Network port

**Updated `Input_device_available()` method:**
- Now handles network mode configuration
- **Host mode**: Player 1 uses local camera (`external` mode), Player 2 receives from network (`network` mode)
- **Client mode**: Player 1 receives from network (`network` mode), Player 2 uses local camera (`network_sender` mode)

### 4. Updated PlayerSelectionScreen

**File: `Util/Game_Screens.py`**

**Changes:**
- Changed "2 Players" option to "2 Players (Network)"
- Added network initialization logic in `__dein__()` method
- Reads command-line arguments (`--host`, `--client`, `--ip`) to determine network role
- Creates and starts `NetworkPeer` instance when network mode is selected
- Starts pose worker for the local player

### 5. Updated Main Initialization

**File: `main.py`**

**Changes:**
- Removed `MultiPlayerPoseWorker` import and initialization
- Simplified to only create a single `PoseWorker` instance
- Removed `multi_pose_worker` parameter from `GameObject` initialization

## How It Works

### Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   Host Device   │                    │  Client Device  │
│   (Player 1)    │                    │   (Player 2)    │
├─────────────────┤                    ├─────────────────┤
│                 │                    │                 │
│  Camera →       │                    │  Camera →       │
│  PoseWorker →   │                    │  PoseWorker →   │
│  InputDevice    │                    │  InputDevice    │
│  (external)     │                    │  (network_      │
│                 │                    │   sender)       │
│                 │                    │                 │
│  NetworkPeer    │◄────TCP Socket────►│  NetworkPeer    │
│  (host)         │                    │  (client)       │
│                 │                    │                 │
│  InputDevice    │                    │  InputDevice    │
│  (network)      │                    │  (network)      │
│  ← Player 2     │                    │  ← Player 1     │
│                 │                    │                 │
└─────────────────┘                    └─────────────────┘
```

### Data Flow

#### Host Device (Player 1):
1. **Local Input (Player 1)**:
   - Camera captures pose → `PoseWorker` processes → `InputDevice` (external mode) → Game
   - Input is also sent to client via `NetworkPeer`

2. **Remote Input (Player 2)**:
   - Receives input from client via `NetworkPeer` → `InputDevice` (network mode) → Game

#### Client Device (Player 2):
1. **Local Input (Player 2)**:
   - Camera captures pose → `PoseWorker` processes → `InputDevice` (network_sender mode)
   - Input is sent to host via `NetworkPeer` AND used locally

2. **Remote Input (Player 1)**:
   - Receives input from host via `NetworkPeer` → `InputDevice` (network mode) → Game

### Network Communication

**NetworkPeer (`Util/network_peer.py`)**:
- Uses TCP sockets for reliable communication
- Sends/receives input data as pickled messages
- Format: `{"type": "input", "frame": frame_number, "input": input_data}`
- Handles connection, disconnection, and message queuing
- Thread-safe with separate send/receive threads

### Input Synchronization

- Each frame, both devices:
  1. Process local camera input
  2. Send local input to the other device
  3. Receive and apply remote input
  4. Update game state with both inputs

- Frame numbers are synchronized to ensure inputs are applied at the correct game frame
- Input queue buffers handle network latency

## Command-Line Arguments

The game accepts these arguments for network setup:

- `--host`: Run as host (Player 1)
- `--client`: Run as client (Player 2)
- `--ip <address>`: Host IP address (required for client)

**Examples:**
```bash
# Host
python main.py --host

# Client
python main.py --client --ip 192.168.1.100
```

## Key Components

### 1. NetworkPeer (`Util/network_peer.py`)
- Handles TCP socket communication
- Manages connection state
- Queues input messages
- Thread-safe send/receive operations

### 2. InputDevice Network Modes
- **`network_mode`**: Receives inputs from `network_peer.get_latest_input()`
- **`network_sender_mode`**: Gets local input, sends via `network_peer.send_input()`, and uses locally

### 3. GameObject Network Configuration
- Detects network mode from player selection
- Configures input devices based on host/client role
- Manages network peer lifecycle

## Game Flow

1. **Startup**: Game loads, creates `PoseWorker` (not started yet)
2. **Player Selection**: User selects "1 Player" or "2 Players (Network)"
3. **Network Setup** (if 2 players):
   - Reads command-line args to determine host/client
   - Creates and starts `NetworkPeer`
   - Waits for connection (host) or connects (client)
4. **Input Device Configuration**:
   - Host: P1 = external, P2 = network
   - Client: P1 = network, P2 = network_sender
5. **Game Start**: Both players' inputs are processed and synchronized

## Technical Details

### Input Format
Game inputs are arrays: `[[x, y], attack1, attack2, ...]`
- `[x, y]`: Movement direction (-1 to 1 for left/right, up/down)
- `attack1-8`: Button states (0 or 1)

### Network Protocol
- **Message Format**: Pickled Python dictionaries
- **Size Prefix**: 4-byte big-endian integer before each message
- **Message Types**:
  - `"input"`: Game input data
  - `"ping"`/`"pong"`: Connection keepalive

### Threading
- `NetworkPeer` uses separate threads for:
  - Receiving data (blocking socket read)
  - Sending data (from queue)
- `PoseWorker` runs in its own thread
- Main game loop processes inputs synchronously

## Limitations & Considerations

1. **Network Latency**: Inputs may arrive with delay, causing slight lag
2. **Frame Synchronization**: Both devices must run at similar frame rates
3. **Connection Stability**: Disconnections will break gameplay
4. **Local Network Only**: Designed for LAN play, not internet
5. **No Rollback**: No prediction/rollback netcode for handling lag

## Future Improvements

- Connection status display in-game
- Automatic reconnection on disconnect
- Input prediction to reduce perceived lag
- Frame rate synchronization
- Better error handling for network issues

