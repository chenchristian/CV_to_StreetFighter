# Quick Start Guide

## Running the Game

### Single Player

Simply run:
```bash
python main.py
```

Then select "1 Player" from the menu. The game will use your camera for controls.

---

## Network Multiplayer (Two Players, Each on Their Own Device)

### Step 1: Find Host IP Address

On the device that will be the **host** (Player 1):

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (e.g., `192.168.1.100`)

**Mac/Linux:**
```bash
ifconfig
```
or
```bash
ip addr
```
Look for "inet" address (e.g., `192.168.1.100`)

### Step 2: Start Host (Player 1)

On the host device, open a terminal and run:
```bash
python main.py --host
```

**What happens:**
- Game starts
- Select "2 Players (Network)" from the menu
- Game waits for Player 2 to connect
- Once connected, game starts automatically

### Step 3: Start Client (Player 2)

On the second device, open a terminal and run:
```bash
python main.py --client --ip <HOST_IP_ADDRESS>
```

**Replace `<HOST_IP_ADDRESS>` with the IP you found in Step 1.**

Example:
```bash
python main.py --client --ip 192.168.1.100
```

**What happens:**
- Game starts
- Select "2 Players (Network)" from the menu
- Game connects to host
- Once connected, game starts automatically

### Step 4: Play!

- **Player 1 (Host)**: Uses their camera to control their character
- **Player 2 (Client)**: Uses their camera to control their character
- Both players see the same game state synchronized over the network

---

## Testing on One Computer

If you want to test network mode on a single computer:

### Option 1: Host + Client (One Camera + Keyboard)

**Terminal 1 (Host):**
```bash
python main.py --host
```
- Select "2 Players (Network)"
- Uses camera for Player 1

**Terminal 2 (Client):**
```bash
python main.py --client --ip 127.0.0.1
```
- Select "2 Players (Network)"
- Will try to use camera for Player 2 (may conflict)

**Note:** Both processes trying to use the camera may cause issues. For better testing, you might want to modify the client to use keyboard instead of camera.

---

## Troubleshooting

### "Cannot connect to host"
- Make sure host is started first
- Check that IP address is correct
- Verify both devices are on the same network
- Check firewall settings (port 5555)

### "Connection timeout"
- Host must be running before client connects
- Try using `127.0.0.1` for localhost testing
- Check network connectivity

### Camera not working
- Make sure camera permissions are granted
- Check that no other application is using the camera
- Try a different camera index: modify `camera_index=0` to `camera_index=1` in `main.py`

### Game not synchronized
- Make sure both players selected "2 Players (Network)"
- Check that network connection is established (you should see connection messages)
- Verify both devices are sending/receiving inputs

---

## Requirements

- Python 3.x
- All game dependencies (pygame, torch, mediapipe, opencv, etc.)
- Camera on each device (for network multiplayer)
- Both devices on the same local network

