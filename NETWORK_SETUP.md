# Network Multiplayer Setup

This guide explains how to set up two-player network multiplayer where each player uses their own device and camera.

## Requirements

- Two computers/devices on the same local network
- Each device needs a camera
- Both devices need to have the game installed

## Setup Steps

### Step 1: Find Host IP Address

On the device that will act as the **host** (Player 1):

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (usually something like 192.168.1.xxx)

**Mac/Linux:**
```bash
ifconfig
```
or
```bash
ip addr
```
Look for "inet" address (usually something like 192.168.1.xxx)

### Step 2: Start Host (Player 1)

On the host device, run:
```bash
python main.py --host
```

The host will:
- Use their camera for Player 1
- Wait for Player 2 to connect
- Receive Player 2's inputs over the network

### Step 3: Start Client (Player 2)

On the second device, run:
```bash
python main.py --client --ip <HOST_IP_ADDRESS>
```

Replace `<HOST_IP_ADDRESS>` with the IP address you found in Step 1.

Example:
```bash
python main.py --client --ip 192.168.1.100
```

The client will:
- Connect to the host
- Use their camera for Player 2
- Send their inputs to the host over the network

### Step 4: Play!

1. Both players select "2 Players (Network)" from the menu
2. Wait for connection (you'll see connection status)
3. Once connected, the game starts automatically
4. Each player controls their character with their camera

## Troubleshooting

### Cannot Connect

- **Check firewall**: Make sure port 5555 is allowed through the firewall on both devices
- **Check IP address**: Verify the IP address is correct and both devices are on the same network
- **Try localhost**: For testing on the same machine, use `127.0.0.1` as the IP

### Connection Timeout

- Make sure the host is started first
- Check that both devices are on the same Wi-Fi/network
- Try restarting both applications

### Input Lag

- Use a wired network connection if possible (lower latency than Wi-Fi)
- Make sure both devices have good network connectivity
- Close other network-intensive applications

## How It Works

- **Host (Player 1)**: Uses local camera, sends inputs to client, receives client inputs
- **Client (Player 2)**: Uses local camera, sends inputs to host, receives host inputs
- Both devices run the game locally but synchronize inputs over the network
- The game state is synchronized by sharing input data each frame

