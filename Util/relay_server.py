"""
Relay server for network multiplayer.
Allows two players to connect through a central server, bypassing NAT/firewall issues.

Run this on a server with a public IP address:
    python relay_server.py --port 5555

Then both clients connect to this server instead of directly to each other.
"""
import socket
import threading
import pickle
import time
import argparse
from typing import Optional, Dict

class RelayServer:
    """Relay server that forwards messages between two connected clients."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        
        # Connected clients: {player_id: (socket, address, thread)}
        self.clients: Dict[int, tuple] = {}
        self.clients_lock = threading.Lock()
        
        # Server state
        self.running = False
        
    def start(self):
        """Start the relay server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(2)  # Accept up to 2 connections
            self.running = True
            
            # Get and display the actual server IP addresses
            print(f"[Relay Server] Listening on {self.host}:{self.port}")
            print(f"[Relay Server] =========================================")
            print(f"[Relay Server] SERVER IP ADDRESSES:")
            
            # Get local IP address (preferred method - most reliable)
            server_ip = None
            try:
                # Connect to a dummy address to get local IP
                temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                temp_sock.connect(("8.8.8.8", 80))
                local_ip = temp_sock.getsockname()[0]
                temp_sock.close()
                server_ip = local_ip
                print(f"[Relay Server]   Local IP: {local_ip}")
            except:
                print(f"[Relay Server]   Local IP: Could not determine")
            
            # Get hostname
            try:
                hostname = socket.gethostname()
                print(f"[Relay Server]   Hostname: {hostname}")
            except:
                pass
            
            # Get all network interfaces (fallback if local_ip method failed)
            if not server_ip:
                try:
                    hostname = socket.gethostname()
                    local_ips = socket.gethostbyname_ex(hostname)[2]
                    local_ips = [ip for ip in local_ips if not ip.startswith("127.")]
                    if local_ips:
                        server_ip = local_ips[0]  # Use first non-localhost IP
                        print(f"[Relay Server]   Network IPs: {', '.join(local_ips)}")
                except:
                    pass
            
            # Try to get public IP (for cross-network connections)
            public_ip = None
            try:
                import urllib.request
                public_ip = urllib.request.urlopen('https://api.ipify.org', timeout=3).read().decode('utf-8')
                print(f"[Relay Server]   Public IP: {public_ip} (use this if clients are on different network)")
            except:
                pass
            
            print(f"[Relay Server] =========================================")
            print(f"[Relay Server] Players should connect with:")
            if server_ip:
                print(f"[Relay Server]   For SAME network: python main.py --relay --relay-ip {server_ip} --relay-port {self.port}")
            if public_ip and public_ip != server_ip:
                print(f"[Relay Server]   For DIFFERENT network: python main.py --relay --relay-ip {public_ip} --relay-port {self.port}")
            if not server_ip:
                print(f"[Relay Server]   python main.py --relay --relay-ip <SERVER_IP> --relay-port {self.port}")
            print(f"[Relay Server] =========================================")
            print(f"[Relay Server] NOTE: If ping fails but server is running, firewall may block ICMP.")
            print(f"[Relay Server] TCP connections (port {self.port}) may still work even if ping fails.")
            print(f"[Relay Server] =========================================")
            print(f"[Relay Server] Waiting for 2 players to connect...")
            
            # Accept connections
            player_id = 1
            while self.running and player_id <= 2:
                try:
                    conn, addr = self.socket.accept()
                    conn.settimeout(0.1)  # Non-blocking with timeout
                    
                    with self.clients_lock:
                        if player_id not in self.clients:
                            # Start handler thread for this client
                            thread = threading.Thread(
                                target=self._handle_client,
                                args=(conn, addr, player_id),
                                daemon=True
                            )
                            thread.start()
                            self.clients[player_id] = (conn, addr, thread)
                            print(f"[Relay Server] Player {player_id} connected from {addr}")
                            player_id += 1
                        else:
                            conn.close()
                            print(f"[Relay Server] Player {player_id} slot already taken")
                            
                except Exception as e:
                    if self.running:
                        print(f"[Relay Server] Error accepting connection: {e}")
                    break
                    
        except Exception as e:
            print(f"[Relay Server] Server error: {e}")
        finally:
            self.stop()
    
    def _handle_client(self, conn: socket.socket, addr: tuple, player_id: int):
        """Handle messages from a client."""
        buffer = b""
        expected_size = None
        
        try:
            while self.running:
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue  # Normal timeout, keep looping
                except Exception as e:
                    print(f"[Relay Server] Error receiving from player {player_id}: {e}")
                    break
                
                if not data:
                    print(f"[Relay Server] Player {player_id} disconnected")
                    break
                
                buffer += data
                
                # Parse messages (format: <size><pickled_data>)
                while len(buffer) >= 4:
                    if expected_size is None:
                        expected_size = int.from_bytes(buffer[:4], byteorder='big')
                        buffer = buffer[4:]
                    
                    if len(buffer) >= expected_size:
                        try:
                            message = pickle.loads(buffer[:expected_size])
                            buffer = buffer[expected_size:]
                            expected_size = None
                            
                            # Forward message to the other player
                            self._forward_message(player_id, message)
                            
                        except Exception as e:
                            print(f"[Relay Server] Error parsing message from player {player_id}: {e}")
                            buffer = buffer[expected_size:]
                            expected_size = None
                    else:
                        break  # Need more data
                        
        except Exception as e:
            print(f"[Relay Server] Handler error for player {player_id}: {e}")
        finally:
            # Clean up client
            with self.clients_lock:
                if player_id in self.clients:
                    del self.clients[player_id]
                try:
                    conn.close()
                except:
                    pass
            print(f"[Relay Server] Player {player_id} handler closed")
    
    def _forward_message(self, from_player: int, message: dict):
        """Forward a message from one player to the other."""
        # Determine target player (1 -> 2, 2 -> 1)
        target_player = 2 if from_player == 1 else 1
        
        with self.clients_lock:
            if target_player in self.clients:
                target_socket, _, _ = self.clients[target_player]
                try:
                    # Serialize and send message
                    data = pickle.dumps(message)
                    size = len(data).to_bytes(4, byteorder='big')
                    target_socket.sendall(size + data)
                except Exception as e:
                    print(f"[Relay Server] Error forwarding to player {target_player}: {e}")
            else:
                # Target player not connected yet, drop message
                pass
    
    def stop(self):
        """Stop the relay server."""
        self.running = False
        
        with self.clients_lock:
            for player_id, (conn, _, _) in list(self.clients.items()):
                try:
                    conn.close()
                except:
                    pass
            self.clients.clear()
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        print("[Relay Server] Server stopped")


def main():
    import os
    parser = argparse.ArgumentParser(description="Relay server for network multiplayer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    # Check for PORT environment variable (used by Railway, Render, Heroku, etc.)
    default_port = int(os.environ.get('PORT', 5555))
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port}, or $PORT env var)")
    
    args = parser.parse_args()
    
    server = RelayServer(host=args.host, port=args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Relay Server] Shutting down...")
        server.stop()


if __name__ == "__main__":
    main()

