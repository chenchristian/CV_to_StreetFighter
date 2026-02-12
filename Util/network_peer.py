"""
Peer-to-peer network module for multiplayer gameplay.
Handles input synchronization between two players on different computers.
"""
import socket
import threading
import queue
import pickle
import time
from typing import Optional, Callable

class NetworkPeer:
    """
    Peer-to-peer network handler for synchronizing game inputs.
    One peer acts as host (listens), the other as client (connects).
    """
    
    def __init__(self, 
                 host: bool = False,
                 remote_ip: str = "127.0.0.1",
                 port: int = 5555,
                 timeout: float = 5.0):
        """
        Initialize network peer.
        
        Args:
            host: If True, this peer listens for connections. If False, connects to host.
            remote_ip: IP address of the host (only used if host=False)
            port: Port number for communication
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.remote_ip = remote_ip
        self.port = port
        self.timeout = timeout
        
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Input queues
        self.input_queue = queue.Queue(maxsize=60)  # Buffer up to 1 second at 60fps
        self.send_queue = queue.Queue()
        
        # Threads
        self.receive_thread: Optional[threading.Thread] = None
        self.send_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Frame synchronization
        self.frame_number = 0
        self.last_received_frame = -1
        self.last_sent_input = None
        
        # Connection callback
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        
    def start(self):
        """Start the network peer (connect or listen)."""
        if self.host:
            self._start_host()
        else:
            self._start_client()
    
    def _start_host(self):
        """Start as host (listening for connections)."""
        try:
            # Create listening socket
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Set a longer timeout for accepting connections (30 seconds)
            listen_socket.settimeout(30.0)
            
            # Bind and listen
            listen_socket.bind(("0.0.0.0", self.port))
            listen_socket.listen(1)
            
            # Get local IP address for display
            try:
                # Connect to a dummy address to get local IP
                temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                temp_sock.connect(("8.8.8.8", 80))
                local_ip = temp_sock.getsockname()[0]
                temp_sock.close()
            except:
                local_ip = "localhost"
            
            print(f"[Network] Host listening on port {self.port}...")
            print(f"[Network] Local IP address: {local_ip}")
            print(f"[Network] Client should connect with: --client --ip {local_ip}")
            print(f"[Network] Waiting for client to connect...")
            print(f"[Network] NOTE: If client can't connect, check firewall settings:")
            print(f"[Network]   - macOS: System Settings > Network > Firewall > Options")
            print(f"[Network]   - Windows: Windows Defender Firewall > Allow an app")
            print(f"[Network]   - Linux: Check iptables/ufw rules for port {self.port}")
            
            # Accept connection in a thread (retry until connected)
            def accept_connection():
                while not self._stop_event.is_set() and not self.connected:
                    try:
                        conn, addr = listen_socket.accept()
                        # Set socket timeout for receive operations (0.1s timeout for non-blocking behavior)
                        conn.settimeout(0.1)
                        with self.connection_lock:
                            self.socket = conn
                            self.connected = True
                        print(f"[Network] Connected to {addr}")
                        if self.on_connected:
                            self.on_connected()
                        self._start_threads()
                        print("[Network] Send/receive threads started")
                        # Close listening socket
                        try:
                            listen_socket.close()
                        except:
                            pass
                        break
                    except socket.timeout:
                        # Timeout is normal, just retry
                        if not self._stop_event.is_set():
                            print(f"[Network] Still waiting for connection... (timeout: 30s)")
                        continue
                    except Exception as e:
                        print(f"[Network] Connection error: {e}")
                        if not self._stop_event.is_set():
                            time.sleep(1)  # Wait before retrying
                        continue
                # Clean up listening socket if we exit
                try:
                    listen_socket.close()
                except:
                    pass
            
            threading.Thread(target=accept_connection, daemon=True).start()
            
        except Exception as e:
            print(f"[Network] Host setup error: {e}")
    
    def _start_client(self):
        """Start as client (connect to host)."""
        # Retry connection in a thread
        def connect_to_host():
            max_retries = 60  # Try for up to 60 seconds
            retry_count = 0
            
            while not self._stop_event.is_set() and not self.connected and retry_count < max_retries:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2.0)  # 2 second timeout per attempt
                    
                    if retry_count == 0:
                        print(f"[Network] Connecting to {self.remote_ip}:{self.port}...")
                    else:
                        print(f"[Network] Retrying connection to {self.remote_ip}:{self.port}... (attempt {retry_count + 1}/{max_retries})")
                    
                    sock.connect((self.remote_ip, self.port))
                    # Set socket timeout for receive operations (0.1s timeout for non-blocking behavior)
                    sock.settimeout(0.1)
                    
                    with self.connection_lock:
                        self.socket = sock
                        self.connected = True
                    print(f"[Network] ✓ Connected to host")
                    
                    if self.on_connected:
                        self.on_connected()
                    
                    self._start_threads()
                    print("[Network] Send/receive threads started")
                    break
                    
                except socket.timeout:
                    retry_count += 1
                    try:
                        sock.close()
                    except:
                        pass
                    if retry_count < max_retries:
                        if retry_count % 5 == 0:  # Print every 5 attempts
                            print(f"[Network] Connection timeout - check:")
                            print(f"[Network]   - Host is running and selected '2 Players (Network)'")
                            print(f"[Network]   - IP address is correct: {self.remote_ip}")
                            print(f"[Network]   - Both computers are on the same network")
                            print(f"[Network]   - Firewall allows port {self.port} (try: python -m http.server {self.port} on host to test)")
                        time.sleep(1)  # Wait 1 second before retry
                    continue
                except ConnectionRefusedError:
                    retry_count += 1
                    try:
                        sock.close()
                    except:
                        pass
                    if retry_count < max_retries:
                        if retry_count % 5 == 0:  # Print every 5 attempts
                            print(f"[Network] Connection refused - check:")
                            print(f"[Network]   - Host is running and selected '2 Players (Network)'")
                            print(f"[Network]   - IP address is correct: {self.remote_ip}")
                            print(f"[Network]   - Firewall on host allows incoming connections on port {self.port}")
                        time.sleep(1)
                    continue
                except OSError as e:
                    retry_count += 1
                    try:
                        sock.close()
                    except:
                        pass
                    if retry_count < max_retries:
                        if retry_count % 5 == 0:  # Print every 5 attempts
                            error_msg = str(e)
                            error_code = getattr(e, 'errno', None)
                            
                            if "No route to host" in error_msg or error_code == 113 or "EHOSTUNREACH" in str(e):
                                print(f"[Network] ✗ No route to host - Network unreachable")
                                print(f"[Network] Troubleshooting:")
                                print(f"[Network]   1. Verify IP address is correct: {self.remote_ip}")
                                print(f"[Network]   2. Check both computers are on the same network:")
                                print(f"[Network]      - Same Wi-Fi network, OR")
                                print(f"[Network]      - Same Ethernet subnet")
                                print(f"[Network]   3. Test connectivity:")
                                print(f"[Network]      - Run: ping {self.remote_ip}")
                                print(f"[Network]      - If ping fails, IP is wrong or not on same network")
                                print(f"[Network]   4. Get host IP address:")
                                print(f"[Network]      - On host, run: ifconfig (macOS/Linux) or ipconfig (Windows)")
                                print(f"[Network]      - Look for IP under your active network interface")
                                print(f"[Network]   5. If using VPN, try disconnecting VPN")
                            elif "Network is unreachable" in error_msg or error_code == 101 or "ENETUNREACH" in str(e):
                                print(f"[Network] ✗ Network unreachable")
                                print(f"[Network]   - Check IP address: {self.remote_ip}")
                                print(f"[Network]   - Verify both computers on same network")
                                print(f"[Network]   - Test: ping {self.remote_ip}")
                            elif "Operation timed out" in error_msg:
                                print(f"[Network] Connection timed out - check:")
                                print(f"[Network]   - Firewall is blocking port {self.port}")
                                print(f"[Network]   - Host firewall allows incoming connections")
                            else:
                                print(f"[Network] Network error ({error_code}): {e}")
                        time.sleep(1)
                    continue
                except Exception as e:
                    retry_count += 1
                    try:
                        sock.close()
                    except:
                        pass
                    if retry_count < max_retries:
                        if retry_count % 5 == 0:  # Print every 5 attempts
                            print(f"[Network] Connection error: {e}")
                        time.sleep(1)
                    continue
            
            if not self.connected and retry_count >= max_retries:
                print(f"[Network] ✗ Failed to connect after {max_retries} attempts")
                print(f"[Network] =========================================")
                print(f"[Network] TROUBLESHOOTING GUIDE")
                print(f"[Network] =========================================")
                print(f"[Network] 1. VERIFY IP ADDRESS")
                print(f"[Network]    Current IP: {self.remote_ip}")
                print(f"[Network]    - On HOST computer, run:")
                print(f"[Network]      macOS/Linux: ifconfig | grep 'inet '")
                print(f"[Network]      Windows: ipconfig")
                print(f"[Network]    - Look for IP under your active Wi-Fi/Ethernet adapter")
                print(f"[Network]    - Should be something like: 192.168.x.x or 10.0.x.x")
                print(f"[Network]")
                print(f"[Network] 2. TEST NETWORK CONNECTIVITY")
                print(f"[Network]    On CLIENT computer, run:")
                print(f"[Network]      ping {self.remote_ip}")
                print(f"[Network]    - If ping fails → Wrong IP or not on same network")
                print(f"[Network]    - If ping works → Firewall issue (see step 4)")
                print(f"[Network]")
                print(f"[Network] 3. VERIFY SAME NETWORK")
                print(f"[Network]    - Both computers must be on the SAME Wi-Fi network")
                print(f"[Network]    - Or both on the SAME Ethernet subnet")
                print(f"[Network]    - Check Wi-Fi network name matches on both computers")
                print(f"[Network]")
                print(f"[Network] 4. CHECK FIREWALL (if ping works but connection fails)")
                print(f"[Network]    HOST computer:")
                print(f"[Network]      macOS: System Settings > Network > Firewall > Options")
                print(f"[Network]        - Allow Python incoming connections")
                print(f"[Network]      Windows: Windows Defender Firewall")
                print(f"[Network]        - Allow Python through firewall")
                print(f"[Network]")
                print(f"[Network] 5. VERIFY HOST IS RUNNING")
                print(f"[Network]    - Host must run: python main.py --host")
                print(f"[Network]    - Host must select '2 Players (Network)' from menu")
                print(f"[Network]    - Wait for 'Host listening on port 5555...' message")
                print(f"[Network]")
                print(f"[Network] 6. TEST WITH LOCALHOST FIRST")
                print(f"[Network]    If localhost works but network doesn't:")
                print(f"[Network]    - This confirms it's a network/firewall issue")
                print(f"[Network]    - Try: python main.py --client --ip 127.0.0.1")
                print(f"[Network] =========================================")
        
        threading.Thread(target=connect_to_host, daemon=True).start()
    
    def _start_threads(self):
        """Start receive and send threads."""
        if self.receive_thread is None or not self.receive_thread.is_alive():
            self._stop_event.clear()
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
        
        if self.send_thread is None or not self.send_thread.is_alive():
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()
    
    def _receive_loop(self):
        """Receive inputs from the other peer."""
        buffer = b""
        expected_size = None
        
        while not self._stop_event.is_set():
            try:
                with self.connection_lock:
                    if not self.connected or self.socket is None:
                        time.sleep(0.1)  # Wait a bit before checking again
                        continue
                    sock = self.socket
                
                # Receive data
                try:
                    data = sock.recv(4096)
                except (socket.timeout, OSError):
                    continue  # Timeout is normal, just continue
                
                if not data:
                    print("[Network] Connection closed by peer")
                    self.disconnect()
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
                            
                            # Handle received input
                            if message.get("type") == "input":
                                frame = message.get("frame", 0)
                                input_data = message.get("input")
                                
                                # Debug: print first few receives
                                if frame < 5:
                                    print(f"[Network] Received input frame {frame}: {input_data[:2] if isinstance(input_data, list) and len(input_data) > 0 else input_data}")
                                
                                # Add to queue (non-blocking)
                                try:
                                    if self.input_queue.full():
                                        self.input_queue.get_nowait()  # Remove oldest
                                    self.input_queue.put_nowait((frame, input_data))
                                    self.last_received_frame = max(self.last_received_frame, frame)
                                except queue.Full:
                                    pass  # Queue full, drop this input
                                except Exception as e:
                                    print(f"[Network] Error queuing received input: {e}")
                            
                            elif message.get("type") == "ping":
                                # Respond to ping
                                self.send_message({"type": "pong", "frame": message.get("frame")})
                                
                        except Exception as e:
                            print(f"[Network] Error parsing message: {e}")
                            buffer = buffer[expected_size:]
                            expected_size = None
                    else:
                        break  # Need more data
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Network] Receive error: {e}")
                self.disconnect()
                break
    
    def _send_loop(self):
        """Send inputs to the other peer."""
        while not self._stop_event.is_set():
            try:
                # Get input from queue (blocking with timeout)
                try:
                    message = self.send_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                with self.connection_lock:
                    if not self.connected or self.socket is None:
                        continue  # Wait for connection instead of breaking
                    sock = self.socket
                
                # Serialize and send
                try:
                    data = pickle.dumps(message)
                    size = len(data).to_bytes(4, byteorder='big')
                    sock.sendall(size + data)
                    # Debug: print first few sends
                    if message.get("type") == "input" and self.frame_number < 5:
                        print(f"[Network] Sent input frame {message.get('frame')}: {message.get('input')[:2]}")
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    print(f"[Network] Send error (connection lost): {e}")
                    self.disconnect()
                    break
                
            except Exception as e:
                print(f"[Network] Send error: {e}")
                import traceback
                traceback.print_exc()
                self.disconnect()
                break
    
    def send_input(self, input_data: list, frame: Optional[int] = None):
        """
        Send input data to the other peer.
        
        Args:
            input_data: Raw input array (same format as pose_worker.get_latest_game_input())
            frame: Frame number (auto-incremented if None)
        """
        # Check connection status
        if not self.is_connected():
            return
        
        if frame is None:
            self.frame_number += 1
            frame = self.frame_number
        
        message = {
            "type": "input",
            "frame": frame,
            "input": input_data,
            "timestamp": time.time()
        }
        
        try:
            self.send_queue.put_nowait(message)
        except queue.Full:
            pass  # Queue full, drop this input
        except Exception as e:
            print(f"[Network] Error queuing input: {e}")
    
    def get_latest_input(self) -> Optional[list]:
        """
        Get the most recent input from the other peer.
        Returns None if no input available.
        """
        latest = None
        latest_frame = -1
        
        # Get all available inputs and keep the latest
        while not self.input_queue.empty():
            try:
                frame, input_data = self.input_queue.get_nowait()
                if frame > latest_frame:
                    latest_frame = frame
                    latest = input_data
            except queue.Empty:
                break
        
        return latest
    
    def send_message(self, message: dict):
        """Send a custom message to the other peer."""
        if not self.connected:
            return
        
        try:
            self.send_queue.put_nowait(message)
        except queue.Full:
            pass
    
    def is_connected(self) -> bool:
        """Check if connected to the other peer."""
        with self.connection_lock:
            return self.connected and self.socket is not None
    
    def disconnect(self):
        """Disconnect from the other peer."""
        with self.connection_lock:
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
        
        self._stop_event.set()
        
        if self.on_disconnected:
            self.on_disconnected()
        
        print("[Network] Disconnected")
    
    def stop(self):
        """Stop the network peer."""
        self.disconnect()
        
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        if self.send_thread:
            self.send_thread.join(timeout=1.0)


