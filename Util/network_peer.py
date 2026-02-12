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
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(self.timeout)
            
            # Bind and listen
            self.socket.bind(("0.0.0.0", self.port))
            self.socket.listen(1)
            print(f"[Network] Host listening on port {self.port}...")
            
            # Accept connection in a thread
            def accept_connection():
                try:
                    conn, addr = self.socket.accept()
                    with self.connection_lock:
                        self.socket = conn
                        self.connected = True
                    print(f"[Network] Connected to {addr}")
                    if self.on_connected:
                        self.on_connected()
                    self._start_threads()
                except socket.timeout:
                    print("[Network] Connection timeout")
                except Exception as e:
                    print(f"[Network] Connection error: {e}")
            
            threading.Thread(target=accept_connection, daemon=True).start()
            
        except Exception as e:
            print(f"[Network] Host setup error: {e}")
    
    def _start_client(self):
        """Start as client (connect to host)."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            print(f"[Network] Connecting to {self.remote_ip}:{self.port}...")
            self.socket.connect((self.remote_ip, self.port))
            
            with self.connection_lock:
                self.connected = True
            print(f"[Network] Connected to host")
            
            if self.on_connected:
                self.on_connected()
            
            self._start_threads()
            
        except socket.timeout:
            print(f"[Network] Connection timeout to {self.remote_ip}:{self.port}")
        except Exception as e:
            print(f"[Network] Client connection error: {e}")
    
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
                        break
                    sock = self.socket
                
                # Receive data
                data = sock.recv(4096)
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
                                
                                # Add to queue (non-blocking)
                                try:
                                    if self.input_queue.full():
                                        self.input_queue.get_nowait()  # Remove oldest
                                    self.input_queue.put_nowait((frame, input_data))
                                    self.last_received_frame = max(self.last_received_frame, frame)
                                except queue.Full:
                                    pass  # Queue full, drop this input
                            
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
                        break
                    sock = self.socket
                
                # Serialize and send
                data = pickle.dumps(message)
                size = len(data).to_bytes(4, byteorder='big')
                sock.sendall(size + data)
                
            except Exception as e:
                print(f"[Network] Send error: {e}")
                self.disconnect()
                break
    
    def send_input(self, input_data: list, frame: Optional[int] = None):
        """
        Send input data to the other peer.
        
        Args:
            input_data: Raw input array (same format as pose_worker.get_latest_game_input())
            frame: Frame number (auto-incremented if None)
        """
        if not self.connected:
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


