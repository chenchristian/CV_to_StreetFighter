"""
Desync detection and recovery system.
Detects when game states diverge and attempts to recover.
"""
import hashlib
import pickle
from typing import Optional, Dict, Tuple


class DesyncDetector:
    """Detects desync by comparing game state checksums."""
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize desync detector.
        
        Args:
            check_interval: Check for desync every N frames (default: 60 = 1 second at 60fps)
        """
        self.check_interval = check_interval
        self.last_check_frame = -1
        self.local_checksums = {}  # {frame: checksum} - our checksums
        self.remote_checksums = {}  # {frame: checksum} - opponent's checksums
        self.desync_detected = False
        self.desync_frame = -1
        self.last_synced_frame = 0
    
    def calculate_state_checksum(self, game) -> str:
        """
        Calculate a checksum of the current game state.
        This should be deterministic - same state → same checksum.
        
        Args:
            game: GameObject instance
            
        Returns:
            Hexadecimal checksum string
        """
        # Collect state data that matters for synchronization
        state_data = {
            'frame': game.emu_frame,
            'hitstop': game.hitstop,
        }
        
        # Add object states (characters, projectiles, etc.)
        objects_data = []
        for obj in game.object_list:
            if hasattr(obj, '__class__') and obj.__class__.__name__ == "BaseActiveObject":
                obj_data = {
                    'type': obj.type if hasattr(obj, 'type') else None,
                    'team': obj.team if hasattr(obj, 'team') else None,
                    'pos': tuple(obj.pos) if hasattr(obj, 'pos') else (0, 0, 0),
                    'face': obj.face if hasattr(obj, 'face') else 1,
                    'current_state': obj.current_state if hasattr(obj, 'current_state') else 'Stand',
                    'frame': tuple(obj.frame) if hasattr(obj, 'frame') else (0, 0),
                    'speed': tuple(obj.speed) if hasattr(obj, 'speed') else (0, 0),
                    'hitstop': obj.hitstop if hasattr(obj, 'hitstop') else 0,
                    'hitstun': obj.hitstun if hasattr(obj, 'hitstun') else 0,
                    'gauges': dict(obj.gauges) if hasattr(obj, 'gauges') else {},
                    'fet': obj.fet if hasattr(obj, 'fet') else 'grounded',
                }
                objects_data.append(obj_data)
        
        # Sort objects by type and team for deterministic ordering
        objects_data.sort(key=lambda x: (x['type'] or '', x['team'] or 0))
        state_data['objects'] = objects_data
        
        # Serialize and hash
        try:
            serialized = pickle.dumps(state_data, protocol=pickle.HIGHEST_PROTOCOL)
            checksum = hashlib.md5(serialized).hexdigest()
            return checksum
        except Exception as e:
            print(f"[DesyncDetector] Error calculating checksum: {e}")
            return ""
    
    def should_check(self, current_frame: int) -> bool:
        """Check if we should perform a desync check at this frame."""
        return (current_frame - self.last_check_frame) >= self.check_interval
    
    def check_desync(self, game, network_peer) -> Tuple[bool, Optional[int]]:
        """
        Check for desync by comparing checksums.
        
        Args:
            game: GameObject instance
            network_peer: NetworkPeer instance
            
        Returns:
            (is_desynced, desync_frame) - True if desync detected, frame where it occurred
        """
        current_frame = game.emu_frame
        
        # Calculate our checksum
        local_checksum = self.calculate_state_checksum(game)
        self.local_checksums[current_frame] = local_checksum
        self.last_check_frame = current_frame
        
        # Send our checksum to opponent
        if network_peer and network_peer.is_connected():
            network_peer.send_message({
                "type": "state_checksum",
                "frame": current_frame,
                "checksum": local_checksum
            })
        
        # Check if we have opponent's checksum for this frame
        # Check received_checksums from network_peer first (most up-to-date)
        remote_checksum = None
        if hasattr(network_peer, 'received_checksums') and current_frame in network_peer.received_checksums:
            remote_checksum = network_peer.received_checksums[current_frame]
            # Store in our remote_checksums for tracking
            self.remote_checksums[current_frame] = remote_checksum
        elif current_frame in self.remote_checksums:
            remote_checksum = self.remote_checksums[current_frame]
        
        if remote_checksum:
            if local_checksum != remote_checksum:
                # Desync detected!
                self.desync_detected = True
                self.desync_frame = current_frame
                print(f"[DesyncDetector] ⚠️  DESYNC DETECTED at frame {current_frame}")
                print(f"[DesyncDetector] Local checksum:  {local_checksum[:16]}...")
                print(f"[DesyncDetector] Remote checksum: {remote_checksum[:16]}...")
                return True, current_frame
            else:
                # Checksums match - update last synced frame
                self.last_synced_frame = current_frame
        
        # Clean up old checksums
        if len(self.local_checksums) > 120:  # Keep last 2 seconds
            oldest = min(self.local_checksums.keys())
            del self.local_checksums[oldest]
        if len(self.remote_checksums) > 120:
            oldest = min(self.remote_checksums.keys())
            del self.remote_checksums[oldest]
        
        return False, None
    
    def receive_checksum(self, frame: int, checksum: str):
        """Receive checksum from opponent."""
        self.remote_checksums[frame] = checksum
    
    def get_recovery_frame(self) -> int:
        """
        Get the frame to rollback to for recovery.
        Returns the last known synchronized frame.
        """
        # Find the last frame where checksums matched
        for frame in sorted(self.local_checksums.keys(), reverse=True):
            if frame in self.remote_checksums:
                if self.local_checksums[frame] == self.remote_checksums[frame]:
                    return frame
        
        # Fallback: return last synced frame
        return max(self.last_synced_frame, 0)
    
    def reset(self):
        """Reset desync detector state."""
        self.desync_detected = False
        self.desync_frame = -1
        self.local_checksums.clear()
        self.remote_checksums.clear()

