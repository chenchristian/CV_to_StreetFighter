"""
Game state management for rollback recovery.
Stores state snapshots and input history for desync recovery.
"""
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from collections import deque


class GameStateSnapshot:
    """A snapshot of game state at a specific frame."""
    
    def __init__(self, game, frame: int):
        """Create a state snapshot from the current game."""
        self.frame = frame
        self.state_data = self._serialize_game_state(game)
        self.checksum = self._calculate_checksum()
    
    def _serialize_game_state(self, game) -> Dict[str, Any]:
        """Serialize game state to a dictionary."""
        state = {
            'frame': game.emu_frame,
            'hitstop': game.hitstop,
            'camera_focus_point': list(game.camera_focus_point),
            'superstop': game.superstop,
            'camera_path': dict(game.camera_path) if hasattr(game, 'camera_path') else {},
            'frame_data': list(game.frame) if hasattr(game, 'frame') else [0, 0],
            'pos': list(game.pos) if hasattr(game, 'pos') else [10, 0, 0],
            'draw_shake': list(game.draw_shake) if hasattr(game, 'draw_shake') else [0, 0, 0, 0, 0, 0],
        }
        
        # Serialize all game objects
        objects = []
        for obj in game.object_list:
            if hasattr(obj, '__class__') and obj.__class__.__name__ == "BaseActiveObject":
                obj_state = self._serialize_object(obj)
                if obj_state:
                    objects.append(obj_state)
        
        # Sort by type and team for deterministic ordering
        objects.sort(key=lambda x: (x.get('type', ''), x.get('team', 0)))
        state['objects'] = objects
        
        return state
    
    def _serialize_object(self, obj) -> Optional[Dict[str, Any]]:
        """Serialize a game object."""
        try:
            return {
                'type': getattr(obj, 'type', None),
                'team': getattr(obj, 'team', None),
                'pos': list(getattr(obj, 'pos', [0, 0, 0])),
                'face': getattr(obj, 'face', 1),
                'current_state': getattr(obj, 'current_state', 'Stand'),
                'frame': list(getattr(obj, 'frame', [0, 0])),
                'speed': list(getattr(obj, 'speed', [0, 0])),
                'acceleration': list(getattr(obj, 'acceleration', [0, 0])),
                'con_speed': list(getattr(obj, 'con_speed', [0, 0])),
                'hitstop': getattr(obj, 'hitstop', 0),
                'hitstun': getattr(obj, 'hitstun', 0),
                'gauges': dict(getattr(obj, 'gauges', {})),
                'fet': getattr(obj, 'fet', 'grounded'),
                'current_command': list(getattr(obj, 'current_command', [])),
                'cancel': list(getattr(obj, 'cancel', [None])),
                'juggle': getattr(obj, 'juggle', 100),
                'wallbounce': getattr(obj, 'wallbounce', False),
                'damage_scaling': list(getattr(obj, 'damage_scaling', [100, 0])),
                'combo': getattr(obj, 'combo', 0),
                'last_damage': list(getattr(obj, 'last_damage', [0, 0])),
                'draw_shake': list(getattr(obj, 'draw_shake', [0, 0, 0, 0, 0, 0])),
                'repeat': getattr(obj, 'repeat', 0),
                'ignore_stop': getattr(obj, 'ignore_stop', False),
                'hold_on_stun': getattr(obj, 'hold_on_stun', False),
                'kara': getattr(obj, 'kara', 0),
                'buffer_state': dict(getattr(obj, 'buffer_state', {})),
                'command_index_timer': self._serialize_command_timer(getattr(obj, 'command_index_timer', {})),
            }
        except Exception as e:
            print(f"[StateManager] Error serializing object: {e}")
            return None
    
    def _serialize_command_timer(self, timer_dict: Dict) -> Dict:
        """Serialize command_index_timer dictionary."""
        result = {}
        for move, timers in timer_dict.items():
            if isinstance(timers, list):
                result[move] = [list(t) if isinstance(t, (list, tuple)) else t for t in timers]
            else:
                result[move] = timers
        return result
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of state data."""
        try:
            serialized = pickle.dumps(self.state_data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(serialized).hexdigest()
        except Exception as e:
            print(f"[StateManager] Error calculating checksum: {e}")
            return ""
    
    def apply_to_game(self, game):
        """Apply this state snapshot to the game."""
        try:
            # Restore game-level state
            game.emu_frame = self.state_data['frame']
            game.hitstop = self.state_data['hitstop']
            game.camera_focus_point = list(self.state_data['camera_focus_point'])
            game.superstop = self.state_data['superstop']
            if hasattr(game, 'camera_path'):
                game.camera_path = dict(self.state_data['camera_path'])
            if hasattr(game, 'frame'):
                game.frame = list(self.state_data['frame_data'])
            if hasattr(game, 'pos'):
                game.pos = list(self.state_data['pos'])
            if hasattr(game, 'draw_shake'):
                game.draw_shake = list(self.state_data['draw_shake'])
            
            # Restore object states
            for obj_state in self.state_data['objects']:
                obj = self._find_matching_object(game, obj_state)
                if obj:
                    self._deserialize_object(obj, obj_state)
                else:
                    print(f"[StateManager] Warning: Could not find object type={obj_state.get('type')}, team={obj_state.get('team')}")
        
        except Exception as e:
            print(f"[StateManager] Error applying state: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_matching_object(self, game, obj_state: Dict) -> Optional[Any]:
        """Find matching object in game.object_list."""
        obj_type = obj_state.get('type')
        obj_team = obj_state.get('team')
        
        for obj in game.object_list:
            if (hasattr(obj, '__class__') and 
                obj.__class__.__name__ == "BaseActiveObject" and
                hasattr(obj, 'type') and hasattr(obj, 'team') and
                obj.type == obj_type and
                obj.team == obj_team):
                return obj
        return None
    
    def _deserialize_object(self, obj, obj_state: Dict):
        """Deserialize object state."""
        try:
            if hasattr(obj, 'pos'):
                obj.pos = list(obj_state['pos'])
            if hasattr(obj, 'face'):
                obj.face = obj_state['face']
            if hasattr(obj, 'current_state'):
                obj.current_state = obj_state['current_state']
            if hasattr(obj, 'frame'):
                obj.frame = list(obj_state['frame'])
            if hasattr(obj, 'speed'):
                obj.speed = list(obj_state['speed'])
            if hasattr(obj, 'acceleration'):
                obj.acceleration = list(obj_state['acceleration'])
            if hasattr(obj, 'con_speed'):
                obj.con_speed = list(obj_state['con_speed'])
            if hasattr(obj, 'hitstop'):
                obj.hitstop = obj_state['hitstop']
            if hasattr(obj, 'hitstun'):
                obj.hitstun = obj_state['hitstun']
            if hasattr(obj, 'gauges'):
                obj.gauges = dict(obj_state['gauges'])
            if hasattr(obj, 'fet'):
                obj.fet = obj_state['fet']
            if hasattr(obj, 'current_command'):
                obj.current_command = list(obj_state['current_command'])
            if hasattr(obj, 'cancel'):
                obj.cancel = list(obj_state['cancel'])
            if hasattr(obj, 'juggle'):
                obj.juggle = obj_state['juggle']
            if hasattr(obj, 'wallbounce'):
                obj.wallbounce = obj_state['wallbounce']
            if hasattr(obj, 'damage_scaling'):
                obj.damage_scaling = list(obj_state['damage_scaling'])
            if hasattr(obj, 'combo'):
                obj.combo = obj_state['combo']
            if hasattr(obj, 'last_damage'):
                obj.last_damage = list(obj_state['last_damage'])
            if hasattr(obj, 'draw_shake'):
                obj.draw_shake = list(obj_state['draw_shake'])
            if hasattr(obj, 'repeat'):
                obj.repeat = obj_state['repeat']
            if hasattr(obj, 'ignore_stop'):
                obj.ignore_stop = obj_state['ignore_stop']
            if hasattr(obj, 'hold_on_stun'):
                obj.hold_on_stun = obj_state['hold_on_stun']
            if hasattr(obj, 'kara'):
                obj.kara = obj_state['kara']
            if hasattr(obj, 'buffer_state'):
                obj.buffer_state = dict(obj_state['buffer_state'])
            if hasattr(obj, 'command_index_timer'):
                obj.command_index_timer = dict(obj_state['command_index_timer'])
        except Exception as e:
            print(f"[StateManager] Error deserializing object: {e}")


class StateManager:
    """Manages game state snapshots and input history for rollback recovery."""
    
    def __init__(self, max_history: int = 120):
        """
        Initialize state manager.
        
        Args:
            max_history: Maximum number of state snapshots to keep (default: 120 = 2 seconds at 60fps)
        """
        self.max_history = max_history
        self.state_history: Dict[int, GameStateSnapshot] = {}  # {frame: snapshot}
        self.input_history: Dict[int, Tuple[List, List]] = {}  # {frame: (player1_input, player2_input)}
        self.save_interval = 3  # Save state every 3 frames (50ms at 60fps)
        self.last_saved_frame = -1
    
    def should_save_state(self, current_frame: int) -> bool:
        """Check if we should save state at this frame."""
        return (current_frame - self.last_saved_frame) >= self.save_interval
    
    def save_state(self, game, frame: int):
        """Save a state snapshot."""
        try:
            snapshot = GameStateSnapshot(game, frame)
            self.state_history[frame] = snapshot
            self.last_saved_frame = frame
            
            # Clean up old states
            if len(self.state_history) > self.max_history:
                oldest_frame = min(self.state_history.keys())
                del self.state_history[oldest_frame]
        except Exception as e:
            print(f"[StateManager] Error saving state: {e}")
    
    def save_inputs(self, frame: int, player1_input: List, player2_input: List):
        """Save inputs for a frame."""
        self.input_history[frame] = (player1_input, player2_input)
        
        # Clean up old inputs
        if len(self.input_history) > self.max_history:
            oldest_frame = min(self.input_history.keys())
            del self.input_history[oldest_frame]
    
    def get_state(self, frame: int) -> Optional[GameStateSnapshot]:
        """Get state snapshot for a specific frame."""
        return self.state_history.get(frame)
    
    def get_nearest_state(self, frame: int) -> Optional[GameStateSnapshot]:
        """Get the nearest state snapshot before or at the specified frame."""
        if not self.state_history:
            return None
        
        # Find the closest state at or before the frame
        candidates = [f for f in self.state_history.keys() if f <= frame]
        if not candidates:
            # If no state before frame, get earliest available
            candidates = list(self.state_history.keys())
        
        if candidates:
            nearest_frame = max(candidates)
            return self.state_history[nearest_frame]
        return None
    
    def rollback_to_frame(self, game, target_frame: int) -> bool:
        """
        Rollback game to a specific frame.
        
        Args:
            game: GameObject instance
            target_frame: Frame to rollback to
            
        Returns:
            True if rollback successful, False otherwise
        """
        current_frame = game.emu_frame
        
        # Get state snapshot
        snapshot = self.get_nearest_state(target_frame)
        if not snapshot:
            print(f"[StateManager] No state snapshot available for frame {target_frame}")
            return False
        
        print(f"[StateManager] Rolling back from frame {current_frame} to frame {snapshot.frame} (target was {target_frame})")
        
        # Restore state
        snapshot.apply_to_game(game)
        
        # Re-simulate from restored frame to current frame
        frames_to_resimulate = current_frame - snapshot.frame
        if frames_to_resimulate > 0:
            print(f"[StateManager] Re-simulating {frames_to_resimulate} frames from {snapshot.frame} to {current_frame}...")
            self._resimulate_frames(game, snapshot.frame, current_frame)
            print(f"[StateManager] ✓ Re-simulation complete")
        
        return True
    
    def _resimulate_frames(self, game, start_frame: int, end_frame: int):
        """Re-simulate frames from start_frame to end_frame using saved inputs."""
        from Util.Common_functions import update_display_shake, calculate_boxes_collitions
        
        for frame in range(start_frame + 1, end_frame + 1):  # Start from frame after snapshot
            # Get inputs for this frame
            inputs = self.input_history.get(frame)
            if not inputs:
                # No inputs saved - use neutral inputs
                player1_input = [[0,0],0,0,0,0,0,0,0,0,0,0]
                player2_input = [[0,0],0,0,0,0,0,0,0,0,0,0]
            else:
                player1_input, player2_input = inputs
            
            # Apply inputs to input devices
            if len(game.input_device_list) > 0:
                game.input_device_list[0].get_press(list(player1_input))
            if len(game.input_device_list) > 1:
                game.input_device_list[1].get_press(list(player2_input))
            
            # Simulate frame (same logic as main gameplay loop)
            game.emu_frame = frame
            for obj in game.object_list:
                obj.update(game.camera_focus_point)
            game.hitstop = game.hitstop - 1 if game.hitstop else 0
            
            for obj in game.object_list:
                update_display_shake(obj)
            calculate_boxes_collitions(game)
            update_display_shake(game.camera)
            game.calculate_camera_focus_point()
            
            # Save state after each frame (for nested rollbacks)
            if self.should_save_state(frame):
                self.save_state(game, frame)

