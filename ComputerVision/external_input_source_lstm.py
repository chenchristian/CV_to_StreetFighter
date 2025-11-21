import sys
import os
import numpy as np

# Add the ComputerVision directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_track_pose import record_pose_data
from lstm_predictor import LSTMPosePredictor


def left_right():
    """
    Continuously yields game-ready keyboard_input vectors
    derived from LSTM pose classification and movement detection.
    Combines left/right movement with punch/kick/idle actions.
    Each output matches the format expected by InputDevice.get_press().
    
    Format: [[x, y], button1, button2, button3, button4, button5, button6, button7, button8]
    - [x, y]: movement (x: -1=left, 0=neutral, 1=right; y: -1=down, 0=neutral, 1=up)
    - buttons: 1=light punch, 2=medium punch, 3=heavy punch, 4=light kick, 5=medium kick, 6=heavy kick
    """
    # Initialize LSTM predictor
    try:
        predictor = LSTMPosePredictor()
        print("✓ LSTM model loaded successfully!")
    except FileNotFoundError as e:
        print(f"⚠ Error: {e}")
        print("  Falling back to movement-only detection...")
        predictor = None
    
    # Shared frame info for display (movement is calculated in detect_track_pose.py)
    movement_info = {'direction': 'STATIONARY', 'x_diff': 0.0, 'action': 'idle', 'confidence': 0.0}
    
    # Main generator loop
    for pose_vector in record_pose_data(mode_test=True, movement_info=movement_info, use_lstm=True):
        # Get movement direction from movement_info (already calculated in detect_track_pose.py)
        direction = movement_info.get('direction', 'STATIONARY')
        
        # Convert movement direction to game input format
        # Format: [x, y] where x: -1=left, 0=neutral, 1=right
        if "RIGHT" in direction:
            movement = [1, 0]  # Move right
        elif "LEFT" in direction:
            movement = [-1, 0]  # Move left
        else:
            movement = [0, 0]  # Stationary or no pose
        
        # Get LSTM action prediction (already set in movement_info by detect_track_pose.py)
        action = movement_info.get('action', 'idle')
        confidence = movement_info.get('confidence', 0.0)
        
        # Convert action to button presses
        # Only trigger if confidence is above threshold (e.g., 0.5) to avoid false positives
        confidence_threshold = 0.5
        buttons = [0, 0, 0, 0, 0, 0, 0, 0]  # [light_punch, med_punch, heavy_punch, light_kick, med_kick, heavy_kick, ...]
        
        if confidence >= confidence_threshold:
            # Swap action labels: if model predicts "punch" when you do kick, swap them
            if action == "punch":
                buttons[3] = 1  # Map "punch" prediction to kick button
            elif action == "kick":
                buttons[0] = 1  # Map "kick" prediction to punch button
            # idle: buttons remain all 0
        
        # Combine movement and actions
        raw_input = [movement] + buttons
        
        yield raw_input, dict(movement_info)


if __name__ == "__main__":
    # Test the LSTM-based input source
    print("Testing LSTM-based input source...")
    print("Press ESC to stop.")
    
    for raw_input, info in left_right():
        print(f"Action: {info['action']}, Confidence: {info['confidence']:.2f}")

