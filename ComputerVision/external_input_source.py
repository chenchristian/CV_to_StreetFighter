import sys
import os
import numpy as np

# Add the ComputerVision directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_track_pose import record_pose_data


def left_right():
    """
    Continuously yields game-ready keyboard_input vectors
    derived from MediaPipe pose data.
    Each output matches the format expected by InputDevice.get_press().
    """
    prev_center_x = None
    movement_threshold = 0.01  # sensitivity threshold

    # MediaPipe landmark setup
    body_face_original_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 23, 24]
    remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

    # Map from original MediaPipe indices → filtered vector positions
    body_face_filtered_positions = []
    filtered_position = 0
    for i in range(33):
        if i not in remove_indices:
            if i in body_face_original_indices:
                body_face_filtered_positions.append(filtered_position)
            filtered_position += 1

    # Shared frame info
    movement_info = {'direction': 'STATIONARY', 'x_diff': 0.0}

    # Main generator loop
    for pose_vector in record_pose_data(mode_test=True, movement_info=movement_info):
        # Extract x coordinates and visibility
        x_coords = pose_vector[0::4]
        visibility = pose_vector[3::4]

        # Select body/face landmarks only
        body_face_x_coords = [x_coords[i] for i in body_face_filtered_positions if i < len(x_coords)]
        body_face_visibility = [visibility[i] for i in body_face_filtered_positions if i < len(visibility)]

        # Keep only visible landmarks
        visible_x_coords = [x for x, v in zip(body_face_x_coords, body_face_visibility) if v > 0.5]

        # Default output
        movement_direction = "STATIONARY"
        x_diff = 0.0

        # Compute movement direction
        if visible_x_coords:
            center_x = np.mean(visible_x_coords)
            if prev_center_x is not None:
                x_diff = center_x - prev_center_x
                if abs(x_diff) > movement_threshold:
                    movement_direction = "MOVING RIGHT" if x_diff > 0 else "MOVING LEFT"
            prev_center_x = center_x
        else:
            prev_center_x = None
            movement_direction = "NO POSE DETECTED"

        # Update movement info
        movement_info['direction'] = movement_direction
        movement_info['x_diff'] = float(x_diff)

        # ✅ Convert to Street Fighter input format
        if movement_direction == "MOVING RIGHT":
            raw_input = [[1, 0], 0, 0, 0, 0, 0, 0, 0, 0]
        elif movement_direction == "MOVING LEFT":
            raw_input = [[-1, 0], 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raw_input = [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0]

        # 🎯 Yield both (game uses raw_input, you can also inspect movement_info)
        yield raw_input, dict(movement_info)