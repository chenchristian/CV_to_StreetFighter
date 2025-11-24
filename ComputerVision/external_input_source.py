import sys
import os
import numpy as np

# Add the current directory so we can import detect_track_pose
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_recorder_back_thread import record_pose_data


def left_right():
    """
    Continuously yields game-ready keyboard_input vectors derived from MediaPipe pose data.
    Output format matches what InputDevice.get_press() expects.

    Yields:
        raw_input   : Street Fighter directional input vector
        movement_info : dict with {'direction': str, 'x_diff': float}
    """

    prev_center_x = None
    movement_threshold = 0.01  # sensitivity threshold for movement LEFT/RIGHT

    # Keep only important body/face landmarks
    body_face_original_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 23, 24]

    # Indices removed from MediaPipe output
    remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

    # Map original indices → filtered vector positions
    body_face_filtered_positions = []
    filtered_position = 0
    for i in range(33):
        if i not in remove_indices:
            if i in body_face_original_indices:
                body_face_filtered_positions.append(filtered_position)
            filtered_position += 1

    # Shared movement info dictionary
    movement_info = {
        'direction': 'STATIONARY',
        'x_diff': 0.0
    }

    # MAIN LOOP: consume from record_pose_data()
    for vector, frame_surface_data, landmarks_list in record_pose_data(
        mode_test=True,
        movement_info=movement_info
    ):
        # --------------------------
        # Extract x-coordinates
        # --------------------------
        x_coords = vector[0::4]          # x positions
        visibility = vector[3::4]        # visibility values

        # Important body/face landmarks only
        body_face_x_coords = [
            x_coords[i] for i in body_face_filtered_positions
            if i < len(x_coords)
        ]
        body_face_visibility = [
            visibility[i] for i in body_face_filtered_positions
            if i < len(visibility)
        ]

        # Visible landmarks only
        visible_x_coords = [
            x for x, v in zip(body_face_x_coords, body_face_visibility)
            if v > 0.5
        ]

        # Movement defaults
        movement_direction = "STATIONARY"
        x_diff = 0.0

        # --------------------------
        # Compute movement direction
        # --------------------------
        if visible_x_coords:
            center_x = np.mean(visible_x_coords)

            if prev_center_x is not None:
                x_diff = center_x - prev_center_x

                if abs(x_diff) > movement_threshold:
                    movement_direction = "MOVING RIGHT" if x_diff > 0 else "MOVING LEFT"

            prev_center_x = center_x

        else:
            # No visible pose — reset tracking
            prev_center_x = None
            movement_direction = "NO POSE DETECTED"

        # Update the shared dictionary
        movement_info['direction'] = movement_direction
        movement_info['x_diff'] = float(x_diff)

        # --------------------------
        # Convert to Street Fighter directional input
        # --------------------------
        if movement_direction == "MOVING RIGHT":
            raw_input = [[1, 0], 0, 0, 0, 0, 0, 0, 0, 0]

        elif movement_direction == "MOVING LEFT":
            raw_input = [[-1, 0], 0, 0, 0, 0, 0, 0, 0, 0]

        else:
            raw_input = [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0]

        # --------------------------
        # FINAL OUTPUT
        # --------------------------
        yield raw_input, dict(movement_info)
