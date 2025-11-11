import sys
import os
# Add the ComputerVision directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_track_pose import record_pose_data
import numpy as np

# For movement detection: track previous position
prev_center_x = None
movement_threshold = 0.01  # Threshold for detecting movement (normalized coordinates)

# MediaPipe pose landmark indices (original, before filtering)
# Body and face landmarks only (excluding arms and legs)
# Face: 0 (nose), 2, 5, 7, 8, 9, 10
# Body: 11 (left shoulder), 12 (right shoulder), 23 (left hip), 24 (right hip)
body_face_original_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 23, 24]

# Indices that are removed in detect_track_pose.py
remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

# Calculate which positions in the filtered array correspond to body/face landmarks
# We need to map original indices to filtered array positions
body_face_filtered_positions = []
filtered_position = 0
for i in range(33):  # MediaPipe has 33 landmarks
    if i not in remove_indices:
        if i in body_face_original_indices:
            body_face_filtered_positions.append(filtered_position)
        filtered_position += 1

# Shared dict to pass movement info to display on screen
movement_info = {'direction': 'STATIONARY', 'x_diff': 0.0}

#gets the keypoint vector out of the loop, to feed in a model in the future
for pose_vector in record_pose_data(mode_test=True, movement_info=movement_info):
    # feed pose_vector to your model in real time
    
    # Detect left/right movement from pose vector
    # The vector format is: [x0, y0, z0, v0, x1, y1, z1, v1, ...]
    # Extract all x-coordinates (every 4th element starting from index 0)
    x_coords = pose_vector[0::4]  # Get x coordinates (indices 0, 4, 8, 12, ...)
    visibility = pose_vector[3::4]  # Get visibility values (indices 3, 7, 11, 15, ...)
    
    # Filter to only body/face landmarks (exclude arms and legs)
    body_face_x_coords = [x_coords[i] for i in body_face_filtered_positions if i < len(x_coords)]
    body_face_visibility = [visibility[i] for i in body_face_filtered_positions if i < len(visibility)]
    
    # Calculate center x-coordinate using only visible body/face landmarks
    visible_x_coords = [x for x, v in zip(body_face_x_coords, body_face_visibility) if v > 0.5]
    
    if visible_x_coords:
        center_x = np.mean(visible_x_coords)
        
        # Determine movement direction
        movement_direction = "STATIONARY"
        x_diff = 0.0
        if prev_center_x is not None:
            x_diff = center_x - prev_center_x
            if abs(x_diff) > movement_threshold:
                if x_diff > 0:
                    movement_direction = "MOVING RIGHT"
                else:
                    movement_direction = "MOVING LEFT"
        
        # Update previous position
        prev_center_x = center_x
        
        # Update shared movement info for display on screen
        movement_info['direction'] = movement_direction
        movement_info['x_diff'] = x_diff
        
        # Display movement direction (optional - also prints to console)
        print(f"Movement: {movement_direction}")
    else:
        # Reset tracking if no visible landmarks
        prev_center_x = None
        movement_info['direction'] = "NO POSE DETECTED"
        movement_info['x_diff'] = 0.0
        print("NO POSE DETECTED")
