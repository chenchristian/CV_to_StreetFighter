import cv2 as cv
import csv
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]
movement_threshold = 0.05

# Try to import LSTM predictor (optional)
try:
    # Import from same directory
    from lstm_predictor import LSTMPosePredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("Warning: LSTM predictor not available. Action classification will be disabled.")

def record_pose_data(mode_test=False, csv_filename="pose_data.csv", movement_info=None, prediction_callback=None, use_lstm=True):
    """
    Capture pose data from webcam, save to CSV or return vectors, 
    record video (only if mode_test=False), and display movement info.
    
    Args:
        mode_test: If True, yield pose vectors instead of saving to CSV
        csv_filename: Output CSV filename (only used if mode_test=False)
        movement_info: Dictionary to store movement/action info (optional)
        prediction_callback: Optional callback function for predictions (deprecated, use use_lstm instead)
        use_lstm: If True, load and use LSTM model for action classification
    """
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    pose_vectors = []

    if not mode_test:
        # Video writer for output
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_writer = cv.VideoWriter("pose_recording.mp4", fourcc, 20, (640, 480))

        # CSV writer
        csv_file = open(csv_filename, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        header = []
        for i in range(33 - len(remove_indices)):
            header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
        csv_writer.writerow(header)

    # Initialize movement_info if not provided
    if movement_info is None:
        movement_info = {"direction": "", "x_diff": 0.0, "action": "idle", "confidence": 0.0}
    prev_center_x = None
    prediction_text = ""
    
    # Load LSTM model if available and requested
    lstm_predictor = None
    if use_lstm and LSTM_AVAILABLE:
        try:
            # Get project root directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, "Model", "pose_lstm_model.pth")
            encoder_path = os.path.join(project_root, "Model", "label_encoder.pkl")
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                lstm_predictor = LSTMPosePredictor(model_path=model_path, encoder_path=encoder_path)
                print("✓ LSTM model loaded successfully! Action classification enabled.")
            else:
                print(f"⚠ LSTM model files not found. Expected:")
                print(f"  - {model_path}")
                print(f"  - {encoder_path}")
                print("  Action classification disabled.")
        except Exception as e:
            print(f"⚠ Error loading LSTM model: {e}")
            print("  Action classification disabled.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            frame_resized = cv.resize(frame, (640, 480))
            h, w = frame_resized.shape[:2]
            img_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                lms_filtered = [lm for i, lm in enumerate(lms) if i not in remove_indices]

                # Draw landmarks
                for lm in lms_filtered:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame_resized, (cx, cy), 5, (0, 255, 0), -1)

                # Draw connections
                connections = [c for c in mp_pose.POSE_CONNECTIONS
                               if c[0] not in remove_indices and c[1] not in remove_indices]
                for start_idx, end_idx in connections:
                    start = lms[start_idx]
                    end = lms[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Pose vector
                vector = np.array([coord for lm in lms_filtered for coord in (lm.x, lm.y, lm.z, lm.visibility)], dtype=np.float32)
                
                # Get LSTM prediction if model is loaded
                if lstm_predictor is not None:
                    try:
                        action, confidence = lstm_predictor.predict(vector)
                        movement_info['action'] = action
                        movement_info['confidence'] = confidence
                    except Exception as e:
                        # If prediction fails, set default values
                        movement_info['action'] = "idle"
                        movement_info['confidence'] = 0.0
                
                # Get prediction from callback if provided (for backward compatibility)
                if mode_test and prediction_callback:
                    prediction_text = prediction_callback(vector)

                if mode_test:
                    pose_vectors.append(vector)
                    yield vector
                else:
                    csv_writer.writerow(vector)
                    video_writer.write(frame_resized)

                # Calculate movement info (left/right)
                center_x = np.mean([lm.x for lm in lms_filtered])
                if prev_center_x is not None:
                    x_diff = center_x - prev_center_x
                    movement_info['x_diff'] = x_diff
                    if x_diff > movement_threshold:
                        movement_info['direction'] = "MOVE RIGHT"
                    elif x_diff < -movement_threshold:
                        movement_info['direction'] = "MOVE LEFT"
                    else:
                        movement_info['direction'] = "STATIONARY"
                prev_center_x = center_x

            else:
                movement_info['direction'] = "NO POSE DETECTED"
                movement_info['x_diff'] = 0.0
                # Reset action when no pose detected
                if 'action' in movement_info:
                    movement_info['action'] = "idle"
                    movement_info['confidence'] = 0.0

            # Display movement info
            direction = movement_info.get('direction', '')
            color = (0, 255, 255)
            if "RIGHT" in direction:
                color = (0, 255, 0)
            elif "LEFT" in direction or "NO POSE" in direction:
                color = (0, 0, 255)

            y_offset = 40
            if direction:
                cv.putText(frame_resized, direction, (20, y_offset), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                y_offset += 50
            
            # Display prediction from callback if provided (for backward compatibility)
            if mode_test and prediction_text:
                if prediction_text == "kick":
                    pred_color = (255, 0, 0)
                elif prediction_text == "punch":
                    pred_color = (0, 0, 255)
                else:
                    pred_color = (255, 255, 255)
                cv.putText(frame_resized, f"Action: {prediction_text}", (20, y_offset), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
                y_offset += 50
            
            # Display LSTM action prediction if available
            if 'action' in movement_info:
                action = movement_info['action'].upper()
                confidence = movement_info.get('confidence', 0.0)
                
                # Color based on action
                if action == "PUNCH":
                    action_color = (0, 165, 255)  # Orange
                elif action == "KICK":
                    action_color = (255, 0, 255)  # Magenta
                else:  # idle
                    action_color = (128, 128, 128)  # Gray
                
                # Display action and confidence
                action_text = f"Action: {action}"
                cv.putText(frame_resized, action_text, (20, y_offset),
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, action_color, 3)
                
                conf_text = f"Confidence: {confidence:.1%}"
                cv.putText(frame_resized, conf_text, (20, y_offset + 50),
                          cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                y_offset += 100
            
            # Display X diff if available
            if 'x_diff' in movement_info:
                cv.putText(frame_resized, f"X Diff: {movement_info['x_diff']:.4f}", (20, y_offset),
                          cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv.imshow("Pose Recording", frame_resized)

            if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    if not mode_test:
        video_writer.release()
        csv_file.close()
    cv.destroyAllWindows()
    if mode_test:
        return


if __name__ == "__main__":
    print("="*60)
    print("Pose Detection with Action Classification")
    print("="*60)
    print("Press ESC to stop.")
    print()

    # Run with LSTM model enabled
    for _ in record_pose_data(mode_test=False, csv_filename="pose_data.csv", use_lstm=True):
        pass

    print("\nRecording stopped. Video saved as 'pose_recording.mp4' and CSV saved as 'pose_data.csv'.")
