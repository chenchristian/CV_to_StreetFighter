"""
Real-time action classification using trained LSTM model.
Shows webcam feed with predicted actions displayed on screen.
"""
import sys
import os
import cv2 as cv
import numpy as np

# Add the ComputerVision directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_track_pose import record_pose_data
from lstm_predictor import LSTMPosePredictor


def realtime_classify():
    """
    Real-time action classification from webcam.
    Displays the video feed with predicted actions.
    """
    # Initialize LSTM predictor
    try:
        predictor = LSTMPosePredictor()
        print("LSTM model loaded successfully!")
        print("Press ESC to exit.\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_lstm.py")
        return
    
    # Shared info dict for displaying predictions on screen
    prediction_info = {
        'action': 'idle',
        'confidence': 0.0,
        'direction': '',  # For compatibility with detect_track_pose
        'x_diff': 0.0,
        'frame_count': 0
    }
    
    print("Starting real-time classification...")
    print("Perform actions (punch, kick, idle) in front of the camera.")
    print("Predictions will be displayed on the video feed.\n")
    
    # Main loop - get pose vectors from camera
    for pose_vector in record_pose_data(mode_test=True, movement_info=prediction_info):
        try:
            # Predict action from current pose vector
            action, confidence = predictor.predict(pose_vector)
            
            # Update prediction info (will be displayed on screen by detect_track_pose)
            prediction_info['action'] = action
            prediction_info['confidence'] = confidence
            prediction_info['frame_count'] += 1
            
            # Print to console (optional, every 30 frames to reduce spam)
            if prediction_info['frame_count'] % 30 == 0:
                print(f"Frame {prediction_info['frame_count']:4d} | Action: {action:8s} | Confidence: {confidence:.1%}")
        
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction_info['action'] = 'error'
            prediction_info['confidence'] = 0.0


if __name__ == "__main__":
    realtime_classify()

