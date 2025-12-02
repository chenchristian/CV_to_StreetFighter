# pose_viewer.py
import cv2 as cv
import numpy as np
from .pose_worker import PoseWorker

class PoseViewer:
    def __init__(self, shared_state: PoseWorker):
        self.shared_state = shared_state
        self._window_created = False
        self.last_frame = None
        self.last_landmarks = []

    def _ensure_window(self):
        if not self._window_created:
            cv.namedWindow("Pose Debug Window", cv.WINDOW_NORMAL)
            self._window_created = True

    def poll(self):
        self._ensure_window()

        snapshot = self.shared_state.get_latest_snapshot()
        if snapshot:
            self.last_frame = snapshot["frame_rgb"].copy()
            self.last_landmarks = snapshot["landmarks"]
            movement_info = snapshot["movement_info"]
            pred_label = snapshot["prediction"]
            confidence = snapshot["confidence"]
        else:
            self.last_frame = None
            self.last_landmarks = []
            movement_info = {"direction": "STATIONARY", "x_diff": 0.0}
            pred_label = None
            confidence = None

        if self.last_frame is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img = cv.cvtColor(self.last_frame, cv.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        remove_indices = [1,3,4,6,17,18,19,20,21,22,31,32]
        for i, (x, y) in enumerate(self.last_landmarks):
            #print(len(self.last_landmarks)) make sure its 33
            if i not in remove_indices:
                cx, cy = int(x * w), int(y * h)
                cv.circle(img, (cx, cy), 4, (0, 255, 0), -1)

        skeleton_connections = [
            # Arms
            (11, 13),  # Left Shoulder → Left Elbow
            (13, 15),  # Left Elbow → Left Wrist
            (12, 14),  # Right Shoulder → Right Elbow
            (14, 16),  # Right Elbow → Right Wrist

            # Torso
            (11, 12),  # Left Shoulder → Right Shoulder
            (11, 23),  # Left Shoulder → Left Hip
            (12, 24),  # Right Shoulder → Right Hip
            (23, 24),  # Left Hip → Right Hip

            # Legs
            (23, 25),  # Left Hip → Left Knee
            (25, 27),  # Left Knee → Left Ankle
            (27, 29),  # Left Ankle → Left Heel
            #(29, 31),  # Left Heel → Left Foot Index

            (24, 26),  # Right Hip → Right Knee
            (26, 28),  # Right Knee → Right Ankle
            (28, 30),  # Right Ankle → Right Heel
            #(30, 32),  # Right Heel → Right Foot Index
        ]

        for start, end in skeleton_connections:
            if start < len(self.last_landmarks) and end < len(self.last_landmarks):
                x1, y1 = self.last_landmarks[start]
                x2, y2 = self.last_landmarks[end]

                cv.line(
                    img,
                    (int(x1 * w), int(y1 * h)),
                    (int(x2 * w), int(y2 * h)),
                    (255, 0, 0),
                    2
                )

        if pred_label is not None and confidence is not None:
            # Get prediction probabilities for all classes
            labels = ["punch", "kick", "idle"]
            
            # Draw bars for each action
            bar_x = 10
            bar_y_start = 90
            bar_width = 200
            bar_height = 25
            bar_spacing = 35
            
            for i, label in enumerate(labels):
                y_pos = bar_y_start + i * bar_spacing
                
                # Background bar
                cv.rectangle(img, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), 
                           (50, 50, 50), -1)
                
                # Confidence bar (assuming confidence is a list/array of probabilities)
                if isinstance(confidence, (list, np.ndarray)) and len(confidence) == 3:
                    conf_value = confidence[i]
                else:
                    # If single value, only show for predicted label
                    conf_value = confidence if label == pred_label else 0.0
                
                fill_width = int(bar_width * conf_value)
                color = (0, 255, 0) if label == pred_label else (100, 100, 255)
                cv.rectangle(img, (bar_x, y_pos), (bar_x + fill_width, y_pos + bar_height), 
                           color, -1)
                
                # Label and percentage
                cv.putText(img, f"{label}: {conf_value:.2%}", (bar_x + 5, y_pos + 18),
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # if pred_label is not None and confidence is not None:
        #     cv.putText(img, f"Action: {pred_label} ({confidence:.2f})", (10, 90),
        #             cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        # overlay movement info
        cv.putText(img, f"Direction: {movement_info['direction']}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(img, f"X diff: {movement_info['x_diff']:.3f}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Draw edge indicator zones
        center_x = movement_info.get("center_x")
        if center_x is not None:
            zone_width = int(0.1 * w)  # 10% of screen width
            
            # Left edge zone
            if center_x <= 0.1:
                # Highlight left edge zone with semi-transparent green overlay
                overlay = img.copy()
                cv.rectangle(overlay, (0, 0), (zone_width, h), (0, 255, 0), -1)
                cv.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            else:
                # Dim left edge zone when not active
                overlay = img.copy()
                cv.rectangle(overlay, (0, 0), (zone_width, h), (50, 50, 50), -1)
                cv.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            
            # Right edge zone
            if center_x >= 0.9:
                # Highlight right edge zone with semi-transparent green overlay
                overlay = img.copy()
                cv.rectangle(overlay, (w - zone_width, 0), (w, h), (0, 255, 0), -1)
                cv.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            else:
                # Dim right edge zone when not active
                overlay = img.copy()
                cv.rectangle(overlay, (w - zone_width, 0), (w, h), (50, 50, 50), -1)
                cv.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        cv.imshow("Pose Debug Window", img)
        if cv.waitKey(1) & 0xFF == 27:
            self.close()

    def close(self):
        try:
            cv.destroyWindow("Pose Debug Window")
        except Exception:
            pass
