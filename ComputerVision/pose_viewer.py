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
        else:
            self.last_frame = None
            self.last_landmarks = []
            movement_info = {"direction": "STATIONARY", "x_diff": 0.0}

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


        # overlay movement info
        cv.putText(img, f"Direction: {movement_info['direction']}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(img, f"X diff: {movement_info['x_diff']:.3f}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv.imshow("Pose Debug Window", img)
        if cv.waitKey(1) & 0xFF == 27:
            self.close()

    def close(self):
        try:
            cv.destroyWindow("Pose Debug Window")
        except Exception:
            pass
