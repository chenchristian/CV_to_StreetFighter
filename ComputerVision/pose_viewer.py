# ComputerVision/pose_viewer.py
import cv2 as cv
import numpy as np
import queue
import mediapipe as mp
from .pose_recorder_back_thread import start_cv_worker

mp_pose = mp.solutions.pose
remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

class PoseViewer:
    def __init__(self, queue_maxsize=2, movement_info=None):
        self.pose_q = queue.Queue(maxsize=queue_maxsize)
        self.last_frame = None
        self.last_landmarks = []
        self._window_created = False
        self._running = False
        self._movement_info = movement_info

    def start_recorder(self):
        if self._running:
            return
        start_cv_worker(self.pose_q, movement_info=self._movement_info)
        self._running = True

    def _ensure_window(self):
        if not self._window_created:
            cv.namedWindow("Pose Debug Window", cv.WINDOW_NORMAL)
            self._window_created = True

    def poll(self):
        self._ensure_window()

        # Get latest frame from queue
        try:
            vector, rgb_frame, landmarks = self.pose_q.get_nowait()
            self.last_frame = rgb_frame.copy()
            self.last_landmarks = landmarks
        except queue.Empty:
            pass

        if self.last_frame is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img = cv.cvtColor(self.last_frame, cv.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # Draw keypoints
        if self.last_landmarks:
            existing_indices = [i for i in range(len(self.last_landmarks)) if i not in remove_indices]

            for i in existing_indices:
                x, y = self.last_landmarks[i]
                cx, cy = int(x * w), int(y * h)
                cv.circle(img, (cx, cy), 4, (0, 255, 0), -1)

            # Draw skeleton connections
            connections = [
                (start_idx, end_idx)
                for start_idx, end_idx in mp_pose.POSE_CONNECTIONS
                if start_idx in existing_indices and end_idx in existing_indices
            ]

            for start_idx, end_idx in connections:
                x1, y1 = self.last_landmarks[start_idx]
                x2, y2 = self.last_landmarks[end_idx]
                cv.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 2)

        cv.imshow("Pose Debug Window", img)
        if cv.waitKey(1) & 0xFF == 27:
            self.close()

    def close(self):
        try:
            cv.destroyWindow("Pose Debug Window")
        except Exception:
            pass
        self._running = False
        self._window_created = False
