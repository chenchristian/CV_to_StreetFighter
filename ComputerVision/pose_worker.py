# pose_worker.py
import threading
import queue
import time
import cv2 as cv
import mediapipe as mp
import numpy as np

class PoseWorker:
    def __init__(self,
                 camera_index=0,
                 width=640,
                 height=480,
                 movement_threshold=0.01,
                 subscriber_queue_maxsize=2):
        # MediaPipe & filtering config
        self.mp_pose = mp.solutions.pose
        self.remove_indices = [1,3,4,6,17,18,19,20,21,22,31,32]
        self.body_face_original_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
        self.movement_threshold = movement_threshold

        # camera config
        self.camera_index = camera_index
        self.width = width
        self.height = height

        # thread control
        self._thread = None
        self._stop_event = threading.Event()

        # latest snapshot (protected by lock)
        # keys: vector, frame_rgb, landmarks, movement_info, game_input, timestamp
        self._latest = {}
        self._lock = threading.Lock()

        # subscriber queues: list of queue.Queue objects
        self._subs = []
        self._sub_maxsize = subscriber_queue_maxsize

        # precompute mapping from original→filtered positions
        self.body_face_filtered_positions = []
        filtered_position = 0
        for i in range(33):
            if i not in self.remove_indices:
                if i in self.body_face_original_indices:
                    self.body_face_filtered_positions.append(filtered_position)
                filtered_position += 1

    # --- public API -----------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def add_subscriber(self):
        """Returns a queue.Queue you can read from (non-blocking get_nowait recommended)."""
        q = queue.Queue(maxsize=self._sub_maxsize)
        self._subs.append(q)
        return q

    def remove_subscriber(self, q):
        try:
            self._subs.remove(q)
        except ValueError:
            pass

    def get_latest_snapshot(self):
        """Return a shallow copy of the latest snapshot dict (or None)."""
        with self._lock:
            return None if not self._latest else dict(self._latest)

    def get_latest_game_input(self):
        snap = self.get_latest_snapshot()
        if not snap:
            # default: standing still (same as your InputDevice expectation)
            return [[0,0], 0,0,0,0,0,0,0,0]
        return snap.get("game_input", [[0,0],0,0,0,0,0,0,0,0])

    # --- internal runner ------------------------------------------------
    def _run(self):
        cap = cv.VideoCapture(self.camera_index)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as pose:
            prev_center_x = None

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # small sleep avoids busy loop on camera failure
                    time.sleep(0.01)
                    continue

                # flip & rgb for MediaPipe & drawing consistency
                frame = cv.flip(frame, 1)
                img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                # prepare defaults
                vector = np.zeros((33 - len(self.remove_indices)) * 4, dtype=np.float32)
                landmarks_for_drawing = []
                movement_info = {"direction": "NO POSE", "x_diff": 0.0}
                game_input = [[0,0],0,0,0,0,0,0,0,0]  # default: stationary
                timestamp = time.time()

                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    # filtered landmarks for ML vector
                    lms_filtered = [lm for i, lm in enumerate(lms) if i not in self.remove_indices]
                    vector = np.array(
                        [coord for lm in lms_filtered for coord in (lm.x, lm.y, lm.z, lm.visibility)],
                        dtype=np.float32
                    )
                    # landmarks for drawing: keep all 33 normalized (x,y)
                    landmarks_for_drawing = [(lm.x, lm.y) for lm in lms]

                    # compute center_x from visible body_face landmarks
                    body_face_x_coords = [
                        vector[i*4] for i in self.body_face_filtered_positions if i*4 < len(vector)
                    ]
                    body_face_visibility = [
                        vector[i*4 + 3] for i in self.body_face_filtered_positions if i*4 + 3 < len(vector)
                    ]
                    visible_x_coords = [x for x, v in zip(body_face_x_coords, body_face_visibility) if v > 0.5]

                    if visible_x_coords:
                        center_x = float(np.mean(visible_x_coords))
                        if prev_center_x is None:
                            # First frame with visible landmarks
                            x_diff = 0.0
                            movement = "STATIONARY"
                            game_input = [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            x_diff = center_x - prev_center_x
                            if x_diff > self.movement_threshold:
                                movement = "MOVING RIGHT"
                                game_input = [[1, 0], 0, 0, 0, 0, 0, 0, 0, 0]
                            elif x_diff < -self.movement_threshold:
                                movement = "MOVING LEFT"
                                game_input = [[-1, 0], 0, 0, 0, 0, 0, 0, 0, 0]
                            else:
                                movement = "STATIONARY"
                                game_input = [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0]

                        movement_info["direction"] = movement
                        movement_info["x_diff"] = x_diff
                        prev_center_x = center_x
                    else:
                        # No visible landmarks
                        prev_center_x = None
                        movement_info["direction"] = "NO POSE"
                        movement_info["x_diff"] = 0.0
                        game_input = [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0]


                # build snapshot
                snapshot = {
                    "vector": vector,
                    "frame_rgb": img_rgb,                   # normalized RGB
                    "landmarks": landmarks_for_drawing,     # list of (x,y) normalized
                    "movement_info": movement_info,         # dict
                    "game_input": game_input,               # ready for InputDevice
                    "timestamp": timestamp
                }

                # store latest
                with self._lock:
                    self._latest = snapshot

                # publish to subscribers (non-blocking put)
                for q in list(self._subs):
                    try:
                        # replace oldest item if queue full (so subscribers always get recent)
                        if q.full():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass
                        q.put_nowait(snapshot)
                    except Exception:
                        # if subscriber queue has been closed/removed, ignore and continue
                        pass

                # short sleep to avoid starving CPU if your MediaPipe is fast
                # (tune if needed; MP's processing time typically dominates)
                time.sleep(0.001)

        cap.release()
