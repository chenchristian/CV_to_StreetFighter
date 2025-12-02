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
                 subscriber_queue_maxsize=2,
                 live_predictor=None): 
        # MediaPipe & filtering config
        self.mp_pose = mp.solutions.pose
        self.remove_indices = [1,3,4,6,17,18,19,20,21,22,31,32]
        self.body_face_original_indices = [
            0,2,5,7,8,9,10,11,12,13,14,15,16,
            23,24,25,26,27,28,29,30
        ]
        self.movement_threshold = movement_threshold

        # camera config
        self.camera_index = camera_index
        self.width = width
        self.height = height

        # thread control
        self._thread = None
        self._stop_event = threading.Event()

        # latest snapshot
        self._latest = {}
        self._lock = threading.Lock()

        # subscriber queues
        self._subs = []
        self._sub_maxsize = subscriber_queue_maxsize

        # live predictor
        self.live_predictor = live_predictor

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
        q = queue.Queue(maxsize=self._sub_maxsize)
        self._subs.append(q)
        return q

    def remove_subscriber(self, q):
        try:
            self._subs.remove(q)
        except ValueError:
            pass

    def get_latest_snapshot(self):
        with self._lock:
            return None if not self._latest else dict(self._latest)

    def get_latest_game_input(self):
        snap = self.get_latest_snapshot()
        if not snap:
            return [[0,0],0,0,0,0,0,0,0,0]
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
                    time.sleep(0.01)
                    continue

                frame = cv.flip(frame, 1)
                img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                # defaults
                vector = np.zeros((33 - len(self.remove_indices)) * 4, dtype=np.float32)
                landmarks_for_drawing = []
                movement_info = {"direction": "NO POSE", "x_diff": 0.0, "center_x": None}
                left_right_up_down = [[0,0]]
                attacks = [0,0,0,0,0,0,0,0]
                timestamp = time.time()

                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    lms_filtered = [lm for i, lm in enumerate(lms) if i not in self.remove_indices]
                    vector = np.array(
                        [coord for lm in lms_filtered for coord in (lm.x, lm.y, lm.z, lm.visibility)],
                        dtype=np.float32
                    )
                    landmarks_for_drawing = [(lm.x, lm.y) for lm in lms]

                    # Movement detection
                    body_face_x_coords = [vector[i*4] for i in self.body_face_filtered_positions if i*4 < len(vector)]
                    body_face_visibility = [vector[i*4+3] for i in self.body_face_filtered_positions if i*4+3 < len(vector)]
                    visible_x_coords = [x for x, v in zip(body_face_x_coords, body_face_visibility) if v > 0.5]

                    if visible_x_coords:
                        center_x = float(np.mean(visible_x_coords))
                        if prev_center_x is None:
                            x_diff = 0.0
                            movement = "STATIONARY"
                            left_right_up_down = [[0,0]]
                        else:
                            x_diff = center_x - prev_center_x
                            
                            # Check if at screen edges (90% threshold)
                            if center_x >= 0.9:
                                # At right edge, force movement right
                                movement = "MOVING RIGHT"
                                left_right_up_down = [[1,0]]
                            elif center_x <= 0.1:
                                # At left edge, force movement left
                                movement = "MOVING LEFT"
                                left_right_up_down = [[-1,0]]
                            elif x_diff > self.movement_threshold:
                                movement = "MOVING RIGHT"
                                left_right_up_down = [[1,0]]
                            elif x_diff < -self.movement_threshold:
                                movement = "MOVING LEFT"
                                left_right_up_down = [[-1,0]]
                            else:
                                movement = "STATIONARY"
                                left_right_up_down = [[0,0]]

                        movement_info["direction"] = movement
                        movement_info["x_diff"] = x_diff
                        movement_info["center_x"] = center_x
                        prev_center_x = center_x
                    else:
                        prev_center_x = None
                        movement_info["direction"] = "NO POSE"
                        movement_info["x_diff"] = 0.0
                        movement_info["center_x"] = None

                # --- Live prediction ---
                pred_label, confidence = None, None
                if self.live_predictor is not None:
                    pred_label, confidence = self.live_predictor.predict(vector)
                    if pred_label is not None:
                        cv.putText(frame, f"{pred_label} ({confidence:.2f})", (30,50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                
                
                # --- Prediction into Game inputs
                if pred_label == "punch":
                    attacks[3] = 1
                elif pred_label == "kick":
                    attacks[0] = 1
                    
                game_input = left_right_up_down + attacks

                # --- build snapshot ---
                snapshot = {
                    "vector": vector,
                    "frame_rgb": img_rgb,
                    "landmarks": landmarks_for_drawing,
                    "movement_info": movement_info,
                    "game_input": game_input,
                    "timestamp": timestamp,
                    "prediction": pred_label,
                    "confidence": confidence
                }

                # thread-safe store
                with self._lock:
                    self._latest = snapshot

                # publish to subscribers
                for q in list(self._subs):
                    try:
                        if q.full():
                            try: q.get_nowait()
                            except queue.Empty: pass
                        q.put_nowait(snapshot)
                    except Exception:
                        pass

                time.sleep(0.001)

        cap.release()
