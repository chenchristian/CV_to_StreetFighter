import cv2 as cv
import mediapipe as mp
import numpy as np
import queue
import threading 

mp_pose = mp.solutions.pose

remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]
movement_threshold = 0.05


def record_pose_data(mode_test=False, csv_filename="pose_data.csv", movement_info=None):
    """
    Generator that yields: (vector, frame_surface_data, landmarks_list)
    """

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if movement_info is None:
        movement_info = {"direction": "", "x_diff": 0.0}

    prev_center_x = None

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
            img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = pose.process(img_rgb)

            vector = []
            landmarks_for_drawing = []

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                lms_filtered = [lm for i, lm in enumerate(lms) if i not in remove_indices]

                vector = np.array(
                    [coord for lm in lms_filtered for coord in (lm.x, lm.y, lm.z, lm.visibility)],
                    dtype=np.float32
                )

                landmarks_for_drawing = [(lm.x, lm.y) for lm in lms]

                center_x = np.mean([lm.x for lm in lms_filtered])
                if prev_center_x is not None:
                    x_diff = center_x - prev_center_x
                    movement_info["x_diff"] = x_diff
                    if x_diff > movement_threshold:
                        movement_info["direction"] = "MOVE RIGHT"
                    elif x_diff < -movement_threshold:
                        movement_info["direction"] = "MOVE LEFT"
                    else:
                        movement_info["direction"] = "STATIONARY"
                prev_center_x = center_x

            else:
                movement_info["direction"] = "NO POSE"
                movement_info["x_diff"] = 0.0
                vector = np.zeros((33 - len(remove_indices)) * 4, dtype=np.float32)
                landmarks_for_drawing = []

            frame_surface_data = img_rgb

            yield vector, frame_surface_data, landmarks_for_drawing

    cap.release()


def start_cv_worker(shared_q: "queue.Queue", mode_test=True, movement_info=None):
    """
    Starts background thread that pushes pose data into shared_q.
    """

    def _worker():
        try:
            for vector, rgb_frame, landmarks in record_pose_data(
                mode_test=mode_test,
                movement_info=movement_info
            ):
                try:
                    shared_q.put((vector, rgb_frame, landmarks), timeout=0.01)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"CV worker error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
