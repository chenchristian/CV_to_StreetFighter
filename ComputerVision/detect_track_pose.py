import cv2 as cv
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0) # laptop web cam, if we using actual camera connected to laptop then set to 1
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True, #something with background
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]

        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lms = results.pose_landmarks.landmark

            xs = [int(lm.x * w) for lm in lms if lm.visibility is not None]
            ys = [int(lm.y * h) for lm in lms if lm.visibility is not None]
            # remove points outside of the frame
            xs = [x for x in xs if 0 <= x <= w]
            ys = [y for y in ys if 0 <= y <= h]

            #bounding box
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                pad = 25
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)

                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv.putText(frame, "Bounding Box", (x_min, y_min - 8),
                            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.imshow("Body Tracking", frame)
        # using ESC to quit
        if cv.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()
