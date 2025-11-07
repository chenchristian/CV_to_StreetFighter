import cv2 as cv
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)  # laptop webcam, set 1 if using external camera
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Indices to remove (unwanted face, finger, and foot landmarks)
remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            # Filter out unwanted landmarks
            lms = results.pose_landmarks.landmark
            lms_filtered = [lm for i, lm in enumerate(lms) if i not in remove_indices]

            # Draw filtered landmarks
            for lm in lms_filtered:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw connections between filtered landmarks
            connections = [c for c in mp_pose.POSE_CONNECTIONS if c[0] not in remove_indices and c[1] not in remove_indices]
            for start_idx, end_idx in connections:
                start = lms[start_idx]
                end = lms[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Compute bounding box from filtered landmarks
            xs = [int(lm.x * w) for lm in lms_filtered if lm.visibility is not None]
            ys = [int(lm.y * h) for lm in lms_filtered if lm.visibility is not None]

            xs = [x for x in xs if 0 <= x <= w]
            ys = [y for y in ys if 0 <= y <= h]

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

        # ESC to quit
        if cv.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()
