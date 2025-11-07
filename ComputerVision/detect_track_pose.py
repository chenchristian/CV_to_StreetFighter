import cv2 as cv
import csv
import mediapipe as mp

mp_pose = mp.solutions.pose
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

remove_indices = [1, 3, 4, 6, 17, 18, 19, 20, 21, 22, 31, 32]

csv_file = open("pose_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
header = []
for i in range(33 - len(remove_indices)):
    header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
csv_writer.writerow(header)

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
        h, w = frame.shape[:2]
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            lms_filtered = [lm for i, lm in enumerate(lms) if i not in remove_indices]

            # Draw landmarks
            for lm in lms_filtered:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw connections between filtered landmarks
            connections = [c for c in mp_pose.POSE_CONNECTIONS
                           if c[0] not in remove_indices and c[1] not in remove_indices]
            for start_idx, end_idx in connections:
                start = lms[start_idx]
                end = lms[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Compute and draw bounding box
            xs = [int(lm.x * w) for lm in lms_filtered if lm.visibility > 0.5]
            ys = [int(lm.y * h) for lm in lms_filtered if lm.visibility > 0.5]
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

            # Record pose data to CSV
            row = []
            for lm in lms_filtered:
                row += [lm.x, lm.y, lm.z, lm.visibility]
            csv_writer.writerow(row)

        cv.imshow("Pose Recording", frame)
        if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
csv_file.close()
cv.destroyAllWindows()
