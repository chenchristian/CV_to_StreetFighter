import cv2 as cv
import numpy as np
import pickle
import os
from collections import deque
from .pose_worker import PoseWorker

class PoseViewer:
    def __init__(self, shared_state: PoseWorker):
        self.shared_state = shared_state
        self._window_created = False
        self.last_frame = None
        self.last_landmarks = []
        self.movement_edge_threshold = 0.15
        
        # --- PROBABILITY HISTORY GRAPH ---
        # Stores the last 90 frames of probabilities
        self.history_length = 90
        self.prob_history = deque(maxlen=self.history_length)
        
        # --- LOAD ENCODER ---
        self.encoder_path = "Models/LSTM_v1/label_encoder.pkl"
        if os.path.exists(self.encoder_path):
            with open(self.encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            self.labels = list(label_encoder.classes_)
        else:
            self.labels = ["Unknown"]
            
        self.num_classes = len(self.labels)
        self.label_colors = self._generate_colors(self.num_classes)

    def _generate_colors(self, n):
        colors = {}
        for i, label in enumerate(self.labels):
            hue = int(180 * i / n) 
            color_bgr = cv.cvtColor(np.uint8([[[hue, 200, 255]]]), cv.COLOR_HSV2BGR)[0][0]
            colors[label] = tuple(int(c) for c in color_bgr)
        return colors

    def _draw_text(self, img, text, pos, font_scale=0.4, thickness=1):
        cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
        cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

    def _ensure_window(self):
        if not self._window_created:
            cv.namedWindow("Pose Debug Window", cv.WINDOW_NORMAL)
            self._window_created = True

    def poll(self):
        self._ensure_window()

        snapshot = self.shared_state.get_latest_snapshot()
        
        # --- THE FIX: Keep OpenCV alive while waiting for data ---
        if not snapshot:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            self._draw_text(img, "WAITING FOR CAMERA OR MODEL...", (120, 240), font_scale=0.7, thickness=2)
            cv.imshow("Pose Debug Window", img)
            cv.waitKey(1)
            return
        # ---------------------------------------------------------

        # 1. Extract updated data from snapshot
        self.last_frame = snapshot["frame_rgb"].copy()
        self.last_landmarks = snapshot["landmarks"]
        movement_info = snapshot["movement_info"]
        pred_label = snapshot.get("prediction", "None")      
        probabilities = snapshot.get("probabilities", None) 

        img = cv.cvtColor(self.last_frame, cv.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Draw Skeleton (Standard Mediapipe connections)
        skeleton_connections = [
            (11, 13), (13, 15), (12, 14), (14, 16), (11, 12), 
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), 
            (27, 29), (24, 26), (26, 28), (28, 30)
        ]
        for start, end in skeleton_connections:
            if start < len(self.last_landmarks) and end < len(self.last_landmarks):
                p1, p2 = self.last_landmarks[start], self.last_landmarks[end]
                cv.line(img, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), (255, 0, 0), 2)

        # 2. UPDATE HISTORY & DRAW HISTOGRAM
        if probabilities is not None:
            # Add to history for the scrolling chart
            self.prob_history.append(probabilities)

            # Draw Vertical Bar Chart (Left Side)
            bar_x, bar_y = 10, 20
            bar_w, bar_h, spacing = 180, 18, 8
            
            # Desired order
            custom_order = [
                "idle",  
                "jab", 
                "cross", 
                "lead_hook", 
                "rear_hook", 
                "uppercut", 
                "jumping_cross",
                "rear_low_kick", 
                "side_kick", 
                "spinning_back_high_kick", 
                "crouching_low_sweep",
                "grab", 
                "hadouken", 
                "shoryuken"
            ]
            
            # --- THE FIX: Case-insensitive matching ---
            # Create a dictionary mapping the lowercase version to the EXACT trained string
            actual_labels_map = {lbl.lower().strip(): lbl for lbl in self.labels}
            
            display_labels = []
            for custom_lbl in custom_order:
                clean_custom = custom_lbl.lower().strip()
                if clean_custom in actual_labels_map:
                    # Append the original exact string that the model expects
                    display_labels.append(actual_labels_map[clean_custom])
            
            # Add any remaining labels to the bottom automatically (just in case)
            display_labels += [lbl for lbl in self.labels if lbl not in display_labels]
        

            # Now loop through the reordered list
            for i, label in enumerate(display_labels):
                # We need to find the ORIGINAL index to grab the correct probability
                orig_index = self.labels.index(label)
                prob = probabilities[orig_index]
                
                y_pos = bar_y + i * (bar_h + spacing)
                
                # Draw Background Bar
                cv.rectangle(img, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (40, 40, 40), -1)
                
                # Draw Filled Probability Bar
                fill_w = int(bar_w * prob)
                cv.rectangle(img, (bar_x, y_pos), (bar_x + fill_w, y_pos + bar_h), self.label_colors[label], -1)
                
                # Highlight winning class
                if label == pred_label:
                    cv.rectangle(img, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (255, 255, 255), 1)

                # Draw Text
                self._draw_text(img, f"{label.upper()}: {prob:.0%}", (bar_x + 5, y_pos + 14))

        # # 3. DRAW ROLLING TIME-SERIES CHART (Top Right)
        # if len(self.prob_history) > 1:
        #     chart_w, chart_h = 220, 120
        #     chart_x, chart_y = w - chart_w - 20, 20
            
        #     # Background box
        #     sub_img = img[chart_y:chart_y+chart_h, chart_x:chart_x+chart_w]
        #     black_rect = np.zeros_like(sub_img)
        #     res = cv.addWeighted(sub_img, 0.4, black_rect, 0.6, 0)
        #     img[chart_y:chart_y+chart_h, chart_x:chart_x+chart_w] = res
        #     cv.rectangle(img, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (200, 200, 200), 1)

        #     step_x = chart_w / (self.history_length - 1)
        #     for c_idx, label in enumerate(self.labels):
        #         points = []
        #         for p_idx, prob_vec in enumerate(self.prob_history):
        #             # Calculate horizontal position based on current history size
        #             x = int(chart_x + p_idx * step_x)
        #             # Y is inverted (0 is top, chart_h is bottom)
        #             y = int(chart_y + chart_h - (prob_vec[c_idx] * chart_h))
        #             points.append((x, y))
                
        #         # Draw the line for this specific class
        #         for i in range(len(points) - 1):
        #             # Only draw lines with significant probability to reduce visual noise
        #             if self.prob_history[i][c_idx] > 0.05 or self.prob_history[i+1][c_idx] > 0.05:
        #                 cv.line(img, points[i], points[i+1], self.label_colors[label], 1, cv.LINE_AA)

        # 4. DRAW ARROWS & EDGE ZONES
        direction = movement_info.get('direction', "STATIONARY")
        if "RIGHT" in direction:
            cv.arrowedLine(img, (w-150, h-50), (w-50, h-50), (255, 255, 255), 3)
        elif "LEFT" in direction:
            cv.arrowedLine(img, (150, h-50), (50, h-50), (255, 255, 255), 3)

        cv.imshow("Pose Debug Window", img)
        if cv.waitKey(1) & 0xFF == 27:
            self.close()

    def close(self):
        cv.destroyAllWindows()