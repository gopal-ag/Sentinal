import ultralytics
from ultralytics import YOLO
import cv2
import os
import numpy as np
import time

cap = cv2.VideoCapture(0)
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_time = time.time()

def b_function(x, y, w, h):
    func = x + (camera_width + 1) * (y + (camera_width + 1) * (w + (camera_width + 1) * h))
    return func

model_pth = r'/Users/gopalagarwal/PycharmProjects/SentinelAI/Vandalism.pt'
folder_pth = r'/Users/gopalagarwal/PycharmProjects/SentinelAI/Vandalism_detection'

def delete_all_files(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

delete_all_files(folder_pth)

detected_set = set()
extra_pixel_width = 10
pred_count = 1

model = YOLO(model_pth)

while True:
    _, frame = cap.read()
    results = model.predict(frame, show=True, classes=[0])  # Modified line
    current = time.time()

    if np.abs(start_time - current) > 5:
        for r in results:
            num_pred = len(r.boxes)
            stacked_tensor = r.boxes.xywhn

            for (x, y, w, h, conf) in stacked_tensor:  # Modified line
                if conf > 0.60:  # Added condition to check confidence
                    detection_id = b_function(x, y, w, h)

                    if detection_id not in detected_set:
                        detected_set.add(detection_id)
                        x = int(float(x) * camera_width)
                        y = int(float(y) * camera_height)
                        w = int(float(w) * camera_width)
                        h = int(float(h) * camera_height)
                        x1 = int(x - w/2)
                        y1 = int(y - h/2)

                        x1 = max(0, x1 - extra_pixel_width)
                        y1 = max(0, y1 - extra_pixel_width)
                        x2 = min(camera_width, x1 + w + 2 * extra_pixel_width)
                        y2 = min(camera_height, y1 + h + 2 * extra_pixel_width)

                        new_img = frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(folder_pth, f'result_{pred_count}.jpg'), new_img)
                        pred_count += 1
                        start_time = time.time()

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
