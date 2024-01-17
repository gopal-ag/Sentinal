import random
import ultralytics
from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import base64
import requests
import boto3
from io import BytesIO
from boto3 import session

camera_ip = '192.168.240.53'
# cap = cv2.VideoCapture(f'http://{camera_ip}:4747/video')
cap = cv2.VideoCapture(0)
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_time = time.time()

def b_function(x, y, w, h):
    func = x + (camera_width + 1) * (y + (camera_width + 1) * (w + (camera_width + 1) * h))
    return func

model_pth = r'/Users/gopalagarwal/PycharmProjects/SentinelAI/Weapon.pt'
folder_pth = r'/Users/gopalagarwal/PycharmProjects/SentinelAI/Weapon_detection'

def delete_all_files(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)



def upload_base_64_image_to_s3(base64_image, bucket_name, object_key, aws_access_key_id, aws_secret_access_key):
    image_data = base64.b64decode(base64_image)

    session1 = session.Session()
    s3 = session1.client('s3',
                      region_name='ap-south-1',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)

    try:
        s3.upload_fileobj(BytesIO(image_data), bucket_name, object_key, ExtraArgs={'ACL': 'public-read'})
        print('uploaded to s3 successfully!')
    except Exception as e:
        print(f"Error uploading to S3: {e}")

def save_base64_and_send_webhook(new_img, webhook_url):
    _, img_encoded = cv2.imencode('.jpg', new_img)
    base64_img = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    random_int = random.randint(10000,99999)

    upload_base_64_image_to_s3(base64_img, 'sentinelai', f'{str(random_int)}.jpg', 'AKIAT7YHQXWSQ2Q6FCN2', 'pSUQDXla+w+rSo7j9kU1j6Ilq4KsXiqDgdYHPy1d')
    picture_url = f"https://sentinelai.s3.ap-south-1.amazonaws.com/{str(random_int)}.jpg"
    response = requests.post(webhook_url, json={'picture': picture_url, 'model': 'weapon', 'cameraIp': camera_ip})

    if response.status_code == 200:
        print("Image sent to server successfully!")
    else:
        print("Error sending image to server:", response.text)

delete_all_files(folder_pth)

detected_set = set()
extra_pixel_width = 10
pred_count = 1

model = YOLO(model_pth)

while True:
    _, frame = cap.read()
    results = model.predict(frame, show=True, classes=[0], conf=.8)
    current = time.time()

    if np.abs(start_time - current) > 5:
        for r in results:
            num_pred = len(r.boxes)
            stacked_tensor = r.boxes.xywhn

            for (x, y, w, h) in stacked_tensor:
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
                        save_base64_and_send_webhook(new_img, 'http://192.168.240.123:8080/new')
                        cv2.imwrite(os.path.join(folder_pth, f'result_{pred_count}.jpg'), new_img)
                        pred_count += 1
                        start_time = time.time()

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
