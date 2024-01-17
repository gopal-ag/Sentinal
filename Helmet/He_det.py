import cv2
from ultralytics import YOLO
import os

img_pth=r'/Users/gopalagarwal/PycharmProjects/SentinelAI/Helmet/img3.jpg'
result_pth=r'Results'
model_pth=r'/Users/gopalagarwal/PycharmProjects/SentinelAI/Helmet/Helmet.pt'

color1=(0,0,255)
color2=(255,0,0)
class_ids=[]

img=cv2.imread(img_pth)
height,width,channels= img.shape

model=YOLO(model_pth)

results=model.predict(source=img,show=False,conf=0.5)
for r in results:
        check=len(r.boxes)
        boxes=r.boxes.cpu().numpy()
        class_ids.append(boxes.cls)
        stacked_tensor=r.boxes.xywhn
        for i, (x,y,w,h) in enumerate(stacked_tensor):
            x=int(float(x)*width)
            y=int(float(y)*height)
            w=int(float(w)*width)
            h=int(float(h)*height)
            x1=int(x-w/2)
            y1=int(y-h/2)
            class_id=int(class_ids[0][i])
            if (class_id==0):
                cv2.rectangle(img,pt1=(x1,y1),pt2=(x1+w,y1+h),color=color1,thickness=2)
            else:
                 cv2.rectangle(img,pt1=(x1,y1),pt2=(x1+w,y1+h),color=color2,thickness=2)
            cv2.imwrite(os.path.join(result_pth,f'Result_3.jpg'),img)


