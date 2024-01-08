import cv2
import glob
import time
from vehicle_detector import VD

vd = VD()
images_folder = glob.glob("/Users/gopalagarwal/Development /ML/SentinalAI/ITMS/images")


vehicles_folder_count = 0
roads = []
for img_path in images_folder:
    print("Img path", img_path)
    img = cv2.imread(img_path)

    vehicle_boxes = vd.detect_vehicles(img)
    vehicle_count = len(vehicle_boxes)
    roads.append(vehicle_count)

    print(vehicle_count)
    vehicles_folder_count += vehicle_count

    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)

        cv2.putText(
            img,
            "No of cars" + str(vehicle_count),
            (20, 50),
            1,
            1,
            (100, 200, 0),
            2,
        )

    cv2.imshow("Cars", img)
    cv2.waitKey(1)


r1 = roads[0]
r2 = roads[1]
r3 = roads[2]
r4 = roads[3]


threshold_low = 10
threshold_high = 20


if r1 < threshold_low:
    traffic_light_condition_r1 = "Green"
elif threshold_low <= r1 < threshold_high:
    traffic_light_condition_r1 = "Yellow"
else:
    traffic_light_condition_r1 = "Red"

if r2 < threshold_low:
    traffic_light_condition_r2 = "Green"
elif threshold_low <= r2 < threshold_high:
    traffic_light_condition_r2 = "Yellow"
else:
    traffic_light_condition_r2 = "Red"

if r3 < threshold_low:
    traffic_light_condition_r3 = "Green"
elif threshold_low <= r3 < threshold_high:
    traffic_light_condition_r3 = "Yellow"
else:
    traffic_light_condition_r3 = "Red"

if r4 < threshold_low:
    traffic_light_condition_r4 = "Green"
elif threshold_low <= r4 < threshold_high:
    traffic_light_condition_r4 = "Yellow"
else:
    traffic_light_condition_r4 = "Red"


max_traffic = max(r1, r2, r3, r4)


if max_traffic == r1:
    print(
        "Road 1 is the busiest --> Traffic Light Condition: ",
        traffic_light_condition_r1,
    )
elif max_traffic == r2:
    print(
        "Road 2 is the busiest --> Traffic Light Condition: ",
        traffic_light_condition_r2,
    )
elif max_traffic == r3:
    print(
        "Road 3 is the busiest --> Traffic Light Condition: ",
        traffic_light_condition_r3,
    )
elif max_traffic == r4:
    print(
        "Road 4 is the busiest --> Traffic Light Condition: ",
        traffic_light_condition_r4,
    )



def control_traffic_lights(road, traffic_light_condition):
    print(
        f"Controlling traffic lights for Road {road}. Condition: {traffic_light_condition}"
    )



control_traffic_lights(1, traffic_light_condition_r1)
control_traffic_lights(2, traffic_light_condition_r2)
control_traffic_lights(3, traffic_light_condition_r3)
control_traffic_lights(4, traffic_light_condition_r4)
