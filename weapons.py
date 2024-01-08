from ultralytics import YOLO

model = YOLO("Model.pt")
results = model.predict(source="0", show=True)
print(results)
