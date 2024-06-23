import ultralytics
from ultralytics import YOLO

ultralytics.checks()

model = YOLO("yolov8x.pt")

results = model.train(data="data.yaml", epochs=100, imgsz=640, device=[0])

validation_results = model.val(data="data.yaml", imgsz=640, batch=25, conf=0.25, iou=0.6, device="0")

model.export()
