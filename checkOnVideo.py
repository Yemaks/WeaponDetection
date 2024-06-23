from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

source = "gun.mp4"
model.predict(source=source, save=True, conf=0.5, classes=[0, 2], stream=True)
