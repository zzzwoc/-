from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.tune(data='data/data.yaml', tune_dir='tune/jsai',epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)