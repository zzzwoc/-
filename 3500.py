from ultralytics import YOLO# Load a model
model = YOLO('yolov8s.pt')  #加载预训练的模型(推荐用于训练)
model.train(data='data/07.yaml', epochs=1000, imgsz=640,device=2,
            name=f'8_2',batch=32,
            save_period=10,
            patience=100,
            project='3500',
            lr0=0.00346,
            lrf=0.00712,
            momentum=0.90435,
            weight_decay=0.00042,
            warmup_epochs=2.86637,
            warmup_momentum=0.50307,
            box=3.06891,
            cls=0.44912,
            dfl=0.74373,
            hsv_h=0.01664,
            hsv_s=0.84765,
            hsv_v=0.40727,
            degrees=0.0,
            translate=0.02217,
            scale=0.35655,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.80824,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
           )
print(model.metrics)

