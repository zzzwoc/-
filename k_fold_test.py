import datetime
import shutil
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold
weights_path = 'yolov8s.pt'
model = YOLO(weights_path, task='detect')
results = {}

# Define your additional arguments here
batch = 16
project = 'kfold_221'
epochs = 500

for k in range(5):
    dataset_yaml = f'../2023-11-21_5-Fold_Cross-val/split_{k+1}/split_{k+1}_dataset.yaml'
    model.train(
        data=dataset_yaml,
        epochs=epochs, 
        batch=batch,
        project=project,
        device=[3],
        patience=50,
        name=f'train_split{k}',
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
    )  # include any train arguments
    results[k] = model.metrics  # save output metrics for further analysis