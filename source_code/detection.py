import pandas as pd
import os
import yaml
import shutil
import configparser
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

class YOLOv8:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, source_img):
        results = self.model.predict(source_img,classes = 0, verbose=False)[0]
        bboxes = results.boxes.xywh.cpu().numpy()
        bboxes[:, :2] = bboxes[:, :2] - (bboxes[:, 2:] / 2)
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        return bboxes, scores, class_ids
