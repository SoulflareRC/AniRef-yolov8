from utils.extract_frames import Extractor
from utils.inference_utils import Segmentor
from ultralytics.yolo.engine.results import *
from ultralytics.yolo.data.dataset import *
from yolov8.dataset import *
from ultralytics import YOLO
from detectron2.structures.instances import Instances
import pathlib
import numpy as np
import cv2


def condinst_dir_to_imgboxes(s:Segmentor, dir_path, output_dir):
    # def dir_to_dataset(model: YOLO, dir_path, output_dir):
        dir = pathlib.Path(dir_path)
        out = pathlib.Path(output_dir)
        if not out.exists():
            out.mkdir(exist_ok=True)
        images = []
        if dir.is_dir():
            fs = list(dir.iterdir())
            suffixes = ['.jpg', '.png', '.bmp']
            for f in fs:
                if f.suffix in suffixes:
                    img = cv2.imread(str(f.resolve()))
                    images.append(img)
            return condinst_list_to_imgboxes(s, images)
        else:
            print(f"{dir} is not a directory")

def condinst_list_to_imgboxes(s:Segmentor, imgs:list):
    print(f"Processing {len(imgs)} images")
    boxes_list = []
    images = []
    for idx,img in enumerate(imgs):
        instances: Instances = s(img)['instances']
        pred_dict = instances.get_fields()
        boxes = s.get_boxes(pred_dict)
        yolo_boxes = []
        for box in boxes:
            conf = box[0]
            pt1, pt2 = box[1:3]
            x1, y1 = pt1
            x2, y2 = pt2
            cls = 0
            nums = [x1, y1, x2, y2, conf, cls]
            nums = [float(x) for x in nums]
            box = Boxes(boxes=np.asarray(nums), orig_shape=img.shape)
            yolo_boxes.append(box)
        images.append(img)
        boxes_list.append(yolo_boxes)
    return images,boxes_list


