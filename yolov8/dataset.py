import numpy as np
import ultralytics.yolo.engine.results
from ultralytics import YOLO
from ultralytics.yolo.engine.results import *
from skimage.metrics import structural_similarity as ssim
import cv2.cv2
import time
import torch
import  pathlib
import os
import shutil
import pathlib

def get_boxes(pred:list[Results],conf_threshold=0.0)->list[Boxes]:
    ret = []
    for res in pred:
        boxes = res.boxes
        for box in boxes:
            if box.conf>conf_threshold:
                ret.append(box)
    return ret
def pad_boxes(img:np.ndarray,boxes:list[Boxes],scale=0.1)->list[Boxes]:
    if scale==0:
        return boxes
    res = []
    for box in boxes:
        conf = float(box.conf)
        cls = float(box.cls)
        box = box.xyxy[0]
        box = [int(x) for x in box]
        h,w,c = img.shape
        x1,y1,x2,y2 = box
        print(x1,y1,x2,y2)
        center_x = (x1+x2)/2
        center_y = (y1+y2)/2
        box_h_half = abs(y2-center_y)
        box_w_half = abs(x2-center_x)
        x3,y3,x4,y4 = x1-box_w_half*scale,y1-box_h_half*scale,x2+box_w_half*scale,y2+box_h_half*scale
        x3 = max(0,x3)
        y3 = max(0,y3)
        x4 = min(w,x4)
        y4 = min(h,y4)
        print(x3,y3,x4,y4)
        nums = np.asarray([x3,y3,x4,y4,conf,cls])
        box = Boxes(boxes=np.asarray(nums),orig_shape=img.shape)
        res.append(box)
    return res
def crop_boxes(img:np.ndarray,boxes:list[Boxes])->list[np.ndarray]:
    res = []
    for box in boxes:
        box = box.xyxy[0]
        box = [int(x) for x in box]
        x1,y1,x2,y2 = box
        cropped = img[y1:y2,x1:x2]
        res.append(cropped)
    return res
def draw_boxes(img:np.ndarray,boxes:list[Boxes],color = (0,0,255)):
    '''
    boxes: list of Boxes
    '''
    res = img.copy()
    for box in boxes:
        box = box.xyxy[0]
        pt1,pt2 = ( int(box[0]),int(box[1]) ), ( int(box[2]),int(box[3]) )
        cv2.rectangle(res,pt1,pt2,color,2)
    return res
def highlight_box(img:np.ndarray,boxes:list[Boxes]):
    dark =  cv2.convertScaleAbs(img,alpha=0.6)
    for box in boxes:
        box = box.xyxy[0]
        box = [int(x) for x in box]
        x1,y1,x2,y2 = box
        cropped = img[y1:y2,x1:x2]
        dark[y1:y2,x1:x2]=cropped
    return dark
def draw_boxes_from_dataset(img:np.ndarray,boxes:list[Boxes],color = (0,0,255)):
    res = img.copy()
    hh,ww,cc = img.shape
    print(ww,hh,cc)
    for box in boxes:
        bbox = box.xyxy[0]
        print("Box:", bbox)
        x1,y1,x2,y2 = bbox
        cls = box.cls
        conf = box.conf
        print(x1,y1,x2,y2)
        # cls,x,y,w,h,conf = box
        # center = (x,y)
        # x,y,w,h = x*ww,y*hh,w*ww,h*hh
        # pt1 = ( int(x-w/2),int(y-h/2) )
        # pt2 = ( int(x+w/2),int(y+h/2) )
        # print(x,y,w,h)
        pt1,pt2 = ( int(x1),int(y1) ), ( int(x2),int(y2) )
        cv2.rectangle(res,pt1,pt2,color,2)
    return res
def box_to_txt(img:np.ndarray,boxes:list[Boxes])->str:
    '''
    YOLODarknet label format:
    (x,y,w,h), where (x,y) is the coord of the center point.
    '''
    res_str = ""
    hh,ww,cc = img.shape
    for box in boxes:
        bbox = box.xywhn[0]
        conf = box.conf
        cls = box.cls
        x,y,w,h = bbox
        nums = [cls,x,y,w,h]
        nums = [str(float(x)) for x in nums]
        res_str+=" ".join(nums)+"\n"
    return res_str
def txt_to_box(img:np.ndarray,txt:str)->list:
    '''
    Takes in original image and text,
    Returns a list of Boxes
    '''
    hh,ww,cc = img.shape
    print("Text:\n",txt)
    res_boxes = []
    lines = txt.split('\n')
    for line in lines:
        line = line.strip().split(' ')
        if len(line)!=5:
            continue
        print("Line:", line)
        cls,x,y,w,h = [float(x) for x in line]
        # box is constructed with xyxy format, but txt is in xywhn format
        x1,y1,x2,y2 = x-w/2, y-h/2, x+w/2, y+h/2
        # constructed with xyxy,conf,cls
        nums = np.asarray([x1*ww,y1*hh,x2*ww,y2*hh,1.0,cls])
        box = Boxes(boxes=np.asarray(nums),orig_shape=img.shape)
        res_boxes.append(box)
    return res_boxes
def yolo_dir_to_imgboxes(model:YOLO,dir_path):
    dir = pathlib.Path(dir_path)
    if dir.is_dir():
        fs =list(dir.iterdir())
        suffixes = ['.jpg','.png','.bmp']
        imgs = []
        for f in fs:
            if f.suffix in suffixes:
                img = cv2.imread(str(f.resolve()))
                imgs.append(img)
        return yolo_list_to_imgboxes(model,imgs)
    else:
        print(f"{dir} is not a directory")
def yolo_list_to_imgboxes(model:YOLO,imgs:list[np.ndarray]):
    images = []
    boxes_list = []
    for idx,img in enumerate(imgs):
        pred = model.predict(source=img)
        boxes = get_boxes(pred)
        images.append(img)
        boxes_list.append(boxes)
    return images,boxes_list
def make_dataset(imgs:list[np.ndarray],boxes_list:list[list[Boxes]],output_dir):

    if len(imgs)!=len(boxes_list):
        print("Image number doesn't match with box number")
        return
    else:
        size = len(imgs)
        print(f"Dataset size: {size}")
        out = pathlib.Path(output_dir)
        if not out.exists():
            out.mkdir(exist_ok=True)
        out_img_dir = out.joinpath("images")
        out_txt_dir = out.joinpath("labels")
        if not out_img_dir.exists():
            out_img_dir.mkdir(exist_ok=True)
        if not out_txt_dir.exists():
            out_txt_dir.mkdir(exist_ok=True)
        for idx in range(size):
            img = imgs[idx]
            boxes = boxes_list[idx]
            txt = box_to_txt(img,boxes)
            target_txt = out_txt_dir.joinpath(str(idx) + ".txt")
            target_img = out_img_dir.joinpath(str(idx) + ".jpg")
            with open(target_txt.resolve(), 'w') as t:
                t.write(txt)
            cv2.imwrite(str(target_img.resolve()), img)
