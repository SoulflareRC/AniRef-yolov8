import subprocess
import sys
import os

import torch

file_dir = os.path.dirname(__file__)
print(file_dir)
sys.path.append(file_dir)
from utils.adet.config import get_cfg
import detectron2 as det2
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances

import shutil
import cv2
import time
from tqdm import tqdm,trange
import numpy as np
from deepdanbooru_onnx import deepdanbooru_onnx, DeepDanbooru, process_image
from PIL import Image

def pil_to_cv(img:Image):
    arr = np.asarray(img)
    arr = cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
    return arr
def cv_to_pil(img:np.ndarray):
    return Image.fromarray(img)
# wrapper class for utils
class Segmentor(object):
    def __init__(self,config_file="utils/configs/CondInst/CondInst-AnimeSeg.yaml",model_file="utils/models/CondInst-AnimeSeg.pth"):
        cfg = get_cfg()
        # load config from config file
        cfg.merge_from_file(config_file)
        # change config
        cfg.MODEL.WEIGHTS = model_file
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.d = DeepDanbooru()
        self.mask_threshold = 0.6
    def __call__(self,img):
        # takes in opencv image, return a dict
        return self.predictor(img)

    def getSegmented(self, img):
        '''
        Get the segmented img(masked only)
        '''
        start = time.time()
        outputs = self.predictor(img)
        default = img
        instances = outputs['instances']
        fields_dict = instances.get_fields()
        threshold = 0.5

        print(fields_dict.keys())
        if 'pred_masks' in fields_dict:  # if no pred_masks then figure not found
            masks = fields_dict['pred_masks']
            scores = fields_dict['scores']
            print(scores)
            print(f'Predicted {len(masks)} masks')
            result = masks[0]
            result = result.cpu().numpy()
            if scores[0] < threshold:
                return default
            for mask, score in zip(masks, scores):
                if score > threshold:
                    mask = mask.cpu().numpy()
                    result = cv2.bitwise_or(result, mask)
            result = cv2.dilate(result, kernel=(7, 7), iterations=1)
            result = result[..., np.newaxis]
            filtered = (img * result).astype(np.uint8)
            print(f'Inference took {time.time() - start}s')
            return filtered
        print(f'Inference took {time.time() - start}s')
        return default

    def benchmark_seg_dd(self,img:Image,d:DeepDanbooru):
        shape = img.size
        print(shape)
        img = img.resize([int(x / 2) for x in shape])

        img_cv = pil_to_cv(img)

        # cv2.waitKey(-1)
        # img.show("original")
        img_seg_cv = self.getSegmented(img_cv)
        img_seg = cv_to_pil(img_seg_cv)
        dict_original = d(process_image(img))
        dict_segmented = d(process_image(img_seg))
        # sort deepdanbooru interpretation result by confidence,this is for comparing segmented/original result
        res_original = sorted(list(zip(dict_original.keys(), dict_original.values())), key=lambda x: x[1], reverse=True)
        res_segmented = sorted(list(zip(dict_segmented.keys(), dict_segmented.values())), key=lambda x: x[1], reverse=True)

        print(res_original)
        print(res_segmented)

        # show original vs segmented image
        cv2.imshow('original', img_cv)
        cv2.imshow('segmented', img_seg_cv)
        cv2.waitKey(-1)
    def get_boxes(self,pred_dict:dict)->list:
        '''
        Takes inthe predicted dict(fields)
        output = s(img)['instances']
        fields = output.get_fields()
        :return:
        a list that contains tuples that has boxes in the form of (score,pt1(upper left),pt2(lower right))
        '''
        res = []
        if 'pred_boxes' in pred_dict.keys():
            boxes=pred_dict['pred_boxes']
            scores = pred_dict['scores']
            idx = 0
            for box in boxes:
                score = scores[idx]
                pt1 = (int(box[0]),int(box[1]))
                pt2 = (int(box[2]),int(box[3]))
                b = (score,pt1,pt2)
                print('Box ',idx,b)
                res.append(b)
                idx+=1
        return res
    def draw_boxes(self,img:np.ndarray,boxes:list)->np.ndarray:
        '''
        :param img: the img to draw
        :param boxes: a list from get_boxes
        :return: drawn img
        '''
        res_img = img.copy()
        for box in boxes:
            if box[0]>=self.mask_threshold:
                cv2.rectangle(res_img,box[1],box[2],(0,0,255),2 )
        return res_img
    def crop_boxes(self,img:np.array,boxes:list):
        '''

        :param img: img to produce cropped results
        :param boxes: boxes from get_boxes
        :return: list of cropped result
        '''
        res = []
        for box in boxes:
            if box[0]>=self.mask_threshold:
                pt1 = box[1]
                pt2 = box[2]
                # print("Pt:",pt1,pt2)
                cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]#be careful with this,index is reversed
                # print("Cropped:",cropped.shape)
                res.append(cropped)
        return res

    def get_masks(self, pred_dict: dict) -> list:
        '''
        Takes in the predicted dict(fields)
        output = s(img)['instances']
        fields = output.get_fields()
        :return:
        a list that contains tuples that has boxes in the form of (score,mask)
        '''
        res = []
        if 'pred_masks' in pred_dict.keys():
            masks = pred_dict['pred_masks']

            scores = pred_dict['scores']
            idx = 0
            for mask in masks:
                mask = mask.cpu().numpy()
                mask = mask[...,np.newaxis]#do this to avoid when np can't broadcast
                print(np.max(mask),np.min(mask),mask.dtype)
                score = scores[idx]
                b = (score, mask)
                res.append(b)
                idx += 1
        return res
    def mask_img_2(self,img:np.ndarray,masks:list):
        '''
        :param masks: a list of masks from get_masks
        :param img: image to be masked
        :return: a list of image with score
        '''
        res = []
        for mask in masks:
            score = mask[0]
            mask = mask[1]
            if score>=self.mask_threshold:
                masked = self.mask_img(img,mask)
                b = (score,masked)
                res.append(b)
        return res
    def mask_img(self,img:np.ndarray,mask:np.ndarray):
        '''
        :param masks: a mask
        :param img: image to be masked
        :return: masked image
        '''
        return (mask*img).astype(np.uint8)
    def merge_masks(self,masks:list):
        '''
        combine multiple masks into one mask by bitwise or
        :param masks: a list of mask
        :return: one mask
        '''
        if len(masks)==0:
            return None
        else:
            result = masks[0]#starting mask
            print(result.shape)
            for mask in masks:
                result = cv2.bitwise_or(result,mask)
            return result
