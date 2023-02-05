import subprocess

import detectron2 as det2
from adet.config import config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from adet.config import get_cfg
import os
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
def getSegmented(predictor,img):
    '''
    Get the segmented img
    '''
    start = time.time()
    outputs = predictor(img)
    default = img
    instances = outputs['instances']
    fields_dict = instances.get_fields()
    threshold = 0.5

    print(fields_dict.keys())
    if 'pred_masks' in fields_dict:#if no pred_masks then figure not found
        masks = fields_dict['pred_masks']
        scores = fields_dict['scores']
        print(scores)
        print(f'Predicted {len(masks)} masks')
        result = masks[0]
        result = result.cpu().numpy()
        if scores[0]<threshold:
            return default
        for mask,score in zip(masks,scores):
            if score>threshold:
                mask = mask.cpu().numpy()
                result = cv2.bitwise_or(result,mask)
        result = cv2.dilate(result,kernel = (7,7),iterations=1)
        result = result[..., np.newaxis]
        filtered = (img * result).astype(np.uint8)
        print(f'Inference took {time.time() - start}s')
        return filtered
    print(f'Inference took {time.time() - start}s')
    return default
def benchmark_seg_dd(predictor,img:Image):
    shape = img.size
    print(shape)
    img = img.resize([int(x / 2) for x in shape])

    img_cv = pil_to_cv(img)

    # cv2.waitKey(-1)
    # img.show("original")
    img_seg_cv = getSegmented(predictor, img_cv)
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
d = DeepDanbooru()

config_file ="configs/CondInst/CondInst-AnimeSeg.yaml"
model_file = "models/CondInst-AnimeSeg.pth"
cfg = get_cfg()
# load config from config file
cfg.merge_from_file(config_file)
# change config
cfg.MODEL.WEIGHTS = model_file
cfg.OUTPUT_DIR = 'output'
print(cfg.OUTPUT_DIR)
print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
# set threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.20
predictor = DefaultPredictor(cfg)#model for inference
img = Image.open("images/33.jpg")

video = "Lyco.mp4"
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

threshold = 0.25
mask_only = False
output_dir = "temp/extracted"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
# run and time the frame extraction
start = time.time()
cmd = f"""ffmpeg -i {video} -vf "select='gt(scene,{threshold})'" -vsync vfr -frame_pts true {output_dir}/%d.jpg"""
subprocess.run(cmd)
print(f"Extracting keyframes took {time.time()-start}s!")
cnt = 0
start_all = time.time()

tasks = os.listdir(output_dir)
pbar = tqdm(tasks)

for f in pbar:
    img_path = os.path.join(output_dir,f)
    img = cv2.imread(img_path)
    start = time.time()
    outputs = predictor(img)
    instances:Instances #just specifying the class
    instances = outputs['instances']
    fields_dict = instances.get_fields()
    threshold = 0.6
    # print(fields_dict.keys())
    if 'pred_masks' in fields_dict:
        cnt += 1
        masks = fields_dict['pred_masks']
        scores = fields_dict['scores']
        if mask_only:
            # print(scores)
            # print(f'Predicted {len(masks)} masks')
            idx = 0
            result = masks[0]
            result = result.cpu().numpy()
            for mask,score in zip(masks,scores):
                if score>threshold:
                    mask = mask.cpu().numpy()
                    result = cv2.bitwise_or(result,mask)
                    result = cv2.dilate(result,kernel = (7,7),iterations=1)
                    result = result[..., np.newaxis]
                    filtered = (img * result).astype(np.uint8)

                    output_img_dir = os.path.join(output_dir, 'output')
                    if not os.path.exists(output_img_dir):
                        os.makedirs(output_img_dir)
                    # print(os.path.splitext(f))
                    fname,ext = os.path.splitext(f)

                    cv2.imwrite(os.path.join(output_img_dir,fname+str(idx)+ext ), filtered)
                    idx += 1
        else:
            if scores[0]>=threshold:
                output_img_dir = os.path.join(output_dir, 'output')
                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)
                fname, ext = os.path.splitext(f)
                cv2.imwrite(os.path.join(output_img_dir, fname+ ext), img)

        # print(f'Inference took {time.time() - start}s')
    pbar.set_description('Processing '+f)

print(f"All inference took {time.time()-start_all}s!Found {cnt}/{len(os.listdir(output_dir))} figures. ")
