import pathlib
import time
import numpy as np
from utils.extract_frames import Extractor,Frame,Segmentor
from utils.inference_utils import Segmentor,pil_to_cv,cv_to_pil
from utils.tagger import Tagger
import cv2
from detectron2.structures.instances import Instances
from dataset import  *

e = Extractor(None,output_dir="temp")
e.video = "test.mp4"
'''
Use CondInst to make dataset
'''
s = Segmentor(model_file=r"D:\pycharmWorkspace\flaskProj\utils\models\CondInst-AnimeSeg.pth")
s.mask_threshold = 0.6 #set higher mask threshold to prevent false positive
# frames:list[Frame] =  e.extract_IPBFrames(type="I") # extract I frames
frames:list[Frame]=e.extract_keyframes(0.15) # extract frames by difference of 0.15
imgs = [f.img for f in frames]
imgs,boxes_list = condinst_list_to_imgboxes(s,imgs)
output_dir = "test"
make_dataset(imgs,boxes_list,output_dir)
# visualize result
for i in range(5):
    img = imgs[i]
    boxes = boxes_list[i]
    imgd = draw_boxes(img,boxes)
    cv2.imshow(f"{i}",imgd)
cv2.waitKey(-1)

'''
Use Yolov8 to make dataset
'''
# model = YOLO('yolov8s.pt')
# e = Extractor(None,output_dir="temp")
# e.video = "test.mp4"
# print("Extracting frames")
# # frames =  e.extract_frames_ssim() # slow, not recommended
# frames:list[Frame]=e.extract_keyframes(0.15) # extract frames by difference of 0.15
# imgs = [f.img for f in frames]
# print("Detecting")
# imgs,boxes_list = yolo_list_to_imgboxes(model,imgs)
# output_dir = "test"
# print("Making dataset")
# make_dataset(imgs,boxes_list,output_dir)
# # visualize result
# for i in range(5):
#     img = imgs[i]
#     boxes = boxes_list[i]
#     imgd = draw_boxes(img,boxes)
#     cv2.imshow(f"{i}",imgd)
# cv2.waitKey(-1)
