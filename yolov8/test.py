# This is a sample Python script.
import ultralytics


# Press Ctrl+Shift+R to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from ultralytics import YOLO

from ultralytics.yolo.engine.results import Boxes
from skimage.metrics import structural_similarity as ssim
import cv2.cv2
import time
import torch
import pathlib
import os
import shutil
import pathlib
from dataset import *

'''
COCO128 dataset format:
2 folders: images, labels
filename should match
each label file is txt, has 
[class id,x,y,w,h]





'''
# print(cv2.__version__)
config_name = "yolov8s.pt"
print(f"Loading model {config_name}")
start = time.time()
model = YOLO(config_name)
end = time.time()
print(f"Done loading model in {end - start}seconds!")
model.info()

data_path = "frames2"
out_path = "test_out"
dir_to_dataset(model, data_path, out_path)

# img_path = "frames/19.jpg"
# img = cv2.imread(img_path)
# pred = model.predict(source=img,save=True)
# boxes = get_boxes(pred)
# imgd = draw_boxes(img,boxes)
# cv2.imshow("Image",imgd)
# cv2.waitKey(-1)
# txt = box_to_txt(img,boxes)
# boxes2 = txt_to_box(img,txt)
# imgd2 = draw_boxes(img,boxes2)
# cv2.imshow("Image",imgd2)
# cv2.waitKey(-1)

# img = cv2.imread("coco128/images/train2017/000000000036.jpg")
# txt = open("coco128/labels/train2017/000000000036.txt",'r').read()
# boxes = txt_to_box(img,txt)
# img = draw_boxes_from_dataset(img,boxes)
# cv2.imshow("Result",img)
# cv2.waitKey(-1)

# def extract_refs_onestage(model:YOLO, video,output_dir, threshold=0.7):
#     # kinda slow, maybe should not use this method.
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#
#     cap = cv2.VideoCapture(video)
#     out = pathlib.Path(output_dir)
#     ret, frame = cap.read()
#
#     pred = model.predict(source=frame)
#     for res in pred:
#         boxes = res.boxes
#         for box in boxes:
#             bbox = box.xyxy[0]
#             conf = box.conf
#             pt1 = (int(bbox[0]),int(bbox[1]))
#             pt2 = (int(bbox[2]), int(bbox[3]))
#             if conf>0.6:
#                 cv2.rectangle(frame,pt1,pt2,(0,0,255),2)
#
#     cnt = 0
#     while ret:
#         print(f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         ret, next_frame = cap.read()
#         if not ret:
#             break
#         # # Convert the next frame to grayscale
#         next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
#         sim = ssim(gray, next_gray)
#         print(sim)
#         if sim < threshold:
#             pred = model.predict(source=frame)
#             for res in pred:
#                 boxes = res.boxes
#                 for box in boxes:
#                     print(box.xyxy)
#                     bbox = box.xyxy[0]
#
#                     conf = box.conf
#                     cls = box.cls
#                     pt1 = (int(bbox[0]), int(bbox[1]))
#                     pt2 = (int(bbox[2]), int(bbox[3]))
#                     if conf > 0.6 and cls==0:
#                         cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
#             imgf = out.joinpath(f"{cnt}.jpg")
#             cv2.imwrite(str(imgf.resolve()),frame )
#         # Update the current frame
#         frame = next_frame
#         cnt+=1
#         # Release the video capture object
#     cap.release()
# video_path = "sakuga.mp4"
# output_dir = "test"
# extract_refs_onestage(model,video_path,output_dir)
# print("Done!")
# # vid_path = "sakuga.mp4"
# # cap = cv2.VideoCapture(vid_path)
# # ret,frame = cap.read()
# # while ret:
# #     start = time.time()
# #     # pred = model.predict(source=frame)
# #     pred = model(frame)
# #     end = time.time()
# #     print(f"Inference took {end-start}s")
# #     boxes = pred[0].boxes
# #     for res in pred:
# #         boxes = res.boxes
# #         for box in boxes:
# #             print(box.xyxy)
# #             print(type(box.xyxy))
# #             conf = box.conf
# #             print(box.conf)
# #             box = box.xyxy[0]
# #             pt1 = (int(box[0]),int(box[1]))
# #             pt2 = (int(box[2]), int(box[3]))
# #             if conf>0.6:
# #                 cv2.rectangle(frame,pt1,pt2,(0,0,255),2)
# #     cv2.imshow("Frame", frame)
# #     cv2.waitKey(2)
# #     ret,frame = cap.read()
#
#


