import os
import time

from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.predictor import BasePredictor
import cv2
import gradio
import numpy as np
from PIL import Image
import gradio as gr
import deepdanbooru_onnx as dd
import os
import subprocess
from pathlib import Path
import os
import shutil
from tqdm import tqdm
import cv2
from PIL import Image
from deepdanbooru_onnx import DeepDanbooru,process_image
import subprocess
import hashlib
import json
import gdown
from utils.lineart_converter import *
from utils.superresolution import esrgan
from utils.extract_frames import *
from utils.tagger import Tagger
from yolov8.dataset import *

class RefExtractor(object):
    def __init__(self,model_path = None):
        if model_path:
            self.model_path = model_path
        self.models = self.grab_models()
        self.model_path = Path("models/yolov8").joinpath(self.models[0]+".pt" if ".pt" not in self.models[0] else self.models[0])
        self.model:YOLO = None
        self.extractor = Extractor(video=None,output_dir="temp")

        self.output_dir:Path = Path("output")
        self.save_img_format = ".jpg"

        # for identifying characters:

        self.tagger = Tagger()
        self.chara_folder = Path("characters")
        self.charas= {}
        self.grab_chara()
    def get_md5(self,img: np.ndarray):
        img = Image.fromarray(img)
        return hashlib.md5(img.tobytes()).hexdigest()
    def grab_models(self):
        models_folder = Path("models")
        if not models_folder.exists():
            models_folder_link = "https://drive.google.com/drive/folders/19Cnkg0y7kYq2uyC05E1DdLX24EEZjGlK?usp=share_link"
            gdown.download_folder(url=models_folder_link)
        models_folder = models_folder.joinpath("yolov8")
        models = [ x.stem for x in list(models_folder.iterdir())]
        print(f"Grabbed {len(models)} models:")
        for model in models:
            print(model)
        return models
    def extract_chara(self,video_path,output_format="imgs",mode="crop",frame_diff_threshold = 0.2,padding=0.0,conf_threshold=0.0)->Path:
        '''
        :param video_path: path to
        :param output_format:
        if output format is imgs,this function will extract keyframes and detect characters on all keyframes and save to output folder
        if output format is video, this function will produce a video with detection result
        :param mode:
        can be one of [crop,draw,highlight]
        :param mode:frame diff threshold is used for extracting keyframes
        :return:
        path to the folder that has result images if imgs
        otherwise return a path to the video
        '''
        video = Path(video_path)
        if video.exists():
            if output_format=="imgs":
                res_paths = []
                if self.model == None:
                    self.model = YOLO(self.model_path)
                target_folder = self.output_dir.joinpath("video_to_imgs").joinpath(mode).joinpath(video.stem)
                if not target_folder.exists():
                    os.makedirs(target_folder)
                self.extractor.video = video_path
                keyframes:list[Frame] = self.extractor.extract_keyframes(frame_diff_threshold)
                for frame in tqdm(keyframes):
                    frame = frame.img
                    res:list[Results] = self.model.predict(frame)
                    boxes = get_boxes(res,conf_threshold)
                    pad_boxes(frame,boxes,scale=padding)
                    if mode=="crop":
                        res_imgs = crop_boxes(frame,boxes)
                        for res_img in res_imgs:
                            target_path = target_folder.joinpath(f"{len(list(target_folder.iterdir()))}{self.save_img_format}").__str__()
                            cv2.imwrite(target_path,res_img)
                            # res_paths.append(target_path)
                    elif mode=="draw":
                        res_img = draw_boxes(frame,boxes)
                        target_path = target_folder.joinpath(f"{len(list(target_folder.iterdir()))}{self.save_img_format}").__str__()
                        cv2.imwrite(target_path,
                            res_img)
                        # res_paths.append(target_path)
                    elif mode=="highlight":
                        res_img = draw_boxes(frame,boxes)
                        res_img = highlight_box(res_img,boxes)
                        target_path = target_folder.joinpath(f"{len(list(target_folder.iterdir()))}{self.save_img_format}").__str__()
                        cv2.imwrite(target_path,
                            res_img)
                        # res_paths.append(target_path)
                torch.cuda.empty_cache()
                return target_folder
            elif output_format=="video":
                target_folder = self.output_dir.joinpath("video_to_video").joinpath(video.stem)
                predictor: BasePredictor = DetectionPredictor()
                predictor.save_dir =target_folder
                self.extractor.video = video_path
                lowf_vid = self.extractor.adjust_framerate(10)
                audio_path = self.extractor.extract_audio(Path(lowf_vid))
                predictor(source=lowf_vid, model=self.model_path)
                torch.cuda.empty_cache()
                merged_video = self.extractor.merge_video_audio(target_folder.joinpath(video.name),audio_path)
                # return target_folder.joinpath(video.name)
                return merged_video
    def lineart(self,img_path)->Path:
        '''
        :param img:convert an image to line art and save it to output,
        :return: path to the image
        '''
        start = time.time()
        img = cv2.imread(img_path)
        img_path = Path(img_path)
        target_folder = self.output_dir.joinpath("lineart")
        if not target_folder.exists():
            os.makedirs(target_folder)
        target_path = target_folder.joinpath(img_path.name)
        contour = extract_lineart(img)
        end = time.time()
        print("extracting line art used:",end-start," seconds")
        cv2.imwrite(target_path.resolve().__str__(),contour)
        # cv2.imshow("test",contour)
        # cv2.waitKey(-1)
        return target_path


    def grab_chara(self):
        if self.chara_folder.exists():
            chara_dirs = list(self.chara_folder.iterdir())
            for dir in chara_dirs:
                config_file = dir.joinpath("config.json")
                with open(config_file) as f:
                    config=json.loads(f)
                self.charas[config['name']]=config
                print(config)
        else:
            os.makedirs(self.chara_folder)
    def create_chara_folder(self,chara_name):
        chara_folder = self.chara_folder.joinpath(chara_name)
        if not chara_folder.exists():
            os.makedirs(chara_folder)
        chara_img_folder = chara_folder.joinpath("images")
        if not chara_img_folder.exists():
            os.makedirs(chara_img_folder)
        return chara_folder
    def update_chara(self,chara_imgs:list[np.ndarray],chara_name,chara_tags):
        chara_folder = self.create_chara_folder(chara_name)
        chara_img_folder = chara_folder.joinpath("images")
        chara_config_path = chara_folder.joinpath("config.json")
        if chara_name in self.chara_dict.keys():
            with open(chara_config_path,'r') as f:
                chara_config = json.loads(f)
        else:
            chara_config = {
                "name":chara_name,
            }
        for img in chara_imgs:
            fname = self.get_md5(img) + ".jpg"
            cv2.imwrite(chara_img_folder.joinpath(fname), img)
        chara_config["img_paths"] = [x.relative_to(chara_folder) for x in chara_img_folder.iterdir()]
        chara_config["tags"] = chara_tags
        with open(chara_config_path, 'w') as f:
            json.dumps(chara_config, f)
        self.grab_chara()



if __name__ == "__main__":
    # models_folder_link = "https://drive.google.com/drive/folders/19Cnkg0y7kYq2uyC05E1DdLX24EEZjGlK?usp=share_link"
    # gdown.download_folder(url=models_folder_link)

    r = RefExtractor()
    r.tagger.d.threshold = 0.4
    # vid_path = r"D:\pycharmWorkspace\flaskProj\videos2\videoplayback(1).mp4"
    # r.extract_chara(vid_path,output_format="imgs",padding=0.0)

    # ref_img = r"output/video_to_imgs/crop/videoplayback(1)/68.jpg"
    # ref_img = cv2.imread(ref_img)
    # tags = r.tagger. tag_chara(ref_img)
    # ref_img2 = cv2.imread(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\videoplayback(1)\3.jpg")
    # tags.union (r.tagger. tag_chara(ref_img2))
    # print(tags)
    # cv2.imshow("ref", ref_img)
    # cv2.waitKey(-1)
    # r.charas["kuriyama"] = {
    #     "tags":tags
    # }
    # r.tagger.chara_tags["kuriyama"]=tags
    test_folder = Path(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\videoplayback(1)")
    # r.tagger.mark_chara_from_folder(test_folder,["kuriyama"])
    imgs = []
    for f in list(test_folder.iterdir()):
        if not f.is_dir():
            imgs.append(Image.open(f))
    # get toi - pose/gesture/face
    with open("tags_class.txt",'r') as f:
        toi = set(f.read().splitlines())
    print("TOI:",toi)
    for img in imgs:
        # img is PIL Image
        img:Image
        print(img.size)
        img = img.resize((x*2 for x in img.size),Image.BOX)
        r.tagger.d.threshold = 0.0
        d = r.tagger(img)
        d =  r.tagger.get_toi_from_dict(d,toi)
        t = r.tagger.dict_to_tuples(d)
        print(t)
        img.show("Test")
        # exit(0)
    # r.tagger.mark_chara_from_imgs(imgs,['kuriyama'],test_folder)

    # img_paths = list(test_folder.iterdir())
    # for p in tqdm(img_paths):
    #     img_test = cv2.imread(p.resolve().__str__())
    #     tags_test = r.tag_chara(img_test)
    #     similarity = r.comp_tags(tags,tags_test)
    #     if similarity>0.4:
    #         print(similarity)
    #         print(tags)
    #         print(tags_test)
    #         shape = (int(img_test.shape[1]/2),int(img_test.shape[0]/2))
    #         img_test = cv2.resize(img_test,shape,cv2.INTER_CUBIC)
    #         cv2.imshow(p.name,img_test)
    #         cv2.waitKey(-1)

    # img = cv2.imread('test_imgs/56.jpg')
    # model = YOLO("models/yolov8/Anidet6000-s-epoch440.pt")
    # pred = model.predict(img)
    # boxes = get_boxes(pred)
    # img1 = draw_boxes(img,boxes)
    # cv2.imshow("img1",img1)
    # boxes_padded = pad_boxes(img,boxes,0.5)
    # img2 = draw_boxes(img,boxes_padded)
    # cv2.imshow("img2",img2)

    # cv2.waitKey(-1)

    # img = cv2.imread(r"D:\pycharmWorkspace\flaskProj\datasets\videoplayback(1).mp41\images\85.jpg")
    # r.lineart(img)
    # r.model_path = "Anidet6000-s-epoch440.pt"
    # out_vid = r.extract_chara(video_path=r"D:\pycharmWorkspace\flaskProj\videos2\videoplayback(1).mp4",output_format="video")
    #
    # print(out_vid)
    # toi = set()
    # with open("tags_desc.txt","r") as f:
    #     tags = f.readlines()
    #     for tag in tags:
    #         tag = tag.replace(" ","_").replace("\n","")
    #         toi.add(tag)
    # with open("tags_character.txt","r") as f:
    #     tags = f.readlines()
    #     for tag in tags:
    #         tag = tag.replace(" ","_").replace("\n","")
    #         toi.add(tag)
    # print(toi)
    #     # toi|=set(tags)
    # # with open("tags_character.txt","r") as f:
    # #     tags = f.readlines()
    # #     toi|=tags
    # print("Total tag of interest(toi) size:",len(toi))
    # img1 = Image.open(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\temp\11.jpg")
    # img2 = Image.open(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\temp\1.jpg")
    # tagger = Tagger()
    # tagger.d.threshold = 0.4
    # d1 = tagger(img1)
    # d2 = tagger(img2)
    # tags1 = set(d1.keys()).intersection(toi)
    # tags2 = set(d2.keys()).intersection(toi)
    # # .intersection(toi)
    # print(tags1)
    # print(tags2)
    # t1 = sorted(list(zip(tags1,[d1[k] for k in tags1] )), key=lambda x: x[1], reverse=True)
    # t2 = sorted(list(zip(tags2, [d2[k] for k in tags2])), key=lambda x: x[1], reverse=True)
    # print(t1)
    # print(t2)
    # if len(tags1)<len(tags2):
    #     short = tags1
    #     long  = tags2
    # else:
    #     short = tags2
    #     long  = tags1
    # intersection = tags1.intersection(tags2)
    # similarity =0 if len(short)==0 else len(intersection)/len(short)
    # print(len(tags1),len(tags2),len(intersection))
    # print("Similarity:",similarity)


    # t1 = tagger.dict_to_tuples(d1)
    # t2 = tagger.dict_to_tuples(d2)

    # model = YOLO("yolov8s.pt")
    # predictor:BasePredictor =  DetectionPredictor()
    # predictor.save_dir = Path("save_dir")
    # predictor(source="videos/sakugashort.mp4",model="yolov8s.pt")
    # model.predictor = predictor
    # res:list[Results] = model.predict(source="videos/sakugashort.mp4",save=True)
    # print(type(res))
    # print(len(res))
    # print(res[0].path)




