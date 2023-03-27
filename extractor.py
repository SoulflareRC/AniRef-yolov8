import os
import time
from datetime import datetime
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CFG
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
from utils.lineart_extractor import *
from utils.upscaler import Upscaler
from utils.extract_frames import *
from utils.tagger import Tagger
from yolov8.dataset import *

class RefExtractor(object):
    def __init__(self,model_path = None,models_folder = Path("models")):
        if model_path:
            self.model_path = model_path
        self.models_folder = models_folder
        self.models = self.grab_models(self.models_folder)
        self.model_path =self.models_folder.joinpath("yolov8").joinpath(self.models[0]+".pt" if ".pt" not in self.models[0] else self.models[0])
        self.model:YOLO = None
        self.extractor = Extractor(video=None,output_dir="temp")

        self.output_dir:Path = Path("output")
        self.save_img_format = ".jpg"

        # for identifying characters:
        self.tagger = Tagger()

        # for upscaling
        self.upscaler = Upscaler()

        # for line art
        self.line_extractor = LineExtractor()
        self.line_extractor.manga_model_path = "models/lineart/manga.pth"
        self.line_extractor.sketch_model_path = "models/lineart/sketch.pth"

    def get_md5(self,img: np.ndarray):
        img = Image.fromarray(img)
        return hashlib.md5(img.tobytes()).hexdigest()
    def grab_models(self,models_folder:Path):
        # models_folder = Path("models")
        if not models_folder.exists():
            models_folder_link = "https://drive.google.com/drive/folders/19Cnkg0y7kYq2uyC05E1DdLX24EEZjGlK?usp=share_link"
            gdown.download_folder(url=models_folder_link)
        models_folder = models_folder.joinpath("yolov8")
        models = [ x.stem for x in list(models_folder.iterdir())]
        print(f"Grabbed {len(models)} models:")
        for model in models:
            print(model)
        return models
    def extract_chara(self,video_path,output_format="imgs",mode="crop",frame_diff_threshold = 0.2,padding=0.0,conf_threshold=0.0,iou_threshold=0.6,min_bbox_size=0)->Path:
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
                # target_folder = self.output_dir.joinpath("video_to_imgs").joinpath(mode).joinpath(video.stem) # don't use video.stem since cv2 fails to write when path has special char
                target_folder = self.output_dir.joinpath("video_to_imgs").joinpath(mode).joinpath(datetime.now().__str__().replace(":", ""))
                if not target_folder.exists():
                    os.makedirs(target_folder)
                self.extractor.video = video_path
                # keyframes:list[Frame] = self.extractor.extract_keyframes(frame_diff_threshold) # this will blow up the memory
                frame_fnames:list[Path] = self.extractor.extract_keyframes2(frame_diff_threshold)
                # for frame in tqdm(keyframes):
                for frame_fname in tqdm(frame_fnames):
                    # frame = frame.img
                    frame_fname:Path
                    print(frame_fname)
                    frame = cv2.imread(frame_fname.resolve().__str__())
                    frame_fname.unlink(missing_ok=True)

                    res:list[Results] = self.model.predict(frame,iou=iou_threshold,conf=conf_threshold)
                    boxes = get_boxes(res,min_bbox_size=min_bbox_size)
                    boxes = pad_boxes(frame,boxes,scale=padding)
                    if mode=="crop":
                        res_imgs = crop_boxes(frame,boxes)
                        for res_img in res_imgs:
                            target_path = target_folder.joinpath(f"{len(list(target_folder.iterdir()))}{self.save_img_format}").__str__()
                            # try:
                            # cv2.imshow("Test",res_img) # not res_img's issue
                            print(res_img.shape)
                            print(target_path)
                            cv2.imwrite(target_path,res_img)
                            # cv2.waitKey(-1)
                            # except:
                            #     print(res_img.shape)
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
                target_folder = self.output_dir.joinpath("video_to_video").joinpath(datetime.now().__str__().replace(":", ""))
                cfg = DEFAULT_CFG
                cfg.iou = iou_threshold
                cfg.conf = conf_threshold
                predictor: BasePredictor = DetectionPredictor(cfg=cfg)
                predictor.save_dir =target_folder
                self.extractor.video = video_path
                lowf_vid = self.extractor.adjust_framerate(10)
                audio_path = self.extractor.extract_audio(Path(lowf_vid))
                predictor(source=lowf_vid, model=self.model_path)
                torch.cuda.empty_cache()
                merged_video = self.extractor.merge_video_audio(target_folder.joinpath(video.name),audio_path)
                # return target_folder.joinpath(video.name)
                return merged_video
            else:
                print("Output format other than video or imgs is not implemented")
        else:
            print("Video ",video_path," doesn't exist")

    def lineart(self,img:np.ndarray,it_dilate=1, ksize_dilate=3,ksize_gausian=3)->np.ndarray:

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
        blurred_img = cv2.GaussianBlur(img_gray, (ksize_gausian, ksize_gausian), 0)  # remove noise from image
        kernel = np.ones((ksize_dilate, ksize_dilate), np.uint8)
        img_dilated = cv2.dilate(blurred_img, kernel, iterations=it_dilate)  # raising the iterations help with darkness
        img_diff = cv2.absdiff(img_dilated, img_gray)
        # img_diff = cv2.absdiff(blurred_img, img_gray)
        contour = 255-img_diff
        # contour = cv2.erode(contour,(1,1),iterations=1)
        # contour = np.clip(contour*1.5,0,255)
        # ret,thresh = cv2.threshold(contour,240,255,cv2.THRESH_BINARY)
        dark = np.where(contour<240)
        contour[dark] = contour[dark] * 0.7

        return contour
    # def sharpen(self,img:np.ndarray)->np.ndarray:
    #     kernel = np.array([[-1, -1, -1],
    #                        [-1, 8, -1],
    #                        [-1, -1, -1]])
    #     sharpened = cv2.filter2D(img, -1, kernel)
    #     return sharpened
    def pad_image(self,img:np.ndarray,padding=0):
        # take in cv2 image, pad to square
        h, w, c = img.shape
        if w > h:
            border_h = int((w - h) / 2)
            res = cv2.copyMakeBorder(src=img, left=padding, right=padding, top=border_h+padding, bottom=border_h+padding,
                                     borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        elif h>w:
            border_w = int((h - w) / 2)
            res = cv2.copyMakeBorder(src=img, left=border_w+padding, right=border_w+padding, top=padding, bottom=padding,
                                     borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        else:
            res = cv2.copyMakeBorder(src=img, left= padding, right=padding, top=padding,
                                     bottom=padding,
                                     borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        return res
    def make_grid(self,imgs:list[np.ndarray], rows=3,cols=3):
        group_size = rows*cols
        for i in range(group_size-len(imgs)):
            white = np.full_like(imgs[0],255)
            imgs.append(white)
        for i in range(rows):
            if i == 0 :
                res =np.hstack(imgs[i*cols:(i+1)*cols])
            else:
                row = np.hstack(imgs[i*cols:(i+1)*cols])
                res = np.vstack((res,row))

        # res = res.reshape((900,900,3))
        print(res.shape)
        # res = res.reshape((rows,cols))
        return res


    # def make_grids(self,imgs:list[np.ndarray],rows=3,cols=3):
    #     group_size = rows*cols
    #     for i in range(0,len(imgs),group_size):

    # def grab_chara(self):
    #     if self.chara_folder.exists():
    #         chara_dirs = list(self.chara_folder.iterdir())
    #         for dir in chara_dirs:
    #             config_file = dir.joinpath("config.json")
    #             with open(config_file) as f:
    #                 config=json.loads(f)
    #             self.charas[config['name']]=config
    #             print(config)
    #     else:
    #         os.makedirs(self.chara_folder)
    # def create_chara_folder(self,chara_name):
    #     chara_folder = self.chara_folder.joinpath(chara_name)
    #     if not chara_folder.exists():
    #         os.makedirs(chara_folder)
    #     chara_img_folder = chara_folder.joinpath("images")
    #     if not chara_img_folder.exists():
    #         os.makedirs(chara_img_folder)
    #     return chara_folder
    # def update_chara(self,chara_imgs:list[np.ndarray],chara_name,chara_tags):
    #     chara_folder = self.create_chara_folder(chara_name)
    #     chara_img_folder = chara_folder.joinpath("images")
    #     chara_config_path = chara_folder.joinpath("config.json")
    #     if chara_name in self.chara_dict.keys():
    #         with open(chara_config_path,'r') as f:
    #             chara_config = json.loads(f)
    #     else:
    #         chara_config = {
    #             "name":chara_name,
    #         }
    #     for img in chara_imgs:
    #         fname = self.get_md5(img) + ".jpg"
    #         cv2.imwrite(chara_img_folder.joinpath(fname), img)
    #     chara_config["img_paths"] = [x.relative_to(chara_folder) for x in chara_img_folder.iterdir()]
    #     chara_config["tags"] = chara_tags
    #     with open(chara_config_path, 'w') as f:
    #         json.dumps(chara_config, f)
    #     self.grab_chara()



if __name__ == "__main__":
    line_extractor = LineExtractor()
    # line_extractor.manga_model_path = "models/lineart/manga.pth"
    # line_extractor.sketch_model_path = "models/lineart/sketch.pth"
    # folder = Path(r"D:\pycharmWorkspace\MangaLineExtraction_PyTorch\test")
    # for f in list(folder.iterdir()):
    #     img = cv2.imread(f.resolve().__str__())
    #     # img = line_extractor.laplacian(img)
    #     img = line_extractor.gaussian(img)
    #     img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #     # cv2.imshow("Laplacian",img)
    #     # img = line_extractor.sketch_line(img)
    #     img = line_extractor.manga_line_batch(img)
    #     cv2.imshow(f.name,img)
    # cv2.waitKey(-1)

    # from utils.Anime2Sketch.test import *
    # input_folder = Path(r"D:\pycharmWorkspace\Anime2Sketch\test")
    # output_folder = Path(r"D:\pycharmWorkspace\Anime2Sketch\test_sketch")
    # model_path = r"D:\pycharmWorkspace\Anime2Sketch\weights\netG.pth"
    # sketch_from_folder (input_folder,output_folder,model_path=model_path)


    # cfg = DEFAULT_CFG
    # cfg.iou = 0.5
    # cfg.conf = 0.2
    #
    # print(cfg)
    # # exit(0)
    # predictor: BasePredictor = DetectionPredictor(cfg=cfg)
    # predictor.save_dir = Path("test_pred")
    #
    # video_path = r"D:\pycharmWorkspace\flaskProj\done\To love ru darkness opening HD.mp4"
    # extractor = Extractor(video_path,output_dir="temp")
    # extractor.video = video_path
    # lowf_vid = extractor.adjust_framerate(10)
    # audio_path = extractor.extract_audio(Path(lowf_vid))
    # # lowf_vid = r"D:\Git\OPENMI-TEAM-4-Project\temp\Lycoris Recoil 第3集—在线播放—樱花动漫[index.m3u8].mp4"
    # predictor ( source=lowf_vid, model=r"D:\Git\OPENMI-TEAM-4-Project\models\yolov8\AniRef40000-n-epoch40.pt")
    # torch.cuda.empty_cache()


    # r = RefExtractor()
    # folder =Path (r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\btr")
    # fs = list(folder.iterdir())
    # imgs = [ ]
    # for i in range(10):
    #     f = fs[i]
    #     img = cv2.imread(f.resolve().__str__())
    #     img = r.pad_image(img,padding=0)
    #     # cv2.imshow(f"{i}",img)
    #     img = cv2.resize(img,(300,300),cv2.INTER_CUBIC)
    #     imgs.append(img)
    # # cv2.waitKey(-1)
    # grid = r.make_grid(imgs,4,3)
    # cv2.imshow("grid",grid)
    # # grid = r.sharpen(grid)
    # # cv2.imshow("sharp",grid)
    #
    # grid = r.lineart(grid)
    #
    # cv2.imshow("line",grid)
    # cv2.waitKey(-1)
    # u = Upscaler()
    # img_path = r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\btr\12.jpg"
    # img = cv2.imread(img_path)
    # res_img = u.upscale_img(img,scale=2)

    # cv2.imshow("Original",img)
    # # cv2.imshow("Upscaled",res_img)
    # res_img = u.upscale_img(img,model_name=u.models[1])
    # res_img = u.sharpen(res_img, mode="USM",ksize=5)
    # res_img2 = u.sharpen(res_img, mode="USM", ksize=15)
    # cv2.imshow("Sharpened", res_img)
    # cv2.imshow("Sharpened2", res_img2)
    # cv2.waitKey(-1)
    # res =  u.upscale(in_path=Path(img_path),out_path="test.jpg",scale=4,model_name=u.models[2])
    # print(res.resolve())


    # models_folder_link = "https://drive.google.com/drive/folders/19Cnkg0y7kYq2uyC05E1DdLX24EEZjGlK?usp=share_link"
    # gdown.download_folder(url=models_folder_link)
    # img = cv2.imread(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\btrd4273e07571f0dd16c4d438760af1900c666d7f3\100.jpg")
    # img = cv2.imread(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\b1fe50c2dcec5699f6522564375b5e41\1.jpg")
    # h,w,c = img.shape
    # if w>h:
    #     border_h = int((w-h)/2)
    #     res = cv2.copyMakeBorder(src=img,left=0,right=0,top=border_h,bottom=border_h,borderType=cv2.BORDER_CONSTANT)
    # else:
    #     border_w = int((h - w) / 2)
    #     res = cv2.copyMakeBorder(src=img, left=border_w, right=border_w,top=0,bottom=0, borderType=cv2.BORDER_CONSTANT)
    # cv2.imshow("test",res)
    # cv2.waitKey(-1)


    # r = RefExtractor()
    # r.tagger.d.threshold = 0.4
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
    # test_folder = Path(r"D:\pycharmWorkspace\OPENMI-TEAM-4-Project\output\video_to_imgs\crop\videoplayback(1)")
    # r.tagger.mark_chara_from_folder(test_folder,["kuriyama"])
    # imgs = []
    # for f in list(test_folder.iterdir()):
    #     if not f.is_dir():
    #         imgs.append(Image.open(f))
    # get toi - pose/gesture/face
    # with open("tags_class.txt",'r') as f:
    #     toi = set(f.read().splitlines())
    # print("TOI:",toi)
    # for img in imgs:
    #     # img is PIL Image
    #     img:Image
    #     print(img.size)
    #     img = img.resize((x*2 for x in img.size),Image.BOX)
    #     r.tagger.d.threshold = 0.0
    #     d = r.tagger(img)
    #     d =  r.tagger.get_toi_from_dict(d,toi)
    #     t = r.tagger.dict_to_tuples(d)
    #     print(t)
    #     img.show("Test")
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




