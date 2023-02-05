import subprocess
import shutil
import os
import pathlib
from pathlib import Path
import cv2
from PIL import Image
import time
import moviepy
from deepdanbooru_onnx import DeepDanbooru,process_image
from moviepy.video.io.ffmpeg_tools import *
from .inference_utils import Segmentor
class Extractor(object):
    def __init__(self,video,output_dir):
        self._video = video
        # self.vid  = cv2.VideoCapture(video)
        self.output_dir = output_dir
        self.frames = []
    # @property
    # def video(self):
    #     return self._video
    # @video.setter
    # def video(self,val):
    #     self._video = val
    #     self.vid = cv2.VideoCapture(val)
    def collect_frames(self):
        p = Path(self.output_dir)
        frame_fnames = list(p.rglob("*.jpg"))
        frames = []
        for f in frame_fnames:
            print(f.resolve())
            # img = Image.open(str(f.resolve()))
            img = cv2.imread(str(f.resolve()))
            cnt = int(f.stem)
            print(cnt)
            frame = Frame(self.video,img,cnt)
            frames.append(frame)
            f.unlink()
        frames = sorted(frames,key=lambda x:x.frameCnt)
        return frames
    def extract_keyframes(self,threshold):
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir,exist_ok=True)
        cmd = f"""ffmpeg -i {self.video} -vf "select='gt(scene,{threshold})'" -vsync vfr -frame_pts true {self.output_dir}/%d.jpg"""
        subprocess.run(cmd)
        return self.collect_frames()
    def extract_IPBFrames(self,type):
        '''
        type should be one of {I,P,B}
        '''

        os.makedirs(self.output_dir,exist_ok=True)
        cmd = f"""ffmpeg -i {self.video} -vf "select='eq(pict_type,{type})'" -vsync vfr -frame_pts true {self.output_dir}/%d.jpg"""
        subprocess.run(cmd)
        return self.collect_frames()
    def extract_clips(self,frameCnt1,frameCnt2,frate=None):
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        # os.makedirs(self.output_dir)
        vid = cv2.VideoCapture(self.video)
        if frate is None:
            frate = vid.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate for extracting clips:",frate)
        frame_rate = vid.get(cv2.CAP_PROP_FPS)
        frame_cnt = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        vid.release()#this has to be closed

        print(frame_rate)
        timestamp1 = frameCnt1/frame_rate
        timestamp2 = frameCnt2/frame_rate
        vid_path = Path(self.video)
        vid_name = vid_path.stem
        vid_ext = vid_path.suffix
        #     -c:v copy -c:a copy
        cmd = f"""ffmpeg -i {self.video} -ss {timestamp1} -to {timestamp2} -filter:v fps={frate}  {self.output_dir}/{vid_name}{frameCnt1}-{frameCnt2}{vid_ext}"""
        subprocess.run(cmd)
        return f'{self.output_dir}/{vid_name}{frameCnt1}-{frameCnt2}{vid_ext}'
    def extract_scene(self,start_frameCnt):
        start_idx=-1
        end_frameCnt = -1
        for i in range(len(self.frames)):
            if self.frames[i].frameCnt==start_frameCnt:
                start_idx = i
                break
        vid = cv2.VideoCapture(self.video)
        if start_idx == len(self.frames)-1:
            end_frameCnt = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            end_frameCnt = self.frames[start_idx+1].frameCnt
        vid.release()
        print(f"Scene goes from frame {start_frameCnt} to {end_frameCnt}")
        shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        temp_clip = self.extract_clips(start_frameCnt,end_frameCnt,frate=5)
        temp = self.video
        self.video = temp_clip
        print("Done extracting scene frames!")
        frames = self.extract_keyframes(threshold=0)
        self.video = temp
        '''
        Segmentation+add bounding box
        '''



        return frames

class Frame(object):
    def __init__(self,video,img,cnt):
        self.video = video
        self.img = img
        self.frameCnt = cnt


