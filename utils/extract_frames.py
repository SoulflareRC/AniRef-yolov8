import subprocess
import shutil
import os
import pathlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time
from skimage.metrics import structural_similarity as ssim
from deepdanbooru_onnx import DeepDanbooru,process_image
from tqdm import tqdm
class Extractor(object):
    def __init__(self,video,output_dir):
        '''
        :param video: video to keep track of
        :param output_dir:
        '''
        self._video = video
        # self.vid  = cv2.VideoCapture(video)
        self.output_dir = output_dir
        self.frames = []
    def collect_frames(self)->np.ndarray:
        '''

        :return: a list of cv2 images(BGR)
        '''
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
        start = time.time()
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir,exist_ok=True)
        cmd = f"""ffmpeg -y -i "{self.video}" -vf "select='gt(scene,{threshold})'" -vsync vfr -frame_pts true "{self.output_dir}/%d.jpg" """
        subprocess.run(cmd)
        end = time.time()
        print(f"Extracting keyframes with threshold {threshold} took {end-start} seconds!")
        return self.collect_frames()
    def extract_keyframes2(self,threshold)->list[Path]:
        '''
        this version is for avoiding opencv exceeding memory when too many frames are extracted, just return the path to resulting files
        '''
        start = time.time()
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir,exist_ok=True)
        cmd = f"""ffmpeg -y -i "{self.video}" -vf "select='gt(scene,{threshold})'" -vsync vfr -frame_pts true "{self.output_dir}/%d.jpg" """
        subprocess.run(cmd)
        end = time.time()
        print(f"Extracting keyframes with threshold {threshold} took {end-start} seconds!")
        p = Path(self.output_dir)
        frame_fnames = list(p.rglob("*.jpg"))
        return frame_fnames
    def extract_IPBFrames(self,type):
        '''
        type should be one of {I,P,B}
        '''

        os.makedirs(self.output_dir,exist_ok=True)
        cmd = f"""ffmpeg -y -i "{self.video}" -vf "select='eq(pict_type,{type})'" -vsync vfr -frame_pts true "{self.output_dir}/%d.jpg" """
        subprocess.run(cmd)
        return self.collect_frames()
    def adjust_framerate(self,frate=30):
        vid_path = Path(self.video)
        vid_name = vid_path.stem
        vid_ext = vid_path.suffix
        cmd = f"""ffmpeg -y -i "{self.video}"  -filter:v fps={frate}  "{self.output_dir}/{vid_name}{vid_ext}" """
        subprocess.run(cmd)
        # return f'{self.output_dir}/{vid_name}{vid_ext}'
        return Path(self.output_dir).joinpath(vid_path.name)
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
        cmd = f"""ffmpeg -y -i "{self.video}" -ss {timestamp1} -to {timestamp2} -filter:v fps={frate}  "{self.output_dir}/{vid_name}{frameCnt1}-{frameCnt2}{vid_ext}" """
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


        return frames
    # def extract_refs(self,frames):
    #     # frames = self.extract_scene(start_frameCnt)
    #     '''
    #     Segmentation+add bounding box
    #     '''
    #     # dir = "refs"
    #     # if not os.path.exists(dir):
    #     #     os.makedirs(dir)
    #     start = time.time()
    #     self.s.mask_threshold = 0.6
    #     print("Start!")
    #     cnt = 0
    #     for frame in frames:
    #         print(frame.img.shape)
    #         instances = self.s(frame.img)['instances']
    #         instances:detectron2.structures.Instances
    #         pred_dict = instances.get_fields()
    #         # print(pred_dict)
    #         boxes = self.s.get_boxes(pred_dict)
    #         cropped = self.s.crop_boxes(frame.img,boxes)
    #         for idx,c in enumerate(cropped):
    #             # cv2.imshow(f"{cnt}",c)
    #             cv2.imwrite( os.path.join(self.output_dir,f"{cnt} {boxes[idx][0]}.jpg"),c)
    #             cnt+=1
    #         # cv2.waitKey(-1)
    #     end = time.time()
    #     print(f"Extracting reference images took {end-start} seconds!")
    #     return frames
    def extract_frames_ssim(self,threshold = 0.4):
        '''
        extract keyframes by ssim using self.video, return a list of cv2 images
        '''
        cap = cv2.VideoCapture(self.video)
        ret, frame = cap.read()
        frames = []
        total_frame_cnt =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(total_frame_cnt)):
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ret, next_frame = cap.read()
            if not ret:
                break
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            similarity = ssim(gray,next_gray)
            if similarity<threshold:
                frames.append(frame)
            frame = next_frame
        cap.release()
        print("Done extracing frames by ssim")
        return frames

    # def extract_refs_onestage(self,video,threshold=0.7):
    #     #kinda slow, maybe should not use this method.
    #     if not os.path.exists("refs"):
    #         os.makedirs("refs",exist_ok=True)
    #
    #     cap = cv2.VideoCapture(video)
    #     video_path = self.extract_clips(frameCnt1=1, frameCnt2=cap.get(cv2.CAP_PROP_FRAME_COUNT),frate=10)
    #     # video_path = self.extract_clips(frameCnt1=1, frameCnt2=350, frate=10)
    #     cap.release()
    #     print("Start!")
    #     cap = cv2.VideoCapture(video_path)
    #     ret,frame = cap.read()
    #     instances: detectron2.structures.Instances = self.s(frame)['instances']
    #     pred_dict = instances.get_fields()
    #     boxes = self.s.get_boxes(pred_dict)
    #
    #     cnt = 0
    #     while ret:
    #         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #         ret, next_frame = cap.read()
    #         if not ret:
    #             break
    #         # # Convert the next frame to grayscale
    #         next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    #         #
    #         # # Compute the frame difference between the current frame and the next frame
    #         # print(gray.shape,next_gray.shape)
    #         # frame_diff = cv2.absdiff(gray, next_gray)
    #         #
    #         # # Threshold the frame difference to detect changes
    #         # frame_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)[1]
    #         #
    #         # # If the frame difference is above the threshold, run object detection on the next frame
    #         # if cv2.countNonZero(frame_diff) > 0:
    #         sim = ssim(gray,next_gray)
    #         print(sim)
    #         if sim<threshold:
    #             instances: detectron2.structures.Instances = self.s(frame)['instances']
    #             pred_dict = instances.get_fields()
    #             boxes = self.s.get_boxes(pred_dict)
    #             cropped = self.s.crop_boxes(frame,boxes)
    #             for idx,crop in enumerate(cropped):
    #
    #                 cv2.imwrite(os.path.join("refs",f"Frame{cap.get(cv2.CAP_PROP_POS_FRAMES)} {idx} {cnt}.jpg"),crop)
    #                 # cv2.imshow(f"Frame{cap.get(cv2.CAP_PROP_POS_FRAMES)} {cnt}",crop)
    #             #     cnt+=1
    #
    #             # cv2.waitKey(-1)
    #
    #         # If the object is detected in the next frame, add it to the list of object frames
    #         # if bbox is not None:
    #         #     object_frames.append(next_frame)
    #
    #         # Update the current frame
    #         frame = next_frame
    #
    #         # Release the video capture object
    #     cap.release()
    #
    #     # return object_frames
    def extract_audio(self,video_path:Path):
        target_audio_path = video_path.parent.joinpath(video_path.stem+".mp3")
        cmd=f"""ffmpeg -y -i  "{video_path.resolve()}" -vn -f mp3 "{target_audio_path}" """
        subprocess.run(cmd)
        return target_audio_path
    def merge_video_audio(self,video_path:Path,audio_path:Path):
        target_video_path = video_path.parent.joinpath(video_path.stem+"_merged"+video_path.suffix)
        print(f"Merging {video_path} and {audio_path} into {target_video_path}")
        cmd=f"""ffmpeg -y \
                        -i "{video_path.resolve()}" -i "{audio_path.resolve()}" \
                        -c:v copy \
                        -map 0:v -map 1:a \
                        -y "{target_video_path.resolve()}" """
        subprocess.run(cmd)

        return target_video_path.resolve()
    def remove_similar(self,imgs):
        #takes in a list of imgs
        pass
class Frame(object):
    def __init__(self,video,img,cnt):
        self.video = video
        self.img = img
        self.frameCnt = cnt


