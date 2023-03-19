import os
import subprocess
import sys
import typing

import numpy as np
from pathlib import Path
import shutil
import numpy as np
import cv2
from PIL import Image
from typing import Union
import tempfile
"""
Using Real-ESRGAN from https://github.com/xinntao/Real-ESRGAN

models option:
General: [RealESRGAN_x4plus,RealESRGAN_x2plus,RealESRNet_x4plus,official ESRGAN_x4,realesr-general-x4v3]
Anime Image: [RealESRGAN_x4plus_anime_6B]
Anime Video: [realesr-animevideov3]
"""

cwd = Path.cwd()
class Upscaler(object):
    def __init__(self):
        # self.model = "realesr-animevideov3"
        self.models = [
            "realesr-animevideov3",
            "realesr-animevideov3-x2",
            "realesr-animevideov3-x3",
            "realesrgan-x4plus-anime"
        ]
        self.sharpen_modes = [
            'USM','Laplace'
        ]

    def upscale(self,in_path:Union[Path,str],out_path:Union[Path,str],scale=2,model_name="realesr-animevideov3",verbose=False):
        '''
        in path and out path can be Path or str
        return the out path (a Path object)
        '''
        cmd = [
            "utils/esrgan/realesrgan-ncnn-vulkan.exe",
            "-i",in_path if type(in_path)==str else in_path.resolve().__str__(),
            "-o",out_path if type(out_path)==str else out_path.resolve().__str__(),
            "-s",str(scale),
            "-n",model_name,
            "-j","4:4:4"
        ]
        if verbose:
            cmd.append('-v')
        # subprocess.run(cmd)
        subprocess.check_call(cmd)
        return Path(out_path)

    def upscale_img(self, img:np.ndarray, scale=2, model_name="realesr-animevideov3",
                verbose=False)->np.ndarray:
        '''
        takes in a cv2 image. writes img to tempfile so that it doesn't leave footprint

        '''
        temp = tempfile.mkdtemp()
        temp_path = Path(temp)
        f_in = temp_path.joinpath("in.png")
        f_out = temp_path.joinpath("out.png")
        cv2.imwrite(str(f_in),img)
        self.upscale(in_path=f_in,out_path=f_out,scale=scale,model_name=model_name,verbose=verbose)
        res_img = cv2.imread(str(f_out))
        shutil.rmtree(temp)
        return res_img
    def sharpen(self,img,mode="USM",ksize=1)->np.ndarray:
        if mode=="USM":
            blurred_img = cv2.GaussianBlur(img,ksize=(ksize,ksize),sigmaX=3)
            res = cv2.addWeighted(img,1.5,blurred_img,-0.5,0)
            return res
        elif mode=="Laplace":
            kernel = np.array([
                [0,-1,0],
                [-1,5,-1],
                [0,-1,0]
            ])
            res = cv2.filter2D(img,ddepth=-1,kernel=kernel)
            return res
        return img
