import os
import cv2
import numpy as np
import subprocess
import tempfile
import torch
import torchvision.transforms as transforms
from utils.Anime2Sketch.model import create_model, Anime2Sketch
from utils.ManagaLineExtraction.model import MangaLineExtractor
"""
Usage: 

Provide filename path that contains directory of images 
Provide outPath to store the converted images


"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LineExtractor(object):
    def __init__(self):

        '''
        this part for gaussian
        '''
        self. it_dilate = 2
        self. ksize_dilate = 3
        self. ksize_gaussian = 3
        # self.lineart_mode = "M" # can be "M"(MangaLineExtractor) or "A"(Anime2Sketch),it's pointless to use both.
        '''
        this part for laplacian 
        '''
        self. ksize_laplacian = 5
        '''
        this part for Anime2Sketch
        '''
        self.sketch_model_path = None
        self.sketch_model = None
        '''
        this part for MangaLineExtractor
        '''
        self.manga_model_path = None
        self.manga_model = None
    def gaussian(self,img:np.ndarray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
        blurred_img = cv2.GaussianBlur(img_gray, (self.ksize_gaussian, self.ksize_gaussian), 0)  # remove noise from image
        kernel = np.ones((self.ksize_dilate, self.ksize_dilate), np.uint8)
        img_dilated = cv2.dilate(blurred_img, kernel, iterations=self.it_dilate)  # raising the iterations help with darkness
        img_diff = cv2.absdiff(img_dilated, img_gray)
        img = 255-img_diff
        dark = np.where(img<240)
        img[dark] = img[dark] * 0.7
        return img
    def laplacian(self,img:np.ndarray):
        img = cv2.Laplacian(img,ddepth=-1,ksize=self.ksize_laplacian,borderType=cv2.BORDER_DEFAULT)
        img = 255-img
        return img
    def sketch_line(self,img:np.ndarray,load_size = 0):
        if self.sketch_model is None:
            self.sketch_model = create_model(self.sketch_model_path)
        self.sketch_model.to(device)
        self.sketch_model.eval()
        with torch.no_grad():
            img = cv2.normalize(img, img, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            original_shape = img.shape
            dim_max = np.max(original_shape)
            load_size = (int(dim_max / 256)) * 256
            if load_size==0:
                load_size = (max(int(dim_max / 256),1)) * 256
            img = cv2.resize(img,(load_size,load_size),cv2.INTER_CUBIC)
            img = torch.tensor(img,dtype=torch.float32)
            norm_alpha = 0.5
            norm_beta = 0.5
            transform=transforms.Compose([
                transforms.Grayscale(3),
                transforms.Normalize((norm_alpha,norm_alpha, norm_alpha), (norm_beta, norm_beta, norm_beta))
            ])
            img = torch.permute(img, [2, 0, 1])
            img = transform(img)
            img = img.unsqueeze(0)
            pred = self.sketch_model(img.to(device))
            pred:torch.Tensor
            pred = pred.cpu().numpy()[0,0,:,:]
            print(pred.shape)
            res_img = cv2.resize(pred, (original_shape[1], original_shape[0]), cv2.INTER_CUBIC)
            torch.cuda.empty_cache()
            return res_img
    def manga_line(self,img:np.ndarray,load_size = 0):
        if self.manga_model is None:
            self.manga_model = MangaLineExtractor()
            self.manga_model.load_state_dict(torch.load(self.manga_model_path))
        self.manga_model.to(device)
        self.manga_model.eval()
        with torch.no_grad():
            original_shape = img.shape
            dim_max = np.max(original_shape)
            if load_size==0:
                load_size = (int(dim_max / 256)) * 256
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (load_size, load_size), cv2.INTER_CUBIC)
            img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            tensor = torch.from_numpy(img)
            tensor = torch.permute(tensor, (0, 1))
            tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.type(torch.FloatTensor).to(device)
            print(tensor.shape)
            y = self.manga_model(tensor)

            yc = y.cpu().numpy()[0, 0, :, :]
            yc = cv2.resize(yc, (original_shape[1], original_shape[0]), cv2.INTER_CUBIC)
            yc = np.clip(yc, 0, 255)
            output = (yc / 255).astype(np.float)
            torch.cuda.empty_cache()
            return output

    def manga_line_batch(self, img: np.ndarray):
        '''
        this uses the split image into batch method from original implementation
        '''
        if self.manga_model is None:
            self.manga_model = MangaLineExtractor()
            self.manga_model.load_state_dict(torch.load(self.manga_model_path))
        self.manga_model.to(device)
        self.manga_model.eval()
        with torch.no_grad():
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # why construct a batch???
            rows = int(np.ceil(img.shape[0] / 16)) * 16
            cols = int(np.ceil(img.shape[1] / 16)) * 16
            #
            # # manually construct a batch. You can change it based on your usecases.
            patch = np.ones((1, 1, rows, cols), dtype="float32")
            patch[0, 0, 0:img.shape[0], 0:img.shape[1]] = img
            tensor = torch.from_numpy(patch).to(device)
            y = self.manga_model(tensor)
            yc = y.cpu().numpy()[0, 0, :, :]
            # yc = cv2.resize(yc, (original_shape[1], original_shape[0]), cv2.INTER_CUBIC)
            output = yc[0:img.shape[0], 0:img.shape[1]]
            yc = np.clip(yc, 0, 255)
            output = (yc / 255).astype(np.float)
            torch.cuda.empty_cache()
            return output

def linearArt(filename, outPath):
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()

    convert_images(filename, temp_dir.name)

    loc = "./utils/esrgan/"

    subprocess.check_call([loc + r"realesrgan-ncnn-vulkan.exe", "-i",  temp_dir.name,
                            "-o", outPath  , "-n", "realesr-animevideov3"])



def extract_lineart(img):
    kernel = np.ones((5,5), np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #  grayscale
    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0) # remove noise from image
    # canny does not work
    # changes to THRESH_BINARY_INV does not work

    # # closing on the dilation result to fill gaps
    # closing = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)

    # median blur to smooth the lines
    # median = cv2.medianBlur(closing, 5)

    img_dilated = cv2.dilate(blurred_img, kernel, iterations=5) #raising the iterations help with darkness
    img_diff = cv2.absdiff(img_dilated, img_gray) 
    contour = 255 - img_diff
    return contour


def convert_images(dir_from, dir_to):
    ctr = 0
    # goes through file and looks for compatible image types (png/jpg)
    for file_name in os.listdir(dir_from):
        if file_name.endswith('jpg') or file_name.endswith('.png'):
            print(file_name)
            img = cv2.imread(os.path.join(dir_from, file_name))
            # transform each image
            img_contour = extract_lineart(img)

            res = file_name.split('.',1)[0]
            out_name = res + "out" + str(ctr) + ".jpg"
            cv2.imwrite(os.path.join(dir_to, out_name), img_contour)
            img_name = dir_to + out_name
            ctr += 1



