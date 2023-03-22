"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""
import numpy as np

import torchvision.transforms as transforms
import os
import torch
from utils.Anime2Sketch.data import get_image_list,read_img_path, tensor_to_img, save_image
from utils.Anime2Sketch.model import create_model
from tqdm import tqdm
from pathlib import Path
import cv2
def sketch(img:np.ndarray):
    pass

def sketch_from_folder(input_folder:Path,output_folder:Path,model_path, load_size = 512):
    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_path).to(device)  # create a model given opt.model and other options
    model.eval()
    # get input data
    if input_folder.is_dir():
        test_list = get_image_list(input_folder.resolve().__str__())
    elif input_folder.is_file():
        test_list = [input_folder.resolve().__str__()]
    else:
        raise Exception(f"{input_folder} is not a valid directory or image file.")
    # save outputs
    save_dir = output_folder
    os.makedirs(save_dir, exist_ok=True)
    # print("Hey")
    for test_path in tqdm(test_list):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(save_dir, basename)
        img = cv2.imread(test_path)
        aus_resize = img.shape

        # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img = cv2.Laplacian(img, ddepth=-1, ksize=5, borderType=cv2.BORDER_DEFAULT)
        img = 255-img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
        blurred_img = cv2.GaussianBlur(img_gray, (7,7), 0)  # remove noise from image
        kernel = np.ones((3,3), np.uint8)
        img_dilated = cv2.dilate(blurred_img, kernel, iterations=2)  # raising the iterations help with darkness
        img_diff = cv2.absdiff(img_dilated, img_gray)
        # img_diff = cv2.absdiff(blurred_img, img_gray)
        img = 255-img_diff
        print(aus_resize)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.imshow("Gray",img)


        aus_max = np.max(aus_resize)
        load_size = (int(aus_max/256))*256


        # img = cv2.resize(img,(aus_resize[1]*4,aus_resize[0]*4),cv2.INTER_CUBIC)
        img = cv2.addWeighted(img, 1, img, 0, 0.0)
        img = cv2.normalize(img,img,0.,1.0,cv2.NORM_MINMAX,dtype=cv2.CV_32F)

        # img = (img/255).astype(np.float)
        # print(img)
        if load_size>0:
            img = cv2.resize(img,(load_size,load_size),cv2.INTER_CUBIC)
        img = torch.tensor(img,dtype=torch.float32)
        norm_alpha = 0.5
        norm_beta = 0.5
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Normalize((norm_alpha,norm_alpha, norm_alpha), (norm_beta, norm_beta, norm_beta))
        ])
        img = torch.permute(img,[2,0,1])
        print(img.shape)
        img = transform(img)
        img = img.unsqueeze(0)
        print(img.shape)

        # img, aus_resize = read_img_path(test_path, load_size)

        aus_tensor = model(img.to(device))
        aus_img = tensor_to_img(aus_tensor)
        print(type(aus_img))
        aus_img = cv2.resize(aus_img,(aus_resize[1],aus_resize[0]),cv2.INTER_CUBIC)
        # aus_img = aus_img*0.4
        cv2.imshow("img",aus_img)

        # return aus_img
        # save_image(aus_img, aus_path, aus_resize)
    cv2.waitKey(-1)