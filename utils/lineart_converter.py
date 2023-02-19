import os
import cv2
import numpy as np
import subprocess
import tempfile

"""
Usage: 

Provide filename path that contains directory of images 
Provide outPath to store the converted images


"""
def linearArt(filename, outPath):
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()

    convert_images(filename, temp_dir.name)

    loc = "./utils/esrgan/"

    subprocess.check_call([loc + r"realesrgan-ncnn-vulkan.exe", "-i",  temp_dir.name,
                            "-o", outPath  , "-n", "realesr-animevideov3"])



def createImage(img):
    kernel = np.ones((5,5), np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #  grayscale
    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0) # remove noise from image
    # canny does not work
    # changes to THRESH_BINARY_INV does not work

    # # closing on the dilation result to fill gaps
    # closing = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)

    # median blur to smooth the lines
    # median = cv2.medianBlur(closing, 5)

    img_dilated = cv2.dilate(blurred_img, kernel, iterations=2) #raising the iterations help with darkness
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
            img_contour = createImage(img)

            res = file_name.split('.',1)[0]
            out_name = res + "out" + str(ctr) + ".jpg"
            cv2.imwrite(os.path.join(dir_to, out_name), img_contour)
            img_name = dir_to + out_name
            ctr += 1



