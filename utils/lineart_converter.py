import os
import cv2
import numpy as np
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import subprocess

def createImage(img):
    kernel = np.ones((5,5), np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #  grayscale
    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0) # remove noise from image
    # canny does not work
    # changes to THRESH_BINARY_INV does not work

    # # closing on the dilation result to fill gaps
    # closing = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)

    # # median blur to smooth the lines
    # median = cv2.medianBlur(closing, 5)

    img_dilated = cv2.dilate(blurred_img, kernel, iterations=2) #raising the iterations help with darkness
    img_diff = cv2.absdiff(img_dilated, img_gray) 
    contour = 255 - img_diff
    return contour


def convert_images(dir_from, dir_to):
    ctr = 0
    for file_name in os.listdir(dir_from):
        if file_name.endswith('jpg') or file_name.endswith('.png'):
            print(file_name)
            img = cv2.imread(os.path.join(dir_from, file_name))
            img_contour = createImage(img)
            res = file_name.split('.',1)[0]
            out_name = res + "out" + str(ctr) + ".jpg"
            cv2.imwrite(os.path.join(dir_to, out_name), img_contour)
            img_name = dir_to + out_name
            ctr += 1

if __name__ == '__main__':
    Tk().withdraw()  # we don't want a full GUI, keeps the root window from appearing
    
    # PROMPT 1 -- provide directory that holds images to edge
    folder_src = filedialog.askdirectory()

    # PROMPT 2 -- provide destination directory to store images
    folder_dest = filedialog.askdirectory() # prompt location to store images
    convert_images(folder_src, folder_dest)

    # PROMPT 3 -- prompt asks where the realesrgan-ncnn-vulkan.exe directory is
    exec_loc = filedialog.askdirectory()
    
    index = folder_dest.split('/')
    var = int(len(index) - 1)
    newPath = ""
    for i in range(0,var):
        newPath = newPath + index[i] + "/"

    # THIS METHOD LOOKS FOR A DIRECTORY/FOLDER CALLED pencil_out stored in same folder as folder_dest
    # Ex: folder_dest = C:\Users\Justin\Downloads\ffmpegProj\realesrgan-Master\output
    # Ex: out_frames = C:\Users\Justin\Downloads\ffmpegProj\realesrgan-Master\pencil_out
    # change the below line to whatever file name
    out_frames = newPath + 'pencil_out'
    print(out_frames)

    exec_loc += "/"

    subprocess.check_call([exec_loc + r"realesrgan-ncnn-vulkan.exe", "-i",  folder_dest,
                            "-o", out_frames  , "-n", "realesr-animevideov3"])



