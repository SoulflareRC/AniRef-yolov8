import os
import subprocess
import sys
import numpy as np
import argparse

if __name__ == "__main__":
    from tkinter import Tk  # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename
    from tkinter import filedialog


    Tk().withdraw()  # we don't want a full GUI, keeps the root window from appearing
    

    filename =  askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    
    # assert os.path.exists(filename), "Could not find the file: " + str(filename) # test for invalid os file


    folder_selected = filedialog.askdirectory() # dialog prompts location to store images

    index = folder_selected.split('/')
    var = int(len(index) - 1)
    test = ""
    for i in range(0,var):
        test = test + index[i] + "/"
    png = folder_selected + "/out%d.png"

    # example: mp4 = "/Users/justinsmith/Desktop/roadOfNaruto.mp4"

    cmd = f"""
    ffmpeg -i {filename} -vf "select='gt(scene, 0.18 )',hflip" -vsync vfr {png}
    """
    print(cmd)

    os.system(cmd)

    # STEP 2 ESRGAN

    # out_frames = r'C:/Users/Justin/Downloads/ffmpegProj/realesrgan-Master/out_frames'
    print(test)
    out_frames = test + 'out_frames'

    subprocess.check_call([test + r"realesrgan-ncnn-vulkan.exe", "-i",  folder_selected  ,
                            "-o", out_frames  , "-n", "realesr-animevideov3", "-s", "2",
                            "-f", "png"])











