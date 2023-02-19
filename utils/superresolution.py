import os
import subprocess
import sys
import numpy as np
import tempfile

"""
USAGE:

Provide filename (.mp4 or equivalent filetype)
Provide outpath that stores converted images

"""

def esrgan(filename, outPath):
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()

    # STEP 1 FFMPEG

    png = temp_dir.name + "\out%d.png"
    cmd =f"""
    ffmpeg -i {filename} -vf "select='gt(scene, 0.18 )',hflip" -vsync vfr {png}
    """
    print(cmd)
    os.system(cmd)

    # STEP 2 ESRGAN

    loc = "./utils/esrgan/"

    subprocess.check_call([loc + r"realesrgan-ncnn-vulkan.exe", "-i",  temp_dir.name  ,
                            "-o", outPath  , "-n", "realesr-animevideov3", "-s", "2",
                            "-f", "png"])

    # Once done using temporary directory, clean it up
    temp_dir.cleanup() 
