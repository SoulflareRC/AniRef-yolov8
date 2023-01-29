import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils

if __name__ == "__main__":
    # from tkinter import Tk  # from tkinter import Tk for Python 3.x
    # from tkinter.filedialog import askopenfilename
    # #
    # Tk().withdraw()  # we don't want a full GUI, keeps the root window from appearing
    # filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    #
    # assert os.path.exists(filename), "Could not find the file: " + str(filename) # test for invalid os file
    #
    # # example: mp4 = "/Users/justinsmith/Desktop/roadOfNaruto.mp4"
    # png = "~/PycharmProjects/ffmpegProj/output/out%d.png" # output location is defaulted
    # # cmd = "ffmpeg -i {} -vf fps=1 {}".format(filename, png)
    # cmd = "ffmpeg -i {} -vf ".format(filename) + '"select=' + "'gt(scene, 0.28)'" + '"' + " -vsync vfr {}".format(png)
    # print(cmd)


    # superresolution

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=''

    from super_resolution import cartoon_upsampling_4x

    large_image = cartoon_upsampling_4x('./sasuke.png', './sasuke.png')

    cv2.imshow(large_image)


    # filters
    os.system(cmd)


    # def error(img1, img2):
    #     diff = cv2.subtract(img1, img2)
    #     err = np.sum(diff ** 2)
    #     mse = err / (float(h * w))
    #     msre = np.sqrt(mse)
    #     return mse, diff
    #
    # img1 = cv2.imread('/Users/justinsmith/PycharmProjects/ffmpegProj/output/out1.png')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # h, w = img1.shape
    #
    # img2 = cv2.imread('/Users/justinsmith/PycharmProjects/ffmpegProj/output/out255.png')
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    # (score, diff) = compare_ssim(img1, img2, full=True)
    # # converts diff to [0,255] for OpenCV
    # diff = (diff * 255).astype("uint8")
    # # score closest to 1 is a perfect match
    # print("SSIM: {}".format(score))
    #
    # # http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    # thresh = cv2.threshold(diff, 0, 255,
    #                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    #
    # # loop over the contours
    # for c in cnts:
    #     # compute the bounding box of the contour and then draw the
    #     # bounding box on both input images to represent where the two
    #     # images differ
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # # show output images
    # cv2.imshow("Original", img1)
    # cv2.imshow("Modified", img2)
    # cv2.imshow("Diff", diff)
    # cv2.imshow("Thresh", thresh)
    # cv2.waitKey(0)


    # match_error12, diff12 = error(img1, img2)
    # print("Image matching Error between image 1 and image 2:", match_error12)
    # plt.subplot(221), plt.imshow(diff12, 'gray'), plt.title("image1 - Image2"), plt.axis('off')
    # plt.show()





