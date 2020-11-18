import matplotlib.pyplot as plt
import numpy as np
import cv2
from SystemVisionModule import VisionModule
from time import time 
import Filters as F

vM = VisionModule()




def process_video(path):
    video_capture = cv2.VideoCapture(path)
    success, img = video_capture.read()
    frame_counter = 0

    while success:
        success, img = video_capture.read()
        if(success and frame_counter < 240):
            t_a = time()
            data, sr = vM.process_img(img.copy(), fn = F.COMBINED_FILTER_A)
            frame_counter += 1
            print(time() - t_a)  

            #plot contours
            vM.plot_contours_on_img(img, data, sr)
            #vM.compare_boundingRect_contours(img, data)
            plt.show()

header = '/Users/kevinmalmsten/Documents/UST/Senior Design/OpenCVShapeExtractor/test_files/'


video_files = [
    'VID_20201105_105142.mp4',
    'VID_20201105_105204.mp4',
    'VID_20201110_125926.mp4'
    ]

for video in video_files:
    p = header + video
    process_video(p)