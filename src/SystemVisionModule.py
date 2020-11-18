import numpy as  np
import matplotlib.pyplot as plt
import cv2
import imutils
import Filters as F
from ContourExtractor import ContourExtractor


class VisionModule:

    _contour_predictor: None
    _contour_extractor: ContourExtractor
    _contour_filter: F.NULL

    def __init__(self):
        self._contour_extractor = None
        self._contour_extractor = ContourExtractor()


    
    def process_img(self, img, fn = F.NULL):
        processed_img, scale_ratio = self.pre_process_img(img.copy(), fn)

        contour_data = self._contour_extractor.return_contours(processed_img)

        contour_data = self._contour_extractor.return_top_n(contour_data)

        contour_data = self._contour_extractor.scale_contours(contour_data, scale_ratio)

        contour_img_data = self._contour_extractor.return_img_data(img, contour_data)
        return contour_img_data
    
    def pre_process_img(self, img, fn, _w = 300):
        resized = imutils.resize(img, _w)
        scale_ration = img.shape[0] / float(resized.shape[0])
        processed = fn(resized)
        return processed, scale_ration
    
    def plot_contours_on_img(self, img, contour_img_data):
        img_a = img.copy()
        for data in contour_img_data:
            cv2.drawContours(img_a, [data['data']['contour']], -1, (255, 255, 255), 3)
        plt.figure()
        plt.imshow(img_a)
        return
    
    def compare_boundingRect_contours(self, img, contour_img_data):
        img_b = img.copy()
        plt.figure()
        i = 1
        for data in contour_img_data:
            plt.subplot(5,2,i)
            x_l, x_h = data['data']['x'], min(data['data']['x'] + data['data']['w'], img.shape[1]-1)
            y_l, y_h = data['data']['y'], min(data['data']['y'] + data['data']['h'], img.shape[0]-1)
            blck = img_b[y_l : y_h,x_l : x_h]
            plt.imshow(blck)
            plt.subplot(5,2,i+1)
            plt.imshow(data['img'])
            i += 2
        return
