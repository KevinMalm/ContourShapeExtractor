import numpy as  np
import matplotlib.pyplot as plt
import cv2
import imutils
import Filters as F
from ContourExtractor import ContourExtractor
from ContourPredictor_Colour import ContourPredictor_COLOUR

class VisionModule:

    _contour_predictor: ContourPredictor_COLOUR
    _contour_extractor: ContourExtractor
    _contour_filter: F.NULL

    def __init__(self):
        self._contour_predictor = ContourPredictor_COLOUR()
        self._contour_extractor = ContourExtractor()


    
    def process_img(self, img, fn = F.NULL):
        #scale down to a more managable size and pass through colour filter
        processed_img, scale_ratio = self.pre_process_img(img.copy(), fn)
        #retrieve contours from image
        contour_data = self._contour_extractor.return_contours(processed_img)
        #filter to get only top 5
        contour_data = self._contour_extractor.return_top_n(contour_data)
        #scale back up to original image size
        contour_data = self._contour_extractor.scale_contours(contour_data, scale_ratio)
        #get masked images 
        contour_img_data = self._contour_extractor.return_img_data(img, contour_data)

        #label contours
        contour_img_data = self._contour_predictor.return_contour_labels(contour_img_data)
        return contour_img_data, scale_ratio
    
    def pre_process_img(self, img, fn, _w = 300):
        resized = imutils.resize(img, _w)
        scale_ration = img.shape[0] / float(resized.shape[0])
        processed = fn(resized)
        return processed, scale_ration
    
    def plot_contours_on_img(self, img, contour_img_data, scale_ratio):
        img_a = img.copy()
        for data in contour_img_data:
            c_x, c_y = int(data['data']['moments']['m10'] / data['data']['moments']['m00'] * scale_ratio) if (data['data']['moments']['m00'] > 0) else 0, int(data['data']['moments']['m01'] / data['data']['moments']['m00'] * scale_ratio) if (data['data']['moments']['m00'] > 0) else 0
            cv2.putText(img_a, data['label'], (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
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
