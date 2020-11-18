import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt


def sort_function(e):
    return e['area']

class ContourExtractor:



    def return_contours(self, img, blur_strength = (5, 5)):
        #blur and apply threshold 
        blurred_img = cv2.GaussianBlur(img, blur_strength, 0)
        threshold = cv2.threshold(blurred_img, 60, 255, cv2.THRESH_BINARY)[1]
        #retrieve contours 
        cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        objects = []
        #construct into simple dict list
        for c in cnts:
            M = cv2.moments(c)
            A = cv2.contourArea(c)
            #get bounds
            bounds = cv2.boundingRect(cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True), True))
            X,Y,W,H = bounds[0], bounds[1], bounds[2], bounds[3]
            objects.append({
                'contour':c,
                'moments':M,
                'area':A,
                'w':W,
                'h':H,
                'x':X,
                'y':Y
            })
        return objects
    
    def return_top_n(self, object_list, n = 5):
        #sort
        object_list.sort(reverse = True, key = sort_function)
        return object_list[:5]

    def scale_contours(self, contours, scale_ratio):
        for c in contours:
            #resize to entire image 
            con = c['contour'].astype('float')
            con = con * scale_ratio
            con = con.astype('int')
            c['contour'] = con
            #scale bounding rect
            for k in ['x', 'y', 'w', 'h']:
                c[k] = int(c[k] * scale_ratio)
        return contours
    
    def return_img_data(self, img, contours):
        img_parts = []
        for c in contours:
            mask, masked = np.zeros([img.shape[0], img.shape[1]], np.uint8), np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask, [c['contour']], 0, 1, -1)
            #multi
            for i in range(3):
                masked[:,:,i] = np.multiply(img[:,:,i], mask)

            x_l, x_h = c['x'], min(c['x'] + c['w'], img.shape[1]-1)
            y_l, y_h = c['y'], min(c['y'] + c['h'], img.shape[0]-1)
            masked = masked[y_l : y_h,x_l : x_h]

            img_parts.append({
                'data': c,
                'label': 'undefined',
                'img': masked
            })
        return img_parts