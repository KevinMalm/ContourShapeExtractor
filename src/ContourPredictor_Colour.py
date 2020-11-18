import numpy as np
import cv2



class ContourPredictor_COLOUR:


    def classify_individual(self, data_lcl):
        img = data_lcl['img']
        r_m = np.mean(img[:,:,0])
        g_m = np.mean(img[:,:,1])
        b_m = np.mean(img[:,:,2])

        if(g_m > max(r_m, b_m)):
            return 'grass'
        return 'undefined'
    def return_contour_labels(self, data_list):

        for item in data_list:
            item['label'] = self.classify_individual(item)
        
        return data_list