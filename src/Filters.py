import numpy as np
import imutils
import cv2
import math
import matplotlib.pyplot as plt


def magnitude(a, b):
    return math.sqrt((a ** 2) + (b ** 2))

Gx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
                ]
Gy = [
    [-1, -2, 1],
    [0, 0, 0],
    [1, 2, 1]
                ]

Gx = [
    [3, 0, -3],
    [+10, 0, -10],
    [+3, 0, -3]
                ]
Gy = [
    [+3, +10, +3],
    [0, 0, 0],
    [-3, -10, -3]
                ]


def NULL(img):
    return img
def FLATTEN(img):
    if(len(img.shape) == 3):
        return np.mean(img, axis = 2, dtype = 'uint8')
    return np.array(img, dtype = 'uint8')
def BW(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def SOBEL_PROCESS(img, T = 120):
    gradient = np.zeros(img.shape)
    #[y,x]
    
    mag = np.zeros(img.shape)
    for y in range(img.shape[0] - 2):
        for x in range(img.shape[1] - 2):
            S1 = sum(sum(Gx * img[y:y+3, x:x+3]))
            S2 = sum(sum(Gy * img[y:y+3, x:x+3]))
            mag[y, x] = magnitude(S1, S2)
    
    mag[mag < T ] = 0
    mag[mag >= T ] = 255
    
    return mag

def SOBEL(img):
    for i in range(3):
        img[:,:,i] = SOBEL_PROCESS(img[:,:,i])
    return np.array(img, dtype = 'uint8')

def SOBEL_FLATTEN(img):
    for i in range(3):
        img[:,:,i] = SOBEL_PROCESS(img[:,:,i])

    return np.mean(img, axis = 2, dtype = 'uint8')
def BW_SOBEL(img):
    return np.array(SOBEL_PROCESS(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), dtype = 'uint8')




#https://haru-atari.com/blog/30-write-simple-image-filters-in-python
def CONTRAST_RBG(img, contrast = 100, r = 0.299, b = 0.587, g = 0.144):
    r_average = np.mean(img[:, :, 0]) * r
    g_average = np.mean(img[:, :, 1]) * b
    b_average = np.mean(img[:, :, 1]) * g
    avg = r_average + g_average + b_average

    palette = np.array([int(avg + contrast * (i - avg)) for i in range(256)])
    palette = np.clip(palette, 0, 255)

    img[:,:,0] = palette[img[:,:,0]]
    img[:,:,1] = palette[img[:,:,1]]
    img[:,:,2] = palette[img[:,:,2]]

    return img

def CONTRAST_BW(img, contrast = 100):
    avg = np.mean(img)

    palette = np.array([int(avg + contrast * (i - avg)) for i in range(256)])
    palette = np.clip(palette, 0, 255)

    img[:,:,0] = palette[img[:,:,0]]
    img[:,:,1] = palette[img[:,:,1]]
    img[:,:,2] = palette[img[:,:,2]]

    return img

def GREY_SCALE(img, r = 0.2126, g = 0.7152, b = 0.0722):
    img[:,:,0] = img[:,:,0] * r
    img[:,:,1] = img[:,:,1] * g
    img[:,:,2] = img[:,:,2] * b

    img = np.sum(img, axis = 2)

    return img



def BRIGHTNESS_RBG(img, f = 1, r = 1, g = 0.2, b = 1):
    img[:,:,0] = np.clip(img[:,:,0] * r * f, 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * g * f, 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * b * f, 0, 255)
    return img

def BRIGHTNESS_BW(img, f = 1):
    img[:,:] = np.clip(img[:,:] * f, 0, 255)
    return img

def FORCE_3D(img):
    if(len(img.shape) == 3):
        return img
    dimmed = np.zeros([img.shape[0], img.shape[1], 3])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            dimmed[x,y] = [img[x,y], img[x,y], img[x,y]]
    return dimmed

def lvl_linear_fn(r):
    r[0] = int(r[0] * 255)
    r[1] = int(r[1] * 255)

    d = r[1] - r[0]
    rgb_scale = [i for i in range(256)]
    rgb_scale[:r[0]] = [0] * r[0]
    rgb_scale[r[1]:] = [255] * (256 - r[1])
    rgb_scale[r[0]:r[1]] = [int(i*(255/d)) for i in range(0, d)]
    return np.array(rgb_scale, dtype = 'uint8')

def LEVELS_RBG(img, r = [0,1], g = [0,1], b = [0,1]):

    r_scale = lvl_linear_fn(r)
    g_scale = lvl_linear_fn(g)
    b_scale = lvl_linear_fn(b)
    #clip 
    img[:,:,0] = r_scale[img[:,:,0]]
    img[:,:,1] = g_scale[img[:,:,1]]
    img[:,:,2] = b_scale[img[:,:,2]]

    return img



def Blue_deviation_filter(img, std = 20):
    avg_rg = np.mean(img[:,:,:1], axis = 2)
    subtracted = np.subtract(img[:, :, 2], avg_rg)
    #now lets find the min 
    min_value = np.min(subtracted) + std
    max_value = min_value + 2
    normalized = np.clip(subtracted, min_value, max_value)
    #for the final image lets set pixel value to min_value + std if Green channel is strongest 
    normalized[np.logical_and(img[:,:, 1] > np.mean(img, axis=2) + std/2, img[:,:, 1] > 135)] = max_value
    return normalized


def INVERT_AND_NORMALIZE(img):
    img -= np.min(img)
    img *= 255 / np.max(img)
    img = 255 - img
    return img
    
def COMBINED_FILTER_A(img):
    contrasted = CONTRAST_RBG(img, 100, g = 0.4)
    brightened = BRIGHTNESS_RBG(contrasted, f = 100, g = 0.1, b = 0.9, r = 0.9)
    return np.array(GREY_SCALE(brightened), dtype = 'uint8')

def COMBINED_FILTER_B(img):
    blue_dev = Blue_deviation_filter(img, std = 50)
    inverted = INVERT_AND_NORMALIZE(blue_dev)
    return np.array(inverted, dtype = 'uint8')



if __name__ == '__main__':
    imgs = [
        'test_files/no_filter_exp a.png',
        'test_files/no_filter_exp b.png',
        'test_files/no_filter_exp c.png',
        'test_files/no_filter_exp d.png'   
    ]

    c = len(imgs)
    r = 2
    plt.figure()
    i = 1
    for path in imgs:
        c_img = cv2.imread(path)
        plt.subplot(c,r,i)
        plt.imshow(c_img, cmap = 'Greys')

        plt.subplot(c,r,i+1)
        b_filtered = Blue_deviation_filter(c_img, std = 50)
        plt.imshow(b_filtered, cmap = 'Greys')
        i += r
    plt.show()