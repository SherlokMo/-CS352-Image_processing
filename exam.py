import numpy as np
import cv2


# Run gui.py to see the app
# Do not change any name or prameters of any function or return just make shure u rturn grayscale image.

# These images are used to test in this file.
TEST_IMAGES = {
    'eren': 'data/images/Eren.jpg',
    'mikasa': 'data/images/Mikasa.jpg',
    'boy': 'data/images/Animeboy.jpg',
    'girl': 'data/images/Animegirl.jpg',
}

# complete these Kernels depends on what model do you have 
# make sure to search about 5X5 kernel and write down it here correctly
KERNEL = {
    'box': {
        '3x3': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]).astype('float') / 9,
        '5x5': np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]).astype('float') / 25,
    },
    'avg': {
        '3x3': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]).astype('float') / 1,
        '5x5': np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]).astype('float') / 1,
    },
    'prewitt': {
        'ver': np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]).astype('float'),
        'hor': np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]).astype('float'),
    },
    'sobel': {
        'hor': np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype('float'),
        'ver': np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype('float'),
    },
    'sharpen': np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]).astype('float'),
}

# You can use it in testing in this file 
# dont remove it because it's used in GUI 
# returns grayscale image
def load_grayscale(filename):
    # Read image
    image = cv2.imread(filename)
    
    # Retrun grayscale image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#################
#  Segmentation #
#################

# this function takes im (image) and thres (desired threshold) that u have entered in gui 
# don't care about how, just implement it here and test it here and it will give the correct result in gui
def apply_manual_thres(image, thres):
    return applyThereshold(image.copy(), thres)

# implement otsu's method here it takes the image
# returns the modified image and the optimal threshold 
# don't change any of return values please.
def otsu(image):
    # this varible will change when used outs's method
    final_thresh = -1
    
    # take a copy of an image to implement final_threshold to it aand return output
    im_out = image.copy()
    final_thresh, _ = cv2.threshold(im_out, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )
    return applyThereshold(im_out, final_thresh), final_thresh


def applyThereshold(image, thres): 
    height, width = image.shape
    for i in range(0, height):
        for j in range(0, width):
            if(image[i, j] >= thres):
                 image[i, j] = 255
            else:
                image[i, j] = 0
    return image
########################
#  Contrast Stretching #
########################
# apply histogram equalization and then return the modified image
def hist_eq(im):
    return im

# apply contrast stretching it takes the image
# and it returns the modified image
def contrast_stretch(im):
    return im


######################
#   Linear Filters   #
######################
# this function used in linear filter takes image and filter(kernel), and kernel size
# you can use it in box filter, average and any other linear filter
# this function returns the modified image 
# you can get the kernel with specific size from KERNEL dictionary by typing KERNEL[filter][size]
def apply_linear_filter(image, filter, size):
    im_out = image.copy()
    height, widht = im_out.shape
    # Getting offset (middle or array):
    offset = len(im_out) // 2
    for col in range(offset, widht - offset):
        for row in range(offset, height - offset):
            for i in range(len(height)):
                for j in range(len(height)):
                    X = col + i - offset
                    Y = row + j - offset
                    im_out[X, Y] *= KERNEL[filter][size][i][j]
                
    return im_out

######################
# Non-Linear Filters #
######################

def max_filter(img, Ksize):
    im_out = np.zeros(img.shape)

    return im_out

def min_filter(img, Ksize):
    im_out = np.zeros(img.shape)

    return im_out

def mean_filter(img, Ksize):
    im_out = np.zeros(img.shape)

    return im_out


######################
#   Edge Detection   #
######################
# you can find the kernel for edge detection by typing KERNEL[filter][type_]
# this function return modified image
def apply_edge_detection(image, filter, type_):
    im_out = image.copy()
    # write your code here

    return im_out

# it takes the vertical and horizontal images
# it returns the edge detection image with horizontal and vertical
def combine_both_edge_images(im_v, im_h):
    im_out = np.zeros(im_h.shape)
    # write ur code here

    return im_out

########################
# CANNY EDGE DETECTION #
########################
# its your turn to implement canny without any help. 
def canny(image):
    # write ur code here
    return image

##################
#   SHARPENING   #
##################
# a is the amount of sharpening if u implement prewitt it will add 1 if its enhanced 
# if u implement sobel u will add a to the sobel kernel.
# now you have addition and after that you have to add to KERNEL['sharpen'] that u choose to sharping filter in Kernels.
# so kernel will be kernel+addition
# and then apply filter for every pixel like any other linear filter.

def apply_sharpening_filter(image, a):
    addition = np.array([
        [0, 0, 0],
        [0, a, 0],
        [0, 0, 0]]).astype('float')
    
    im_output = image.copy()
    
    return im_output

######################
#  Time Transmission #
######################
# returns time transmission in seconds just type the equation and return the time
def time_transmission(img, baudrate, channels):
    time = 0
    # write ur code here
    
    return time


###################
#    Histogram    #
###################

# returns 1d array have its length = 255, for example at index 0 it has how many 0 appears
# if you implement this function means that u implement show_hist() and both_hist() 
# dont touch show_hist() and both_hist() it will work if you implement hist() correctly

def hist(grayscale):
    return [0, 1, 2, 3]

def show_hist(figure, grayscale):
    plt = figure.subplots()
    plt.clear()

    h = hist(grayscale)
    plt.bar(range(256), h)

    figure.canvas.draw()

def both_hist(figure, original, modified):
    plt = figure.subplots()
    plt.clear()
    
    h1 = hist(original)
    h2 = hist(modified)
    plt.plot(range(256), h1, color='r', label='original')
    plt.plot(range(256), h2, color='g', label='output')

    figure.canvas.draw()



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('For testing purposes')
    # for example
    print(hist(load_grayscale(TEST_IMAGES['eren'])))







