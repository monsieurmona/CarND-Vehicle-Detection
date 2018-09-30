from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import importlib
import ObjectDetection as od
importlib.reload(od)

import ObjectDetection as od
import importlib
importlib.reload(od)

import numpy as np
import time

plt_img_width = 10
plt_img_height = 8

images_path = 'test_images/large_set_car_non_car/**/*.png'



'''
print('Test Cell Per Block')
testparameters = [{
        'extract_spatial_features':False, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':2, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':False, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':False, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':4, 'hog_channel':'ALL',
        'C': 1
    },
]
'''

'''
print('Test Pixel per Cell')
testparameters = [{
        'extract_spatial_features':False, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':8, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':False, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':12, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':False, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
]
'''

'''
print('Test Spatial Binning')

testparameters = [{
        'extract_spatial_features':True, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':True, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':24, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':True, 'extract_color_features':False, 'extract_hog_features':True,
        'spatial':32, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
]
'''


'''
print('Test Histogram Features')

testparameters = [{
        'extract_spatial_features':True, 'extract_color_features':True, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':8,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':True, 'extract_color_features':True, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
    {
        'extract_spatial_features':True, 'extract_color_features':True, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':32,
        'color_space':'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 1
    },
]
'''

print('Test Color Spaces')
testparameters = [{
        'extract_spatial_features':True, 'extract_color_features':True, 'extract_hog_features':True,
        'spatial':16, 'histogram_bins':16,
        'color_space':'HSV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 0.001
    },
    {
        'extract_spatial_features': True, 'extract_color_features': True, 'extract_hog_features': True,
        'spatial': 16, 'histogram_bins': 16,
        'color_space': 'LUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 0.001
    },
    {
        'extract_spatial_features': True, 'extract_color_features': True, 'extract_hog_features': True,
        'spatial': 16, 'histogram_bins': 16,
        'color_space': 'HLS',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 0.001
    },
    {
        'extract_spatial_features': True, 'extract_color_features': True, 'extract_hog_features': True,
        'spatial': 16, 'histogram_bins': 16,
        'color_space': 'YUV',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 0.001
    },
    {
        'extract_spatial_features': True, 'extract_color_features': True, 'extract_hog_features': True,
        'spatial': 16, 'histogram_bins': 16,
        'color_space': 'YCrCb',

        'orient':9, 'pix_per_cell':16, 'cell_per_block':3, 'hog_channel':'ALL',
        'C': 0.001
    }
]



for setting in testparameters:
    od.train_single_svm(
        images_path,
        setting['color_space'],
        setting['extract_spatial_features'],
        setting['extract_color_features'],
        setting['extract_hog_features'],
        setting['spatial'],
        setting['histogram_bins'],
        setting['orient'],
        setting['pix_per_cell'],
        setting['cell_per_block'],
        setting['hog_channel'],
        LinearSVC(C=setting['C']))

