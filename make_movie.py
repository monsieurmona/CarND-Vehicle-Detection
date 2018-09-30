import ObjectDetection as od
import importlib
importlib.reload(od)
import inspect
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

plt_img_width = 10
plt_img_height = 8

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

import ObjectDetection as od
importlib.reload(od)
import object_tracking as ot
importlib.reload(ot)

from scipy.ndimage.measurements import label

import pickle

# load a pe-trained svc model from a serialized (pickle) file
#dist_pickle = pickle.load(open("models/udacity_pretrained_svc_pickle.p", "rb"))
dist_pickle = pickle.load(open("models/car_detection_svc_model_large.p", "rb"))

# get attributes of our svc object
svcs = dist_pickle["svc"]
X_scalers = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
hog_channel = dist_pickle["hog_channel"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
extract_spatial_features = dist_pickle["extract_spatial_features"]
extract_color_features = dist_pickle["extract_color_features"]
extract_hog_features = dist_pickle["extract_hog_features"]

#video_file_name = "test_video"
video_file_name = "project_video"
video_file_name_ext = ".mp4"
output_video_file_name = "output_video/"+video_file_name + "_output" + video_file_name_ext
video = VideoFileClip("test_video/" + video_file_name + video_file_name_ext)#.subclip(28,31)

object_tracking = ot.ObjectTracking()

output_video = video.fl_image(lambda image: ot.car_detection_pipeline(
    image=image,
    ystart=400, detection_window_size=64,
    scale_min=1.0, scale_max=4.0, steps=5,
    svcs=svcs, X_scalers=X_scalers,
    orient=orient,
    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    hog_channel=hog_channel,
    spatial_size=spatial_size, hist_bins=hist_bins,
    color_space=color_space,
    extract_spatial_features=extract_spatial_features,
    extract_color_features=extract_color_features,
    extract_hog_features=extract_hog_features,
    object_tracking=object_tracking
))

# %time output_video.write_videofile(output_video_file_name, audio=False)
output_video.write_videofile(output_video_file_name, audio=False)
print("Done")