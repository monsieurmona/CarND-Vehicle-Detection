# Vehicle Detection [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
by Mario Lüder

The Project
---
My goal was to write a software pipeline to detect vehicles in a video.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. Here is a link to the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment training data.  

[//]: # (Image References)
[image_car]: ./output_images/hog/car.png
[image_notcar]: ./output_images/hog/notcar.png
[image_cell_per_block_compare]: ./output_images/hog/cell_per_block_compare.png
[image_pixel_per_cell_compare_notcar]: ./output_images/hog/pixel_per_cell_compare_notcar.png
[image_pixel_per_cell_compare]: ./output_images/hog/pixel_per_cell_compare.png
[image_spatial_binnig]: ./output_images/color/spatial_binning.png
[image_spatial_binnig_features]: ./output_images/color/spatial_binning_features.png
[image_luv_histogram]: ./output_images/color/luv_histogram.png
[image_luv_histogram_image]: ./output_images/color/luv_histogram_image.png
[image_concatenate_features]: ./output_images/color/concatenate_features.png
[image_training_learning_rate_HSV]: ./output_images/training/training_learning_rate_HSV.png
[image_training_learning_rate_LUV]: ./output_images/training/training_learning_rate_LUV.png
[image_sliding_window_car_tops]: ./output_images/sliding_window/sliding_window_car_tops.jpg
[image_sliding_window_scale_1]: ./output_images/sliding_window/sliding_window_scale_1.png
[image_sliding_window_scale_2]: ./output_images/sliding_window/sliding_window_scale_2.png
[image_sliding_window_scale_3]: ./output_images/sliding_window/sliding_window_scale_3.png
[image_sliding_window_scale_4]: ./output_images/sliding_window/sliding_window_scale_4.png
[image_sliding_window_scale_5]: ./output_images/sliding_window/sliding_window_scale_5.png
[image_classified]: ./output_images/sliding_window/classified.png
[image_heatmap]: ./output_images/sliding_window/heat_map.png



[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation. The implemenation consists of three parts.
* ObjectDetection.ipynb (IPython notebook used as frontend to display and compare results)
* ObjectDetection.py (all functions for object detection)
* object_tracking.py (all functions for object tracking and the pipeline)

### Features
#### Histogram of Oriented Gradients (HOG)

The code to evaluate HOG images is contained in the IPython notebook in section "HOG image" and "Not Car HOG". I selected a car image as well as a not-car image randomly and compared different HOG parameters.

|Car|Not Car|
|:---:|:---:|
| ![alt text][image_car]<br>![alt text][image_pixel_per_cell_compare] | ![alt text][image_notcar]<br>![alt text][image_pixel_per_cell_compare_notcar] |

The gradients are very different for the two images. With a bit of fantasy you may "see" a car in the HOG visualization for the car image. 8 Pixels per cell seem to be quit noisy. Whereas 12 Pixel per cell shape the outline and the structure of the car much better. 16 pixel per cell seem to provide similar results. 

I compared also visualization of different "cell per block parameters"
![alt text][image_cell_per_block_compare]

But I could not see any difference yet.

#### Spatial Color Binning Features and Color Histogram Features

Besides HOG, raw pixel seem to be useful to detect cars. As we want just enough information, we resize the image to a smaller resolution and use the binned pixels as features.

![alt text][image_spatial_binnig]

Original:
![alt text][image_spatial_binnig_features]

Even 16x16 pixel seem to retain enough information to detect a car.

Also the color histogram might be useful for car detection. This is an example for th LUV color space.

![alt text][image_luv_histogram_image]

![alt text][image_luv_histogram]
 

#### Color Space

I have tested the feature extraction with different color spaces as those affect all three feature type.
* HSV
* LUV
* HLS
* YUV
* YCrCb

#### Feature Combining

These three features are combined in a singe feature vector and then normalized using the [`StandardScaler()` from scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). See "Concatenate Feature" section in the IPython file. 

![alt_text][image_concatenate_features]


### Training

I use the implementation of a linear Support Vector Machine from scikit-learn to train a classifier. See section "Single Classification" in the IPython file. The result is then stored in a "pickle" file "models/car_detection_svc_model_large.p" 

#### Image Augmentation

I have chosen to use the images from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). I have flipped all images around the vertical axis to get more images for training.  

#### Sequence
1. Get all image file from car images and from not car images
1. Load images
1. Augment images
1. Extract features from images
1. Shuffle and split into training set and test set
1. Fit a scaler on the training set
1. Use the scaler to normalize the training set and test set
1. Train the classifier with the training set
1. Test the classifier with test set
1. Calculate Precision and Recall
1. Dump the results
1. Store Classifier and Scaler with parameters in a pickle file

#### Parameter Decision

I change only one parameter at the time, try to find the best and continue to find the next parameter.

**HOG**

Pixel per cell selection:

|C|Color<br>Space|orien-<br>tations|Cells<br>per<br>block|Pixel<br>per<br>cell|Spatial<br>Binning<br>Size|Histo-<br>gram<br>Bins|Accu-<br>racy|Preci-<br>sion|Recall|Feat|Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|LUV|9|3|**8**|none|none|0.981|0.981|0.981|8748|28.22 s|
|1|LUV|9|3|**12**|none|none|0.972|0.970|0.974|2187|9.45 s|
|1|LUV|9|3|**16**|none|none|0.974|0.973|0.974|972|7.19 s|

Even though the accuracy, precision and recall are best with 8 pixel per cell, I choose 16 pixel per cell as the amount of features is much smaller. I loose a bit of accuracy and gain speed.  

|C|Color<br>Space|orien-<br>tations|Cells<br>per<br>block|Pixel<br>per<br>cell|Spatial<br>Binning<br>Size|Histo-<br>gram<br>Bins|Accu-<br>racy|Preci-<br>sion|Recall|Feat|Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|LUV|9|**2**|16|none|none|0.972|0.976|0.968|972|6.02 s|
|1|LUV|9|**3**|16|none|none|0.974|0.972|0.977|972|6.95 s|
|1|LUV|9|**4**|16|none|none|0.974|0.973|0.974|432|6.32 s|

I do not want to reduce more features. So I choose 3 cells per block. 

**Spatial Binning**

|C|Color<br>Space|orien-<br>tations|Cells<br>per<br>block|Pixel<br>per<br>cell|Spatial<br>Binning<br>Size|Histo-<br>gram<br>Bins|Accu-<br>racy|Preci-<br>sion|Recall|Feat|Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|LUV|9|3|16|**16x16**|none|0.982|0.986|0.977|1740|4.63 s|
|1|LUV|9|3|16|**24x24**|none|0.98|0.990|0.969|2700|7.8 s|
|1|LUV|9|3|16|**32x32**|none|0.982|0.989|0.975|4044|12.88 s|

All three tests boost the performance by roughly the same amount. Thus 16x16 would be the best option to go as this has the smallest set of features.

**Histogram Features**

|C|Color<br>Space|orien-<br>tations|Cells<br>per<br>block|Pixel<br>per<br>cell|Spatial<br>Binning<br>Size|Histo-<br>gram<br>Bins|Accu-<br>racy|Preci-<br>sion|Recall|Feat|Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|LUV|9|3|16|16x16|**8**|0.983|0.987|0.978|1764|4.28 s|
|1|LUV|9|3|16|16x16|**16**|0.987|0.989|0.986|1788|4.25 s|
|1|LUV|9|3|16|16x16|**32**|0.984|0.987|0.981|1836|4.11 s|

There is only a small gain here. 16 Bins for a histogram seem to be the best.
    
**Parameter C**

I chose to use LinearSVM to train a classifier with parameter C = ``[{'C': [0.00001, 0.0001,0.001,0.01,0.1, 1,10]}]``. In order to get quick results and comprehensible documentation I have use ``GridSearchCV`` from scikit-learn to execute this test. The best parameters C is:

C = 0.001

Grid scores on test set:
```
0.979 (+/-0.004) for {'C': 1e-05}
0.989 (+/-0.003) for {'C': 0.0001}
0.990 (+/-0.004) for {'C': 0.001}
0.987 (+/-0.003) for {'C': 0.01}
0.985 (+/-0.004) for {'C': 0.1}
0.984 (+/-0.003) for {'C': 1}
0.984 (+/-0.003) for {'C': 10}
```

Detailed classification report:

```
             precision    recall  f1-score   support

        0.0      0.988     0.994     0.991      3544
        1.0      0.994     0.988     0.991      3560

avg / total      0.991     0.991     0.991      7104
```

**Color Space** 

The used color space effects also the performance of the classifier. Therefor I conducted a series of test with different color spaces as seen below. 

|C|Color<br>Space|orien-<br>tations|Cells<br>per<br>block|Pixel<br>per<br>cell|Spatial<br>Binning<br>Size|Histo-<br>gram<br>Bins|Accu-<br>racy|Preci-<br>sion|Recall|Feat|Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0.001|HSV|9|3|16|16x16|16|0.993|0.996|0.990|1788|1.18 s|
|0.001|HLS|9|3|16|16x16|16|0.992|0.995|0.988|1788|1.41 s|
|0.001|LUV|9|3|16|16x16|16|0.991|0.996|0.986|1788|2.87 s|
|0.001|YUV|9|3|16|16x16|16|0.990|0.994|0.986|1788|2.07 s|
|0.001|YCrCb|9|3|16|16x16|16|0.990|0.993|0.987|1788|2.15 s|

Three tests per color space were executed and the average of the results were put into this table. The performace is similar in this table, but when the classifiers are used in the video they show very different performances. HSV fails to classify white cars for example. As LUV generalizes better, I choose this color space.

**Learning Rates**

Finally, I want to check if the classifier is overfitting. 

|Learning Rates for a classifier<br>with HSV color space|Learning Rates for a classifier<br>with LUV color space|
|:---:|:---|
|![alt text][image_training_learning_rate_HSV]|![alt text][image_training_learning_rate_LUV]|
The HSV classifier doesn't improve anymore aver 17000 training examples, whereas the LUV classifier would probably benefit from more training examples than available.


### Classification
The previously trained classifier, its parameters and the scaler is loaded from the pickle file. This is then used in the following pipeline.

1. Load Video Frame
1. Convert the image to the desired color space, calculate HOG
1. Create sliding windows of multiple scales
1. For each sliding window
   1. Move the sliding window from top left to bottom right of the image (start at a given vertical offset, stop at a given vertical offset)
   1. Use the trained classifier to detect a car for each window
   1. Push the window shape and a confidence level to a list 
1. Create a heat map with the previously found windows and confidence level
1. Extract bounding boxes from heatmaps 
1. Track bounding boxes and increase its age at every frame. Combine heatmaps from previous detection with new detection.
1. Extract bounding boxes from combined heatmaps
1. Delete detections if they were not update for a few rounds.
1. Show bounding boxes, if they are old enough   
    
#### Preprocessing

A frame from video is first converted to the color space that was just while creating the classifier. The HOG is then calculated for each channel for the the image with the same parameters.   
    
#### Sliding Window

Cars are detected using a sliding window that cuts a rectangle from the image. The cutouts are classified then as car or as not-car. As there are cars that appear small and big in an image we need sliding windows of different scales. 

Searching for cars in the sky would be for this task a waste of computation time. And searching for small cars in front of the vehicle is also not beneficial. So we need to define the search areas. 

The camera was probably mounted close to the mirror in the middle of the car. The horizon and the car tops are roughly inline, no matter how far they are. The blue line in the picture below shows this. 

![alt_text][image_sliding_window_car_tops]

This means that the tops of the sliding windows of all scales should start at the same height. The images below show the 5 scales I have chosen.

|![alt_text][image_sliding_window_scale_1]|![alt_text][image_sliding_window_scale_2]|![alt_text][image_sliding_window_scale_3]|
|:---:|:---:|:---:|
|![alt_text][image_sliding_window_scale_4]|![alt_text][image_sliding_window_scale_5]| |

The bottom height is calculated by:
```
ystop = int(ystart + detection_window_size * scale * 1.6)
```
where `detection_window_size = 64`. This is the size of the trained classifier. 
  
The HOG, spatial and histogram features are extracted from the sliding window and then used for classification. 

![alt_text][image_classified]

#### Heatmap

The measured confidence for an rectangle is used to built a heatmap and then threshold by a confidence level. The heatmap is labels with the `label` function from `scipy.ndimage.measurements`. The heatmap and resulting bounding boxes are shown below. 

![alt_text][image_heatmap]

As very high confidences are not useful to track cars the heat map is clipped at a certain threshold.

#### Tracking
The object tracking is implemented in file ```object_tracking.py```. See function `def track(self, detections, heat_map)`

For each detection (extracted bounding boxes from clipped heatmaps) we try to find a corresponding detection from previous frame. The detections must have a similar size and must be close to each other. If there are multiple choices, I choose the closest. The heatmap of the previous detection is weakened to reduce the influence of the previous detections. An empty heatmap of the size of an image is created and the two heatmaps of the previous and current detection is added. This heatmap is labeled again. The new detections are stored as last detection with an increased age. Those detections are used as "previous detections" while tracking detections in the next frame. 

Old enough detections will be provided as bounding boxes and drawn into the video frame.

Previous detections that don't match any current detection are also aged (heatmap weakended) and their "last updated" parameter is increased. If their last update is too long ago, they get deleted. New detections that are not related to previous detections are stored as last detections for the next tracking round.

#### Pipeline 

The object tracking is implemented in file `object_tracking.py`. See function `def car_detection_pipeline(...)` which calls the steps written above. 

The video is created with file  `make_movie.py`

### Video

You may see the result in a video: Here's a [link to my video result](./output_video/project_video_output.mp4)

### Discussion

I belief that using three classifiers, for cars seen on left, straight and right side would be beneficial. Also using depth of field would probably increase the accuracy of detections a lot. 

Improving tracking by analysing more similarity properties (similar colors, similar HOG) could also help.

I believe that getting the motion of the detected vehicles and predicting where they would be in the next frame will improve the tracking algorithm. 