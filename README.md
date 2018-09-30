# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

My goal was for this project to write a software pipeline to detect vehicles in a video.


The Project
---

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
    
**ParameterC**

I use ``GridSearchCV`` from scikit-learn to train a classifier with different parameters. I chose LinearSVM with ``C = [{'C': [0.00001, 0.0001,0.001,0.01,0.1, 1,10]}]``

Best parameters set found is:

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

|C|Color<br>Space|orien-<br>tations|Cells<br>per<br>block|Pixel<br>per<br>cell|Spatial<br>Binning<br>Size|Histo-<br>gram<br>Bins|Accu-<br>racy|Preci-<br>sion|Recall|Feat|Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0.001|HSV|9|3|16|16x16|16|0.993|0.996|0.990|1788|1.18 s|
|0.001|HLS|9|3|16|16x16|16|0.992|0.995|0.988|1788|1.41 s|
|0.001|LUV|9|3|16|16x16|16|0.991|0.996|0.986|1788|2.87 s|
|0.001|YUV|9|3|16|16x16|16|0.990|0.994|0.986|1788|2.07 s|
|0.001|YCrCb|9|3|16|16x16|16|0.990|0.993|0.987|1788|2.15 s|

Three tests per color space were executed and the average of the results were put into this table. The performace is similar in this table, but when the classifiers are used in the video they show very different performances. HSV fails to classify white cars for example. As LUV generalizes better, I choose this color space.


### Classification
The trained classifier, its parameters and the scaler is loaded from the pickle file 

#### Sliding Window

#### Heatmap

#### Tracking

#### Filter










#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

