from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import ntpath
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.feature_selection import RFECV, RFE

from sklearn.pipeline import Pipeline

from scipy.ndimage.measurements import label


# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
#from sklearn.cross_validation import train_test_split

# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split

# draw boxes into to a copy of an input image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)

    for box in bboxes:
        if len(box) == 3:
            #box_color = [0, 0, 0]
            #box_color[int(box[2] - 1)] = 1
            #color = list(box_color)
            color = box[2]

        cv2.rectangle(draw_img, box[1], box[0], color, thick)

    return draw_img


def plot3d(pixels, colors_rgb,axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def plot3dOnFigure(ax, pixels, colors_rgb,axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

# compute an color histogram
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features


# Define a function to compute color histogram features
# color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != '':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

# Extract meta data of the data set
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


# get hog features
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=False,
                       visualize=vis, feature_vector=feature_vec)
        return features

# converts and RGB image to a desried color space
def get_color_space(image, cspace='RGB'):
    converted_image = None

    if cspace != '':
        if cspace == 'HSV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            converted_image = np.copy(image)
    else:
        converted_image = np.copy(image)

    return converted_image


def vstack_features(feature_lists):
    X = None

    for feature_list in feature_lists:
        if (len(feature_list) > 0):
            if (X is None):
                X = (feature_list)
            else:
                X = np.vstack((
                    X,
                    feature_list))

    return X


# extract features from an image
def extract_features_single_image(
        image,
        extract_spatial_features=True, extract_color_features=True, extract_hog_features=True,
        spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
        orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0):
    single_image_feature = []

    if (extract_spatial_features == True):
        # Apply bin_spatial() to get spatial color features
        single_image_feature.append(bin_spatial(image, size=spatial_size))

    if (extract_color_features == True):
        _, _, _, _, hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
        single_image_feature.append(hist_features)

    if (extract_hog_features == True):
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = []
        hog_features_histogram = []
        if hog_channel == 'ALL':
            for channel in range(image.shape[2]):
                hog_features_channel = (get_hog_features(image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features_histogram_channel = hog_features_channel.reshape(int(len(hog_features_channel) / orient), orient)
                hog_features_histogram_channel = np.sum(hog_features_histogram_channel, axis=0)

                hog_features.extend(hog_features_channel)
                hog_features_histogram.extend(hog_features_histogram_channel)

        else:
            hog_features = get_hog_features(image[:, :, hog_channel],
                                            orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        single_image_feature.append(hog_features)
        single_image_feature.append(hog_features_histogram)

    return np.concatenate(single_image_feature)


# extract features from a list of images and normalize them
def extract_features(imgs, cspace='RGB',
                     extract_spatial_features=True, extract_color_features=True, extract_hog_features=True,
                     spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                     orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0, flip = 0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        feature_image = get_color_space(image, cspace)

        features.append(extract_features_single_image(
            image=feature_image,
            extract_spatial_features = extract_spatial_features,
            extract_color_features = extract_color_features,
            extract_hog_features=extract_hog_features,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            hist_range=hist_range,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel
        ))

        if flip > 0:
            features.append(extract_features_single_image(
                image=cv2.flip(feature_image, 1),
                extract_spatial_features = extract_spatial_features,
                extract_color_features = extract_color_features,
                extract_hog_features=extract_hog_features,
                spatial_size=spatial_size,
                hist_bins=hist_bins,
                hist_range=hist_range,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel
            ))
        '''
        if flip > 1:
            features.append(extract_features_single_image(
                image=cv2.flip(feature_image, -1),
                extract_spatial_features = extract_spatial_features,
                extract_color_features = extract_color_features,
                extract_hog_features=extract_hog_features,
                spatial_size=spatial_size,
                hist_bins=hist_bins,
                hist_range=hist_range,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel
            ))

            features.append(extract_features_single_image(
                image=cv2.flip(feature_image, 0),
                extract_spatial_features = extract_spatial_features,
                extract_color_features = extract_color_features,
                extract_hog_features=extract_hog_features,
                spatial_size=spatial_size,
                hist_bins=hist_bins,
                hist_range=hist_range,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel
            ))
        '''

    # Return list of feature vectors
    return features

# get file names for car and non car images
def get_image_filename_sets(image_path):
    images = glob.glob(image_path, recursive=True)

    notcars = []
    cars_left = []
    cars_middle = []
    cars_right = []
    cars_unknown = []

    for image in images:
        #image_basename = ntpath.basename(image)
        if '/non' in image:
            notcars.append(image)
        elif 'Left' in image:
            cars_left.append(image)
        elif 'Middle' in image or 'Far' in image:
            cars_middle.append(image)
        elif 'Right' in image:
            cars_right.append(image)
        else:
            cars_unknown.append(image)

    print("Non Vehicles:", len(notcars),
          " Vehicles Left", len(cars_left),
          " Middle:", len(cars_middle),
          " Right:", len(cars_right),
          " Unknown:", len(cars_unknown))

    #return notcars[:1000], cars_left[:500], cars_middle[:500], cars_right[:500]
    return notcars, cars_left, cars_middle, cars_right, cars_unknown

def get_features(notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, cars_unknown_filenames,
                 extract_spatial_features=True, extract_color_features=True, extract_hog_features=True,
                 color_space='RGB', spatial_size=(32, 32),
                 histogram_bins=32, histogram_range=(0, 256),
                 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    notcar_features = extract_features(notcars_filenames,
                                    extract_spatial_features=extract_spatial_features,
                                    extract_color_features=extract_color_features,
                                    extract_hog_features=extract_hog_features,
                                    cspace=color_space, spatial_size=spatial_size,
                                    hist_bins=histogram_bins, hist_range=histogram_range,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel = hog_channel, flip=2)

    car_left_features = extract_features(cars_left_filenames,
                                       extract_spatial_features=extract_spatial_features,
                                       extract_color_features=extract_color_features,
                                       extract_hog_features=extract_hog_features,
                                       cspace=color_space, spatial_size=spatial_size,
                                       hist_bins=histogram_bins, hist_range=histogram_range,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, flip=1)

    car_middle_features = extract_features(cars_middle_filenames,
                                       extract_spatial_features=extract_spatial_features,
                                       extract_color_features=extract_color_features,
                                       extract_hog_features=extract_hog_features,
                                       cspace=color_space, spatial_size=spatial_size,
                                       hist_bins=histogram_bins, hist_range=histogram_range,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, flip=1)

    car_right_features = extract_features(cars_right_filenames,
                                       extract_spatial_features=extract_spatial_features,
                                       extract_color_features=extract_color_features,
                                       extract_hog_features=extract_hog_features,
                                       cspace=color_space, spatial_size=spatial_size,
                                       hist_bins=histogram_bins, hist_range=histogram_range,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, flip=1)


    car_unknown_features = extract_features(cars_unknown_filenames,
                                       extract_spatial_features=extract_spatial_features,
                                       extract_color_features=extract_color_features,
                                       extract_hog_features=extract_hog_features,
                                       cspace=color_space, spatial_size=spatial_size,
                                       hist_bins=histogram_bins, hist_range=histogram_range,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, flip=1)

    return notcar_features, car_left_features, car_middle_features, car_right_features, car_unknown_features

# train an provide an SVM classifier
def train_svm(images_path, color_space,
              extract_spatial_features, extract_color_features, extract_hog_features,
              spatial, histogram_bins,
              orient, pix_per_cell, cell_per_block, hog_channel,
              clfs
              ):
    notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, cars_unknown_filenames = \
        get_image_filename_sets(images_path)

    # load the images and provide the extracted features
    # of car and non car images
    notcar_features, car_left_features, car_middle_features, car_right_features, _ = get_features(
        notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, [],
        extract_spatial_features=extract_spatial_features,
        extract_color_features=extract_color_features,
        extract_hog_features=extract_hog_features,
        color_space=color_space, spatial_size=(spatial, spatial),
        histogram_bins=histogram_bins, histogram_range=(0, 256),
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    # Create an array stack of feature vectors
    #X = np.vstack((notcar_features, car_left_features, car_middle_features, car_right_features)).astype(np.float64)

    # Define the labels vector
    #y = np.hstack((
    #    np.zeros(len(notcar_features)),
    #    np.ones(len(car_left_features)),
    #    np.ones(len(cars_middle_filenames)) * 2,
    #    np.ones(len(cars_right_filenames)) * 3))

    X_left   = np.vstack((notcar_features, car_left_features)).astype(np.float64)
    X_middle = np.vstack((notcar_features, car_middle_features)).astype(np.float64)
    X_right  = np.vstack((notcar_features, car_right_features)).astype(np.float64)

    y_left   = np.hstack((np.zeros(len(notcar_features)), np.ones(len(car_left_features)) ))
    y_middle = np.hstack((np.zeros(len(notcar_features)), np.ones(len(car_middle_features)) ))
    y_right  = np.hstack((np.zeros(len(notcar_features)), np.ones(len(car_right_features)) ))

    print(len(y_left), " ", len(X_left))
    assert(len(y_left) == len(X_left))
    assert(len(y_middle) == len(X_middle))
    assert(len(y_right) == len(X_right))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train_left, X_test_left, y_train_left, y_test_left = train_test_split(
        X_left, y_left, test_size=0.2, random_state=rand_state)

    X_train_middle, X_test_middle, y_train_middle, y_test_middle = train_test_split(
        X_middle, y_middle, test_size=0.2, random_state=rand_state)

    X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(
        X_right, y_right, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler only on the training data
    X_scaler_left = StandardScaler().fit(X_train_left)
    X_scaler_middle = StandardScaler().fit(X_train_middle)
    X_scaler_right = StandardScaler().fit(X_train_right)

    # Apply the scaler to X_train and X_test
    X_train_left = X_scaler_left.transform(X_train_left)
    X_test_left = X_scaler_left.transform(X_test_left)

    X_train_middle = X_scaler_middle.transform(X_train_middle)
    X_test_middle = X_scaler_middle.transform(X_test_middle)

    X_train_right = X_scaler_right.transform(X_train_right)
    X_test_right = X_scaler_right.transform(X_test_right)

    print('Using spatial binning of:', spatial,
          'and', histogram_bins, 'histogram bins')
    print('Feature vector length:', len(X_train_left[0]))

    # Check the training time for the SVC
    t = time.time()
    clfs[0].fit(X_train_left, y_train_left)
    clfs[1].fit(X_train_middle, y_train_middle)
    clfs[2].fit(X_train_right, y_train_right)
    t2 = time.time()

    print(round(t2 - t, 2), 'Seconds to train SVC...')

    y_pred_left = clfs[0].predict(X_test_left)
    y_pred_middle = clfs[0].predict(X_test_middle)
    y_pred_right = clfs[0].predict(X_test_right)

    # Check the score of the SVC
    print('Test Accuracy of SVC Left   = ', round(clfs[0].score(X_test_left, y_test_left), 4))
    print('    Precision:', precision_score(y_test_left, y_pred_left))
    print('    Recall:', recall_score(y_test_left, y_pred_left))
    print('Test Accuracy of SVC Middle = ', round(clfs[1].score(X_test_middle, y_test_middle), 4))
    print('    Precision:', precision_score(y_test_middle, y_pred_middle))
    print('    Recall:', precision_score(y_test_middle, y_pred_middle))
    print('Test Accuracy of SVC Right  = ', round(clfs[2].score(X_test_right, y_test_right), 4))
    print('    Precision:', precision_score(y_test_right, y_pred_right))
    print('    Recall:', recall_score(y_test_right, y_pred_right))

    return clfs, \
           [X_scaler_left, X_scaler_middle, X_scaler_right], \
           [X_test_left, X_test_middle, X_test_right], \
           [y_test_left, y_test_middle, y_test_right]


# train an provide an SVM classifier
def train_single_svm(images_path, color_space,
              extract_spatial_features, extract_color_features, extract_hog_features,
              spatial, histogram_bins,
              orient, pix_per_cell, cell_per_block, hog_channel,
              clf
              ):
    notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, cars_unknown_filenames = \
        get_image_filename_sets(images_path)

    # load the images and provide the extracted features
    # of car and non car images
    notcar_features, car_left_features, car_middle_features, car_right_features, cars_unknown_features = get_features(
        notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, cars_unknown_filenames,
        extract_spatial_features=extract_spatial_features,
        extract_color_features=extract_color_features,
        extract_hog_features=extract_hog_features,
        color_space=color_space, spatial_size=(spatial, spatial),
        histogram_bins=histogram_bins, histogram_range=(0, 256),
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    # Create an array stack of feature vectors
    X = vstack_features((
        notcar_features,
        car_left_features,
        car_middle_features,
        car_right_features,
        cars_unknown_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((
        np.zeros(len(notcar_features)),
        np.ones(len(car_left_features)),
        np.ones(len(car_middle_features)),
        np.ones(len(car_right_features)),
        np.ones(len(cars_unknown_features))
    ))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)

    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print("--- train_single_svm ---")
    print("Extract spatial features:", extract_spatial_features,
          " Extract hist features:", extract_color_features,
          " Extract hog features:", extract_hog_features,
          " Color space", color_space,
          " Spatial size (", spatial,",",spatial,")",
          " Histogram Bins:", histogram_bins,
          " HOG Orientations:",orient,
          " HOG Pixel per Cell:", pix_per_cell,
          " HOG Cell per Block:", cell_per_block,
          " HOG Channel(s)", hog_channel)

    print('Feature vector length:', len(X_train[0]))

    # Check the training time for the SVC
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2 - t, 2), 'Seconds to train SVC...')

    y_pred = clf.predict(X_test)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    print('    Precision:', precision_score(y_test, y_pred))
    print('    Recall:', recall_score(y_test, y_pred))

    return [clf], \
           [X_scaler], \
           [X_test], \
           [y_test]

# provide curves for model selection
def model_selection_learning_single_svm(images_path, color_space,
              extract_spatial_features, extract_color_features, extract_hog_features,
              spatial, histogram_bins,
              orient, pix_per_cell, cell_per_block, hog_channel,
              clf
              ):
    notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, cars_unknown_filenames = \
        get_image_filename_sets(images_path)

    # load the images and provide the extracted features
    # of car and non car images
    notcar_features, car_left_features, car_middle_features, car_right_features, cars_unknown_features = get_features(
        notcars_filenames, cars_left_filenames, cars_middle_filenames, cars_right_filenames, cars_unknown_filenames,
        extract_spatial_features=extract_spatial_features,
        extract_color_features=extract_color_features,
        extract_hog_features=extract_hog_features,
        color_space=color_space, spatial_size=(spatial, spatial),
        histogram_bins=histogram_bins, histogram_range=(0, 256),
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    # Create an array stack of feature vectors

    # Create an array stack of feature vectors
    X = vstack_features((
        notcar_features,
        car_left_features,
        car_middle_features,
        car_right_features,
        cars_unknown_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((
        np.zeros(len(notcar_features)),
        np.ones(len(car_left_features)),
        np.ones(len(car_middle_features)),
        np.ones(len(car_right_features)),
        np.ones(len(cars_unknown_features))
    ))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=rand_state)

    X_scaler = StandardScaler()

    pipeline = Pipeline(steps=[('scaler', X_scaler),
                               ('clf', clf)])

    train_sizes = np.linspace(.1, 1.0, 8)
    t = time.time()
    train_sizes, train_scores, test_scores = \
        learning_curve(pipeline, X, y, train_sizes=train_sizes, cv=cv, n_jobs=8)
    t2 = time.time()

    print(round(t2 - t, 2), 'Seconds to get the curve...')
    return train_sizes, train_scores, test_scores, y




def predict(svc, X_test, y_test, n_predict = 10):
    # Check the prediction time for a single sample
    t = time.time()
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

# Provides a list of boxes by sliding of er a given image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    if nx_pix_per_step == 0 or ny_pix_per_step == 0:
        return []

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


#
# Predict car not-cars in provided window boundaries
def search_windows(image, windows, clfs, scalers, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, extract_spatial_features=True,
                   extract_color_features=True, extract_hog_features=True):
    # Create an empty list to receive positive detection windows
    on_windows = []

    # Change color space
    new_image = get_color_space(image, color_space)

    # iterate over all classifiers
    for i in range(len(clfs)):
        clf = clfs[i]
        scaler = scalers[i]

        # Iterate over all windows in the list
        for window in windows:
            # Extract the test window from original image
            test_img = cv2.resize(new_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # Extract features for that window using single_img_features()
            features = extract_features_single_image(
                test_img,
                spatial_size=spatial_size, hist_bins=hist_bins,
                orient=orient, pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel,
                extract_spatial_features=extract_spatial_features,
                extract_color_features=extract_color_features,
                extract_hog_features=extract_hog_features)

            # scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # Predict using your classifier
            prediction = clf.predict(test_features)
            # If prediction == 1 (left) or  prediction == 2 (middle) or prediction == 3 (right)then save the window
            draw_all = False
            if prediction == 1 or prediction == 2 or prediction == 3 or draw_all is True:
                on_windows.append(window)

    # Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars_in_sub_window(
        image, ystart, ystop, scale, svcs, X_scalers, orient, pix_per_cell, cell_per_block, hog_channel,
        spatial_size, hist_bins,
        color_space, extract_spatial_features, extract_color_features, extract_hog_features,
        draw):
    draw_image = np.copy(image)
    detection_boxes = []
    draw_all = False

    image_tosearch = image[ystart:ystop, :, :]
    ctrans_tosearch = get_color_space(image_tosearch, cspace=color_space)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nx_blocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    ny_blocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step + 1
    ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step + 1

    if (extract_hog_features == True):
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    classification_results = np.zeros(len(svcs))
    decision_values = np.zeros(len(svcs))

    for xb in range(nx_steps):
        for yb in range(ny_steps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            feature_window = []

            if (extract_spatial_features == True or extract_color_features == True):
                # Extract the image patch
                sub_image = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                if (extract_spatial_features == True):
                    spatial_features = bin_spatial(sub_image, size=(spatial_size, spatial_size))
                    feature_window.append(spatial_features)

                if (extract_color_features == True):
                    _, _, _, _, hist_features = color_hist(sub_image, nbins=hist_bins)
                    feature_window.append(hist_features)

            if (extract_hog_features == True):
                # Extract HOG for this patch

                if (extract_hog_features == True):
                    # Call get_hog_features() with vis=False, feature_vec=True
                    hog_features = []
                    hog_features_histogram = []
                    if hog_channel == 'ALL':
                        hog_features1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_features2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_features3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_features1, hog_features2, hog_features3))

                        hog_features_histogram_1 = hog_features1.reshape(
                            int(len(hog_features1) / orient), orient)
                        hog_features_histogram_1 = np.sum(hog_features_histogram_1, axis=0)

                        hog_features_histogram_2 = hog_features2.reshape(
                            int(len(hog_features2) / orient), orient)
                        hog_features_histogram_2 = np.sum(hog_features_histogram_2, axis=0)

                        hog_features_histogram_3 = hog_features3.reshape(
                            int(len(hog_features3) / orient), orient)
                        hog_features_histogram_3 = np.sum(hog_features_histogram_3, axis=0)

                        hog_features_histogram = np.hstack((hog_features_histogram_1, hog_features_histogram_2, hog_features_histogram_3))

                    else:
                        if hog_channel == 0:
                            hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        if hog_channel == 1:
                            hog_features = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        if hog_channel == 2:
                            hog_features = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                    feature_window.append(hog_features)
                    feature_window.append(hog_features_histogram)

            # Scale features and make a prediction
            feature_window = np.concatenate(feature_window)

            # test with all classifier
            for i in range(len(svcs)):
            #for i in range(1,2):
                svc = svcs[i]
                X_scaler = X_scalers[i]

                test_features = X_scaler.transform(
                    np.array(feature_window).reshape(1, -1))

                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                #test_prediction = svc.predict(test_features)
                decision_value = svc.decision_function(test_features)
                test_prediction = int(decision_value > 0.0)

                classification_results[i] = test_prediction
                decision_values[i] = decision_value + 0.1

            #if test_prediction > 0 or draw_all == True:
            #draw_all = True
            if 1 in classification_results or draw_all is True:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = [(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                       list(classification_results), list(decision_values)]
                detection_boxes.append(box)

                if draw == True  or draw_all == True:
                    cv2.rectangle(draw_image, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                                  list(classification_results), 6)

    return detection_boxes, draw_image


# finds detects cars in an images and increases the detection window
# while going down the image
def find_cars(
        image, ystart, detection_window_size, scale_min, scale_max, steps,
        svcs, X_scalers, orient, pix_per_cell, cell_per_block, hog_channel,
        spatial_size, hist_bins,
        color_space, extract_spatial_features, extract_color_features, extract_hog_features):

    all_detection_boxes = []
    ymax = int(ystart + detection_window_size * scale_max)


    for scale in np.linspace(scale_min, scale_max, steps):
        ystop = int(ystart + detection_window_size * scale * 1.6)

        if (ystop > ymax):
            ystop = ymax

        detection_boxes, out_image = find_cars_in_sub_window(
            image=image,
            ystart=ystart, ystop=ystop,
            scale=scale,
            svcs=svcs, X_scalers=X_scalers,
            orient=orient,
            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
            hog_channel=hog_channel,
            spatial_size=spatial_size, hist_bins=hist_bins,
            color_space=color_space,
            extract_spatial_features=extract_spatial_features,
            extract_color_features=extract_color_features,
            extract_hog_features=extract_hog_features,
            draw=False)

        all_detection_boxes.extend(detection_boxes)

    return all_detection_boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        weight = 1.0
        if len(box) == 4:
            weight = box[3]
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_boxes_from_labels(labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        min_x = np.min(nonzerox)
        min_y = np.min(nonzeroy)
        max_x = np.max(nonzerox)
        max_y = np.max(nonzeroy)

        width = abs(max_x - min_x)
        height = abs(max_y - min_y)
        ratio = 0.0

        if (height > 0 and width > 0):
            ratio = width / height

        if (ratio > 0.5 and ratio < 3):
            # Define a bounding box based on min/max x and y
            bbox = ((min_x, min_y), (max_x, max_y))
            boxes.append(bbox)

    return boxes

def draw_labeled_bboxes(img, boxes, color=(0, 0, 1), thick=6):
    if boxes is not None:
        for bbox in boxes:
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img

