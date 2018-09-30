import numpy as np
import importlib
import ObjectDetection as od
import cv2
importlib.reload(od)


class Detection:
    def __init__(self, bounding_box, heat_map):
        self.bounding_box = bounding_box
        self.heat_map = heat_map[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]]
        self.heat_center = self.get_center(bounding_box)
        self.probable_move_x = 0.0
        self.probable_move_y = 0.0
        self.belief = 0.0
        self.age = 0
        self.last_updated = 0;

    def older(self):
        self.age += 1
        self.heat_map -= 0.4
        self.heat_map = np.clip(self.heat_map, 0.0, 5.0)

        alive = False

        if np.count_nonzero(self.heat_map) > 50:
            alive = True

        return alive

    # get the heat center of a detection
    def get_center(self, bounding_box):
        # get hottest value
        hottest =  np.max(self.heat_map)
        hottest_indexes = (self.heat_map == hottest).nonzero()
        hottest_x_indexes = np.array(hottest_indexes[1])
        hottest_y_indexes = np.array(hottest_indexes[0])

        min_x = np.min(hottest_x_indexes)
        min_y = np.min(hottest_y_indexes)
        max_x = np.max(hottest_x_indexes)
        max_y = np.max(hottest_y_indexes)

        width = abs(max_x - min_x)
        height = abs(max_y - min_y)

        ((bb_min_x, bb_min_y), (bb_max_x, bb_max_y)) = bounding_box

        return [bb_min_x + min_x + width / 2, bb_min_y + min_y + height / 2]


    def add_to_heat_map(self, heat_map):
        heat_map[
            self.bounding_box[0][1]:self.bounding_box[1][1],
            self.bounding_box[0][0]:self.bounding_box[1][0]] += self.heat_map


class Trail:
    def __init__(self):
        self.estimated_detection = None
        self.current_detection = None
        self.distance = 0

    def get_distance(self, a, b):
        a_diff = a[0] - b[0]
        b_diff = a[1] - b[1]
        return np.sqrt(a_diff**2 + b_diff**2)

    def find_probable_trail(self, prediction, current_detections, max_distance):
        min_distance = max_distance

        ((bb_min_x, bb_min_y), (bb_max_x, bb_max_y)) = prediction.bounding_box

        width = bb_max_x - bb_min_x
        height = bb_max_y - bb_min_y

        if width > height and width < max_distance:
            min_distance = width  * 0.3
        elif width <= height and height < max_distance:
            min_distance = height * 0.3

        found = False
        size_variance = 0.8

        for current_detection in current_detections:
            distance = self.get_distance(current_detection.heat_center, prediction.heat_center)

            ((cd_bb_min_x, cd_bb_min_y), (cd_bb_max_x, cd_bb_max_y)) = current_detection.bounding_box

            cd_width = cd_bb_max_x - cd_bb_min_x
            cd_height = cd_bb_max_y - cd_bb_min_y

            if distance < min_distance and (abs(cd_width - width) / width) < size_variance and (abs(cd_height - height) / height) < size_variance:
                self.current_detection = current_detection
                self.estimated_detection = prediction
                self.distance = distance
                min_distance = distance
                found = True

        return found


class ObjectTracking:
    def __init__(self):
        self.last_detections = None
        self.current_detections = None
        self.predictions = None
        self.heat_map = None
        self.all_detections = None


    def set_current_detections(self, detections, heat_map):
        self.current_detections = []

        for detection in detections:
            detection_object = Detection(detection, heat_map)
            self.current_detections.append(detection_object)

    def get_boxes(self, heat_map):
        heat_map = np.clip(heat_map, 0, 5.0)
        heat_map = od.apply_threshold(heat_map, 0.5)

        # Find final boxes from heatmap using label function
        labels = od.label(heat_map)
        boxes = od.get_boxes_from_labels(labels)
        return boxes

    def predict_update(self, heat_map):
        updated_detections = []

        if self.last_detections is None or len(self.last_detections) == 0:
            # first prediction
            # there is no state that may lead to a
            # meaningfull prediction
            self.predictions = None
            updated_detections = self.current_detections

        else:
            search_radius = 100

            self.predictions = self.last_detections
            self.last_detections = []

            for prediction in self.predictions:
                trail = Trail()
                # try to find pairs of previous and current detections
                found = trail.find_probable_trail(prediction, self.current_detections, search_radius)

                # make predictions older
                prediction.older()

                if found == True:
                    new_heat_map = np.zeros_like(heat_map)
                    shift_x = trail.current_detection.heat_center[0] - prediction.heat_center[0]
                    shift_y = trail.current_detection.heat_center[1] - prediction.heat_center[1]

                    # prediction.probable_move_x = prediction.probable_move_x * 0.7 + shift_x * 0.3
                    # prediction.probable_move_y = prediction.probable_move_y * 0.7 + shift_y * 0.3


                    # combine pervious heat map with heat map from current detection
                    prediction.add_to_heat_map(new_heat_map)
                    trail.current_detection.add_to_heat_map(new_heat_map)
                    new_heat_map = np.clip(new_heat_map, 0, 5.0)
                    new_boxes = self.get_boxes(new_heat_map)

                    # store new detections
                    for detection in new_boxes:
                        detection_object = Detection(detection, new_heat_map)
                        detection_object.age = prediction.age
                        detection_object.last_updated = 0
                        updated_detections.append(detection_object)

                else:
                    prediction.probable_move_x = prediction.probable_move_x * 0.8
                    prediction.probable_move_y = prediction.probable_move_y * 0.8
                    prediction.last_updated += 1
                    if (prediction.last_updated < 10):
                        updated_detections.append(prediction)


        return updated_detections



    def track(self, detections, heat_map):
        self.set_current_detections(detections, heat_map)
        updated_detections = self.predict_update(heat_map)

        self.last_detections = updated_detections

        self.heat_map = np.zeros_like(heat_map)
        aged_detections = []
        for detection in self.last_detections:
            detection.add_to_heat_map(self.heat_map)
            if detection.age > 5:
                aged_detections.append(detection.bounding_box)

        return aged_detections



        '''
        # try to find pairs of previous and current detections

        heat_map_width = len(heat_map[0])
        heat_map_height = len(heat_map)

        if self.predictions is not None:


                shift_x = int(prediction.probable_move_x)
                shift_y = int(prediction.probable_move_y)

                print('------------------')
                print('------------------')
                print(trail.current_detection.heat_center)
                print(prediction.heat_center)
                print('------------------')
                print(prediction.bounding_box[0][0])
                print(prediction.bounding_box[0][1])
                print(prediction.bounding_box[1][0])
                print(prediction.bounding_box[1][1])
                print(prediction.bounding_box)

                if (prediction.bounding_box[0][0] + shift_x) < 0:
                    shift_x = prediction.bounding_box[0][0]

                if (prediction.bounding_box[1][0] + shift_x) >= heat_map_width:
                    shift_x = heat_map_width - prediction.bounding_box[1][0] - 1

                if (prediction.bounding_box[0][1] + shift_y) < 0:
                    shift_y = prediction.bounding_box[0][1]

                if (prediction.bounding_box[1][1] + shift_y) >= heat_map_height:
                    shift_y = heat_map_height - prediction.bounding_box[1][1] - 1

                prediction.bounding_box = ((
                    prediction.bounding_box[0][0] + shift_x,
                    prediction.bounding_box[0][1] + shift_y),(
                    prediction.bounding_box[1][0] + shift_x,
                    prediction.bounding_box[1][1] + shift_y))

                prediction.heat_center[0] += shift_x
                prediction.heat_center[1] += shift_y

                print('------------------')
                print(prediction.bounding_box)
                print(prediction.bounding_box[0][0])
                print(prediction.bounding_box[0][1])
                print(prediction.bounding_box[1][0])
                print(prediction.bounding_box[1][1])

                # age predictions
                prediction.older()
                prediction.add_to_heat_map(new_heat_map)

        else:
            self.heat_map = heat_map
            self.last_detections = self.current_detections.copy()
            return []


        # add the heat meap from current detection
        for current_detection in self.current_detections:
            current_detection.add_to_heat_map(new_heat_map)

        new_heat_map = np.clip(new_heat_map, 0, 255)
        new_heat_map = od.apply_threshold(new_heat_map, 3)

        # Find final boxes from heatmap using label function
        labels = od.label(new_heat_map)
        self.heat_map = new_heat_map
        boxes = od.get_boxes_from_labels(labels)
        self.all_detections = boxes

        # store new detections
        self.set_current_detections(boxes, new_heat_map)

        # try to find trails from old predictions to new detections
        # to keep track of parameters
        for trail in trails:
            new_trail = Trail()
            prediction = trail.estimated_detection
            found = new_trail.find_probable_trail(prediction, self.last_detections, self.current_detections, search_radius)

            if found == True:
                new_trail.current_detection.age = prediction.age
                new_trail.current_detection.probable_move_x = prediction.probable_move_x
                new_trail.current_detection.probable_move_y = prediction.probable_move_y

        self.last_detections = self.current_detections.copy()

        aged_detections = []
        for detection in self.last_detections:
            if detection.age > 5:
                aged_detections.append(detection.bounding_box)

        return aged_detections
        '''

def car_detection_pipeline(
        image, ystart, detection_window_size, scale_min, scale_max, steps,
        svcs, X_scalers, orient, pix_per_cell, cell_per_block, hog_channel,
        spatial_size, hist_bins,
        color_space, extract_spatial_features, extract_color_features, extract_hog_features,
        object_tracking):
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32) / 255

    heat_map = np.zeros_like(image[:, :, 0]).astype(np.float)

    all_detection_boxes = od.find_cars(
        image=image,
        ystart=ystart, detection_window_size=detection_window_size,
        scale_min=scale_min, scale_max=scale_max, steps=steps,
        svcs=svcs, X_scalers=X_scalers,
        orient=orient,
        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_size=spatial_size, hist_bins=hist_bins,
        color_space=color_space,
        extract_spatial_features=extract_spatial_features,
        extract_color_features=extract_color_features,
        extract_hog_features=extract_hog_features)

    # Add heat to each box in box list
    heat_map = od.add_heat(heat_map, all_detection_boxes)

    # Apply threshold to help remove false positives
    heat_map = od.apply_threshold(heat_map, 0.5)
    heat_map_clipped = np.clip(heat_map, 0, 5.0)

    # Find final boxes from heatmap using label function
    labels = od.label(heat_map_clipped)
    boxes = od.get_boxes_from_labels(labels)
    boxes = object_tracking.track(boxes, heat_map_clipped)
    draw_img = od.draw_labeled_bboxes(np.copy(image), boxes)
    draw_img *= 255

    '''
    draw_img = np.zeros_like(image)
    draw_img[:, :, 0] = object_tracking.heat_map * 40.0
    '''

    return draw_img
