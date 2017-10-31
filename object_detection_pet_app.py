import collections
import time
from threading import Thread

import cv2
import numpy as np
import six
import tensorflow as tf
import struct
import logging
import argparse

from object_detection.utils import label_map_util

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)


class WebCamStream(object):
    def __init__(self, src=0, resolution=(480, 360)):
        self.src = src
        self.resolution = resolution
        self._frame = None
        self._stopped = False

    def __enter__(self):
        self.stream = cv2.VideoCapture()
        self.stream.open(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        self.thread = Thread(target=self._run, args=())
        self.thread.start()
        self.stopped = False
        while self.frame() is None:
            time.sleep(1)

        return self

    def _run(self):
        self._stopped = False
        while not self._stopped:
            ret, self._frame = self.stream.read()

    def frame(self):
        return self._frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stopped = True
        while self.thread.is_alive():
            time.sleep(1)
        self.stream.release()
        cv2.destroyAllWindows()


class ObjectDetector(object):
    NUM_CLASSES = 90

    def __init__(self, model_path, labels_path):
        self._model = self._load_model(model_path)
        self._session = tf.Session(graph=self._model)
        self._category_index = self._load_cat_index(labels_path)

        # Each box represents a part of the image where a particular object was detected.
        self._boxes = self._model.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self._scores = self._model.get_tensor_by_name('detection_scores:0')
        self._classes = self._model.get_tensor_by_name('detection_classes:0')
        self._num_detections = self._model.get_tensor_by_name('num_detections:0')
        self._image_tensor = self._model.get_tensor_by_name('image_tensor:0')

    @classmethod
    def _load_cat_index(cls, labels_path):
        label_map = label_map_util.load_labelmap(labels_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=ObjectDetector.NUM_CLASSES, use_display_name=True)
        return label_map_util.create_category_index(categories)

    @classmethod
    def _load_model(cls, model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def detect(self, frame):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)

        # Actual detection.
        boxes, scores, classes, num_detections = self._session.run(
            [self._boxes, self._scores, self._classes, self._num_detections],
            feed_dict={self._image_tensor: image_np_expanded})

        return boxes, scores, classes

    def category_index(self):
        return self._category_index


class BoxRenderer(object):
    def __init__(self, resolution=(480, 360), max_boxes_to_draw=20, min_score_threshold=.5):
        self._resolution = resolution
        self._max_boxes_to_draw = max_boxes_to_draw
        self._min_score_threshold = min_score_threshold
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame, boxes, classes, scores, category_index):
        rect_points, class_names, class_colors = self._boxes_with_labels(
            np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index)

        width = self._resolution[0]
        height = self._resolution[1]
        for point, name, color in zip(rect_points, class_names, class_colors):
            x_min = int(point['xmin'] * width)
            x_max = int(point['xmax'] * width)
            y_min = int(point['ymin'] * height)
            y_max = int(point['ymax'] * height)
            cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color, thickness=3)

            cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(x_min + len(name[0]) * 6, y_min - 10),
                          color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(frame, text=name[0], org=(x_min, y_min),
                        fontFace=self._font, fontScale=0.3, color=(0, 0, 0), thickness=1)
        return frame

    def _boxes_with_labels(self, boxes, classes, scores, category_index):
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        for i in range(min(self._max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > self._min_score_threshold:
                box = tuple(boxes[i].tolist())
                if scores is None:
                    box_to_color_map[box] = 'black'
                else:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))
                    box_to_display_str_map[box].append(display_str)
                    box_to_color_map[box] = 'EE7600'

        # Store all the coordinates of the boxes, class names and colors
        rect_points = []
        class_names = []
        class_colors = []
        for box, color in six.iteritems(box_to_color_map):
            ymin, xmin, ymax, xmax = box
            rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
            class_names.append(box_to_display_str_map[box])
            class_colors.append(struct.unpack('BBB', bytes.fromhex('EE7600')))
        return rect_points, class_names, class_colors


def main():
    renderer = BoxRenderer(resolution=(1024, 768))
    detector = ObjectDetector(
        model_path='ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb',
        # model_path='ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb',
        labels_path='object_detection/data/mscoco_label_map.pbtxt')

    with WebCamStream(1, resolution=(1024, 768)) as stream:
        while True:
            frame = stream.frame()
            boxes, scores, classes = detector.detect(frame)
            rendered_frame = renderer.render(frame, boxes, classes, scores, detector.category_index())
            cv2.imshow('Video', rendered_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break


if __name__ == '__main__':
    main()
