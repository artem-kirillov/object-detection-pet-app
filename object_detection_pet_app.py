import argparse
import collections
import logging
import time

import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)

Box = collections.namedtuple('Box', ['y_min', 'x_min', 'y_max', 'x_max'])
Object = collections.namedtuple('Object', ['box', 'name', 'score'])


class FPS(object):
    def __init__(self):
        self._start = None
        self._stop = None
        self._num_frames = 0

    def start(self):
        self._start = time.time()

    def is_started(self):
        return self._start is not None

    def update(self):
        self._num_frames += 1

    def stop(self):
        self._stop = time.time()

    def get_fps(self):
        return int(round(self._num_frames / (time.time() - self._start)))


class WebCamStream(object):
    def __init__(self, src=0, resolution=(480, 360)):
        self.src = src
        self.resolution = resolution

    def __enter__(self):
        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        return self

    def capture_frame(self):
        ret, frame = self.stream.read()
        frame = cv2.flip(frame, flipCode=1)
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.release()
        cv2.destroyAllWindows()


class ObjectDetector(object):
    NUM_CLASSES = 90

    def __init__(self, model_path, labels_path):
        self._model_path = model_path
        self._labels_path = labels_path

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

    def __enter__(self):
        self._model = self._load_model(self._model_path)
        self._session = tf.Session(graph=self._model)
        self._category_index = self._load_cat_index(self._labels_path)

        # Each box represents a part of the image where a particular object was detected.
        self._boxes = self._model.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self._scores = self._model.get_tensor_by_name('detection_scores:0')
        self._classes = self._model.get_tensor_by_name('detection_classes:0')
        self._num_detections = self._model.get_tensor_by_name('num_detections:0')
        self._image_tensor = self._model.get_tensor_by_name('image_tensor:0')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()

    def recognize(self, frame):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)

        # Actual detection.
        boxes, scores, classes, num_detections = self._session.run(
            [self._boxes, self._scores, self._classes, self._num_detections],
            feed_dict={self._image_tensor: image_np_expanded})

        return boxes, scores, classes

    def category_index(self):
        return self._category_index


class BoxDrawing(object):
    def __init__(self, resolution=(480, 360), max_boxes_to_draw=20, min_score_threshold=.5):
        self._resolution = resolution
        self._max_boxes_to_draw = max_boxes_to_draw
        self._min_score_threshold = min_score_threshold
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame, boxes, classes, scores, category_index):
        objects = self._boxes_with_labels(
            np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index)

        width = self._resolution[0]
        height = self._resolution[1]
        for (box, name, color) in objects:
            x_min = int(box.x_min * width)
            x_max = int(box.x_max * width)
            y_min = int(box.y_min * height)
            y_max = int(box.y_max * height)
            cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(x_max, y_max), color=color, thickness=3)

            cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(x_min + len(name) * 6, y_min - 10),
                          color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(frame, text=name, org=(x_min, y_min),
                        fontFace=self._font, fontScale=0.3, color=(0, 0, 0), thickness=1)
        return frame

    def _boxes_with_labels(self, boxes, classes, scores, category_index):
        objects = []
        for i in range(min(self._max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > self._min_score_threshold:
                box = Box(*boxes[i])
                color = (0, 0, 0)
                name = 'N/A'
                if scores is not None:
                    class_name = category_index.get(classes[i], {'name': 'N/A'})['name']
                    name = '{}: {}%'.format(class_name, int(100 * scores[i]))
                    color = (0, 118, 238)
                objects.append(Object(box, name, color))
        return objects


class Renderer(object):
    def __init__(self, window_name='Video'):
        self._window_name = window_name
        self._fps_meter = FPS()
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame):
        if not self._fps_meter.is_started():
            self._fps_meter.start()
        cv2.putText(frame, text='FPS={}f/s'.format(self._fps_meter.get_fps()), org=(6, 10), fontFace=self._font,
                    fontScale=0.3, color=(0, 0, 0), thickness=1)
        cv2.imshow(self._window_name, frame)
        self._fps_meter.update()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fps_meter.stop()
        cv2.destroyAllWindows()


def loop(stream, detector, drawing, renderer):
    logging.info('Starting main loop...')
    while True:
        frame = stream.capture_frame()
        boxes, scores, classes = detector.recognize(frame)
        rendered_frame = drawing.render(frame, boxes, classes, scores, detector.category_index())
        renderer.render(rendered_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-source', type=int, default=0, help='Video device ID')
    parser.add_argument('--resolution', type=lambda x: tuple(x.split('x')), default=(480, 360),
                        help='Input / output video resolution')
    parser.add_argument('--model-path', type=str, required=True, help='TF model path')
    parser.add_argument('--labels-path', type=str, required=True, help='Model labels path')

    return parser.parse_args()


def main():
    args = parse_args()
    drawing = BoxDrawing(args.resolution)

    with WebCamStream(args.video_source, args.resolution) as stream:
        with ObjectDetector(args.model_path, args.labels_path) as detector:
            with Renderer() as renderer:
                loop(stream, detector, drawing, renderer)


if __name__ == '__main__':
    main()
