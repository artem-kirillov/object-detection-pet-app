# Object detection pet app

Result of playing with [https://github.com/tensorflow/models/blob/master/research/object_detection](https://github.com/tensorflow/models/blob/master/research/object_detection).

## How to run
1. Clone [https://github.com/tensorflow/models](https://github.com/tensorflow/models)
2. Set up environment:
    
        export TF_RESEARCH=<path to ./tensorflow/models/research>
        export MODEL_HOME=<path for models downloading>
3. Generate proto interfaces:

        cd "${TF_RESEARCH}"
        protoc object_detection/protos/*.proto --python_out=.
4. Set up `PYTHONPATH`:

        export PYTHONPATH="${TF_RESEARCH}:${PYTHONPATH}"
5. Download frozen model from [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

        cd ${MODEL_HOME}
        curl -o model.tar.gz <model-url>
        tar -xvf model.tar.gz
6. Run example

        python object_detection_pet_app.py \
            --model-path ${MODEL_HOME}/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb \
            --labels-path "${TF_RESEARCH}/object_detection/data/mscoco_label_map.pbtxt"