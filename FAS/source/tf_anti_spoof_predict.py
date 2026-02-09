# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : tf_anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import numpy as np
import tensorflow as tf

from source.data_io import tf_transform as trans
from source.utility import get_kernel, parse_model_name
from source.model.TrFAS import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id=0):
        super(AntiSpoofPredict, self).__init__()
        # Setup GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and device_id < len(gpus):
            try:
                tf.config.set_visible_devices(gpus[device_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[device_id], True)
            except RuntimeError as e:
                print(e)
        self.model = None

    def _load_model(self, model_path):
        """
        Load TensorFlow model from saved format or H5
        """
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
        
        # Check if it's a TensorFlow saved model or H5 file
        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            self.model = tf.keras.models.load_model(model_path)
        elif os.path.isdir(model_path):
            # Load SavedModel format
            self.model = tf.saved_model.load(model_path)
        else:
            # Build model from scratch if you have the architecture
            if model_type in MODEL_MAPPING:
                self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size)
                # Load weights if available
                if os.path.exists(model_path):
                    self.model.load_weights(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return None

    def predict(self, img, model_path):
        """
        Predict anti-spoofing score
        Args:
            img: input image (numpy array)
            model_path: path to TensorFlow model
        Returns:
            result: softmax probabilities
        """
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img = tf.expand_dims(img, axis=0)
        
        self._load_model(model_path)
        
        # Inference
        result = self.model(img, training=False)
        result = tf.nn.softmax(result).numpy()
        
        return result
