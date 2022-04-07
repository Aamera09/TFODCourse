import os
import tensorflow as tf
import tf_slim as slim
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt

class DetectionAPI:
    def __init__(self):
        self.category_index = label_map_util.create_category_index_from_labelmap(r'C:\FYPimp\TFODCourse\Tensorflow\workspace\annotations\label_map.pbtxt')
        configs = config_util.get_configs_from_pipeline_file(r'C:\FYPimp\TFODCourse\Tensorflow\workspace\Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config')
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        
   

    # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(r'C:\FYPimp\TFODCourse\Tensorflow\workspace\Tensorflow\workspace\models\my_ssd_mobnet\ckpt-21.index').expect_partial()

    def detect_fn(self,image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections
    def detect(self,image_path):
        img = cv2.imread(image_path)
        image_np = np.array(img)
        image_name=''

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        #print(detections['detection_classes'].astype(np.int64))

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
                    
        print(image_np_with_detections)
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()




obj= DetectionAPI()
obj.detect(r"C:\FYPimp\TFODCourse\Tensorflow\workspace\images\test\Egg Shell Images_ Stock Photos....jpg")