"""
################################################
## This Python script uses TensorFlow and OpenCV
## to detect objects in images, specifically looking
## for apples among a predefined list of image files
################################################
##
## pip install tensorflow
## pip install numpy opencv-python pillow
##
## cd models
## mkdir tensorflow_models
## cd tensorflow_models
## git clone https://github.com/tensorflow/models.git
##
## Find mscoco_label_map.pbtxt
## models/research/object_detection/data/mscoco_label_map.pbtxt
## /home/nuc8/05_development/02_python/AI/hf_env_wpl/models/tensorflow_models/models/research/object_detection/data/mscoco_label_map.pbtxt
##
## Install the Object Detection API
## Navigate into the cloned models/research/ directory and install
## the Object Detection API:
## e.g. cd /home/nuc8/05_development/02_python/AI/hf_env_wpl/tensorflow_models/models/research
## Compile protos.
## protoc object_detection/protos/*.proto --python_out=.
## Install TensorFlow Object Detection API.
## cp object_detection/packages/tf2/setup.py .
## python -m pip install .
##
## Download a Pre-trained Model
## https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
##
## model: SSD MobileNet V2 FPNLite 640x640
## 
## wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
## tar -zxvf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
##
##
################################################
"""

import cv2  # OpenCV library for computer vision tasks, such as reading and processing images
import numpy as np  # A library for numerical operations, particularly useful for handling arrays
import os
import platform
import tensorflow as tf  # tensorflow (tf): The core library for running TensorFlow models

# object_detection.utils: Functions from the TensorFlow Object Detection API for handling label maps and visualization
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as viz_utils


def clear_screen():
    try:
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")
    except Exception as e:
        print(f"Error clearing screen: {e}")


APPLE_FILES = [
    "apple_1.jpg",
    "apple_2.jpg",
    "apple_3.png",
    "pear_4.jpg",
    "car_5.jpg",
    "hammer_6.jpg",
    "garden_7.jpg",
]

if __name__ == "__main__":
    clear_screen()

    # Loading the Object Detection Model
    # loads a pre-trained TensorFlow object detection model from (model_dir,
    # this model is capable of detecting various objects in images, including apples
    model_dir = "/home/nuc8/05_development/02_python/AI/hf_env_wpl/models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model"

    try:
        detect_fn = tf.saved_model.load(model_dir)
    except Exception as e:
        print(f"Failed to load model from {model_dir} with error: {e}")
        exit(1)

    # Loading the Label Map:
    # label map (label_map_path) is loaded to map the detected object IDs to human-readable labels,
    # for instance, the label map would indicate which ID corresponds to an apple
    # path to label map
    label_map_path = "/home/nuc8/05_development/02_python/AI/hf_env_wpl/models/tensorflow_models/models/research/object_detection/data/mscoco_label_map.pbtxt"

    try:
        category_index = label_map_util.create_category_index_from_labelmap(
            label_map_path, use_display_name=True
        )
    except Exception as e:
        print(f"Failed to load label map from {label_map_path} with error: {e}")
        exit(1)

    # Image Processing Loop:
    # Iterate through apple files for detection
    for image_name in APPLE_FILES:
        try:
            image_path = image_name  # Assuming the images are in the current directory
            # Image Reading: Each image is read into a NumPy array using OpenCV
            image_np = cv2.imread(image_path)

            if image_np is None:
                raise FileNotFoundError(f"Error loading image {image_name}")

            # Ensure the image is in uint8 before converting to a tensor
            image_np = np.array(image_np, dtype=np.uint8)
            # Convert the image to a tensor, ensuring it's correctly defined within this block
            # the image is converted into a tensor, a format required by TensorFlow models
            input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)

            # Image loading and preprocessing as before
            detections = detect_fn(input_tensor)

            # The loaded TensorFlow model is used to detect objects in the image,
            # the detection results include various details, such as the classes of detected
            # objects and their locations
            num_detections = int(detections.pop("num_detections"))
            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections["num_detections"] = num_detections

            # Convert detection classes from float to int, and count 'apple' detections
            detection_classes = detections["detection_classes"].astype(int)
            apple_detections = sum(
                detection == 53 for detection in detection_classes
            )  # 53 is the class ID for 'apple'
            # looks for detections with the class ID corresponding to apples (ID 53 in this case),
            # it then prints a message indicating whether an apple was detected in each image.

            # Visualization of the results of a detection as before

            # Check if any apples were detected and print appropriate message
            if apple_detections > 0:
                print(f"Apple detected in     ---> {image_name} <---")
            else:
                print(f"Apple not detected in ---> {image_name} <---")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")

    cv2.destroyAllWindows()
    print("Processing complete.")
