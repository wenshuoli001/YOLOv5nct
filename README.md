# YOLOv5nct
 Final Project Dissertation: CNN-transformer mixed model for object detection


YOLOv5nct: CNN-transformer mixed model based on yolov5n

The model is improved by adding my proposed Conv-transformer block to YOLOv5n
To see the structure of YOLOv5nct, please check models/yolov5nct.yaml
To see the code of the Conv-Transformer block, check models/common.py

Please install the packages in requirements.txt before running, If you use pycharm, it will be installed automatically.
Installation: first use %cd to locate the directory under the yolov5-master folder, and then type: !pip install -r requirements.txt

If you want to use the YOLOv5nct model with 300 epochs trained on the COCO dataset for target detection, run detect.py
If you want to use the YOLOv5nct model with 100 epochs trained on the Pascal VOC dataset for target detection, run detect1.py

The images to be detected are in the data/images folder, and after running detect.py or detect1.py, the results will be saved in the runs/detect folder
