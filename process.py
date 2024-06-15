import os
import xml.dom
import numpy as np
import codecs
import json
import glob
import cv2
import shutil
import xml.etree.ElementTree as ET
import os
import os
import random
import shutil
import sys
from ultralytics import YOLO
sys.path.append('./../')
from utils.data_provider import DataProvider

# 修改对应的路径，labelme_path是训练接的路径，saved_path是xml保存路径
saved_path = r"data/Annotations/train/"
dataset_dir= 'exp/datasets'    # 目标文件夹

# train
labelme_path = ["data/frames/spaghetti2/" , "data/frames/clogging1/"]

data_provider = DataProvider(labelme_path=labelme_path, saved_path=saved_path)
data_provider.get_img_file()
img_xmls = os.listdir(saved_path)

for img_xml in img_xmls:
    if img_xml.endswith('xml'):
        label_name = img_xml.split('.')[0]
        data_provider.convert_annotation(image_id=label_name)

# 调用函数分离数据集
data_provider.split_dataset(labelme_path, dataset_dir, 0.6, 0.4, 0)
data_provider.move_files(os.path.join(dataset_dir, 'train'))
data_provider.move_files(os.path.join(dataset_dir, 'val'))

# test
labelme_path = "data/frames/spaghetti1/"

data_provider = DataProvider(labelme_path=labelme_path, saved_path=saved_path)
data_provider.get_img_file()
img_xmls = os.listdir(saved_path)

for img_xml in img_xmls:
    if img_xml.endswith('xml'):
        label_name = img_xml.split('.')[0]
        data_provider.convert_annotation(image_id=label_name)

# 调用函数分离数据集
data_provider.split_dataset(labelme_path, dataset_dir, 0, 0, 1)
data_provider.move_files(os.path.join(dataset_dir, 'test'))
