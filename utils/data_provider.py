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
import xml.dom.minidom as xmldom
import os

class DataProvider:
    def __init__(self, labelme_path, saved_path):
       
        self.labelme_path = labelme_path
        if isinstance(self.labelme_path, str):
            self.labelme_path = [self.labelme_path]
        self.saved_path = saved_path
        self.classes = ['normal', 'spaghetti', 'possible-spaghetti']  # 类别
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

    def get_img_file(self):
        files = []
        if isinstance(self.labelme_path, list):
            for path in self.labelme_path:
                files.extend(glob.glob(f"{path}/*.json"))
        else:
            files = glob.glob(f"{self.labelme_path}/*.json")

        # 3.读取标注信息并写入 xml
        f_cnt = 0
        for json_filename in sorted(files[:]):
            f_cnt += 1
            json_file = json.load(open(json_filename, "r", encoding="utf-8"))
            i = 0
            # 图像名字，若图像格式不是bmp，需要修改此处
            img_name = json_filename.replace(".json", ".jpg")
            
            height, width, channels = cv2.imread(img_name).shape
            # xml名字
            xmlName = os.path.join(self.saved_path, "_".join(json_filename.split("/")[-2:]).replace(".json", ".xml"))
          
            if f_cnt%20==0:
                print(f"""写入 xmlName 名称：{json_file},\n {xmlName}""")

            with codecs.open(xmlName, "w", "utf-8") as xml:
                # print(2)
                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'jpg' + '</folder>\n')
                xml.write('\t<filename>' + img_name + '</filename>\n')
                # -------------------------------------------------
                xml.write('\t<source>\n')
                xml.write('\t\t<database>hulan</database>\n')
        
                # --------------------------------------------------
                xml.write('\t</source>\n')
                # -----------------------------------------------------------
                xml.write('\t<size>\n')
                xml.write('\t\t<width>' + str(width) + '</width>\n')
                xml.write('\t\t<height>' + str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
                # ------------------------------------------------
                xml.write('\t</size>\n')
                xml.write('\t\t<segmented>0</segmented>\n')
        
                # 节点判断
                for multi in json_file["shapes"]:
                    points = np.array(multi["points"])
                    xmin = min(points[:, 0])
                    xmax = max(points[:, 0])
                    ymin = min(points[:, 1])
                    ymax = max(points[:, 1])
                    label = multi["label"]
                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>' + json_file["shapes"][i]["label"] + '</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>0</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')
                        # print(json_filename, xmin, ymin, xmax, ymax, label)
                        i = i + 1
                xml.write('</annotation>')
        #annotation_path = saved_path #需要根据实际路径更改

        annotation_names = sorted([os.path.join(self.saved_path, i) for i in os.listdir(self.saved_path) if 'xml' in i])
        f_cnt = 0
        labels = list()
        for names in annotation_names:
            f_cnt+=1
            xmlfilepath = names
            if f_cnt%20==0:
                print(f"xmlfilepath: {xmlfilepath}")
            domobj = xmldom.parse(xmlfilepath)
            # 得到元素对象
            elementobj = domobj.documentElement
            # 获得子标签
            subElementObj = elementobj.getElementsByTagName("object")
            for s in subElementObj:
                label = s.getElementsByTagName("name")[0].firstChild.data
                # print(label)
                if label not in labels:
                    labels.append(label)

        print(f'labels: {labels}')
        
    def convert(self,size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0    # (x_min + x_max) / 2.0
        y = (box[2] + box[3]) / 2.0    # (y_min + y_max) / 2.0
        w = box[1] - box[0]   # x_max - x_min
        h = box[3] - box[2]   # y_max - y_min
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
    
    def convert_annotation(self, image_id):
        in_file = open(f'{self.saved_path}/{image_id}.xml', encoding='UTF-8')
        out_file = open(f'{self.saved_path}/{image_id}.txt', 'w')  # 生成txt格式文件
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    
        for obj in root.iter('object'):
            cls = obj.find('name').text
            # print(cls)
            if cls not in self.classes:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def get_dir(self, file):
        file_name = "_".join(file.split("/")[-2:])
        frame_file = file.split("/")[-1]
        dir = file.split("/")[-2]
        for _dir in self.labelme_path:
            if dir in _dir:
                dir = _dir
                break
        return file_name, os.path.join(dir, frame_file)

    def split_dataset(self, data_dir, train_val_test_dir, train_ratio, val_ratio, test_ratio):
        # 创建目标文件夹
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        train_dir = os.path.join(train_val_test_dir, 'train')
        val_dir = os.path.join(train_val_test_dir, 'val')
        test_dir = os.path.join(train_val_test_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
        # 获取数据集中的所有文件
        files = []
        if isinstance(data_dir, list):
            for _dir in data_dir:
                files.extend(
                    [
                        os.path.join(
                            _dir, __dir)
                        for __dir in os.listdir(_dir)
                    ]
                )
        else:
            files = os.listdir(data_dir)
    
        # 筛选图片文件
        image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

        # 随机打乱文件列表
        random.shuffle(image_files)
    
        # 计算切分数据集的索引
        num_files = len(image_files)
        num_train = int(num_files * train_ratio)
        num_val = int(num_files * val_ratio)
        num_test = int(num_files * test_ratio)
    
        # 分离训练集
        train_files = image_files[:num_train]

        for file in train_files:
            file_name, image_name = self.get_dir(file)
            src_image_path = image_name
            src_label_path = os.path.join(self.saved_path, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            dst_image_path = os.path.join(train_dir, file_name)
            dst_label_path = os.path.join(train_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)
    
        # 分离验证集
        val_files = image_files[num_train:num_train+num_val]
        for file in val_files:
            file_name, image_name = self.get_dir(file)
            src_image_path = image_name
            src_label_path = os.path.join(self.saved_path, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            dst_image_path = os.path.join(val_dir, file_name)
            dst_label_path = os.path.join(val_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)
    
        # 分离测试集
        test_files = image_files[num_train+num_val:]
        for file in test_files:
            file_name, image_name = self.get_dir(file)
            src_image_path = image_name
            src_label_path = os.path.join(self.saved_path, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            dst_image_path = os.path.join(test_dir, file_name)
            dst_label_path = os.path.join(test_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)
    
        print("数据集分离完成！")
        print(f"训练集数量：{len(train_files)}")
        print(f"验证集数量：{len(val_files)}")
        print(f"测试集数量：{len(test_files)}")
 
    def move_files(self,data_dir):
        #data_dir = train_val_test_dir
        # 创建目标文件夹
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
    
        # 获取数据集中的所有文件
        files = os.listdir(data_dir)
    
        # 移动PNG文件到images文件夹
        png_files = [f for f in files if f.endswith('.jpg')]
        for file in png_files:
            src_path = os.path.join(data_dir, file)
            dst_path = os.path.join(images_dir, file)
            shutil.move(src_path, dst_path)
    
        # 移动TXT文件到labels文件夹
        txt_files = [f for f in files if f.endswith('.txt')]
        for file in txt_files:
            src_path = os.path.join(data_dir, file)
            dst_path = os.path.join(labels_dir, file)
            shutil.move(src_path, dst_path)
    
        print(f"{data_dir}文件移动完成！")
        print(f"总共移动了 {len(png_files)} 个PNG文件到images文件夹")
        print(f"总共移动了 {len(txt_files)} 个TXT文件到labels文件夹")
    
 



