import pandas as pd
import glob
import os
from xml.etree import ElementTree as elemTree
import pickle
from keras import *
from keras.layers import *
import cv2
import numpy as np
def color_inversion(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = np.where(img>127, 0, 255)
    img_file_name = filename[filename.rfind("\\")+1:]
    return img_file_name, img
def extract_boxes(filename):
        ## 바운딩박스 정보가 들어있는 xml파일에서 박스 정보 추출.
        ## 추가로 사진 규격도 추출
        tree = elemTree.parse(filename)
        root = tree.getroot()
        
        boxes = []
        # 사진규격 추출
        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        # 클래스이름
        name = root.find(".//name").text
        # 바운딩박스 정보 추출
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        return boxes , width, height, name

xmls = glob.glob("-/resource/scoreImg/*.xml")
lis = []
input_data = []
target_data = []
for xml in xmls:
    boxes , width, height,name= extract_boxes(xml)
    xml = xml[xml.rfind("\\")+1:xml.rfind(".")]+".png"
    for box in boxes:
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        # dic =  {"filename":xml,"width":width, "height":height,"name": name ,  
            # "xmin": xmin ,"ymin":ymin,"xmax": xmax,"ymax":ymax}
        lis.append([xml,width,height,name,xmin,ymin,xmax,ymax])
        input_data.append(extract_boxes(xml))
input_data = np.asarray(input_data, dtype=np.int8)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(max_shape[0], max_shape[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer='adam',
              loss="spar",
              metrics=['accuracy'])