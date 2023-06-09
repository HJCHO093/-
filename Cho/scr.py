import cv2
import os
import numpy as np
from pdf2image import convert_from_path
import functions as fs
from PIL import Image
import modules as modules
import matplotlib.pyplot as plt
from xml.etree import ElementTree as elemTree 
import sys

#getting path
print(os.getcwd())
resource_path=os.getcwd()+ "/resource/"          #path
file_name="bluewhale.pdf"                       #file name
pdffile_path=resource_path + file_name          

#converting pdf to jpg
pdfs=convert_from_path(pdffile_path,500)
for i, page in enumerate(pdfs):
    page.save(resource_path + file_name + str(i+1)+ '.jpg')
    
#reading image
i=2
image0=cv2.imread(resource_path + file_name + str(i+1) + '.jpg')
print(image0.shape, "image0")
#removing noise
image=modules.remove_noise(image0)
print(image.shape, "image")
#removing staves
image1, staves=modules.remove_staves(image)
print(image1.shape, "image1")

#normalizing image
image2, staves=modules.normalization(image1, staves, 20)
print(image2.shape, "image2")

#Detecting each objects
image3, objects=modules.object_detection(image2, staves)
print(image3.shape, "image3")
def get_test_img(objects, index):
    bbox = objects[index]
    line, location = bbox[0], bbox[1]
    x = location[0]
    y = location[1]
    w = location[2]
    h = location[3]
    area = location[4]
    separated_image = image3[y:y+h, x:x+w]
    mask_image = np.zeros(separated_image.shape)
    if "zzz" not in os.listdir('./test_data/'):
        os.mkdir('./test_data/' + "zzz")
    path_ = "./test_data/"+"zzz"+'/'
    height, width  = image.shape
    l_height, l_width = image2.shape
    cv2.imwrite(path_ + '0_original.jpg', image0[int(y*height/l_height):int((y+h)*height/l_height), int(width/l_width*x):int(width/l_width*(x+w)), 0])
    cv2.imwrite(path_ +'1_remove_noise.jpg', image[int(y*height/l_height):int((y+h)*height/l_height), int(width/l_width*x):int(width/l_width*(x+w))])
    cv2.imwrite(path_ +'2_remove_staves.jpg', image1[int(y*height/l_height):int((y+h)*height/l_height), int(width/l_width*x):int(width/l_width*(x+w))])
    image2[y-1,x-1:x+w+1] = 125
    image2[y+h+1,x-1:x+w+1] = 125
    image2[y-1:y+h+1,x-1] = 125
    image2[y-1:y+h+1,1+x+w] = 125
    image3[y-1,x-1:x+w+1] = 125
    image3[y+h+1,x-1:x+w+1] = 125
    image3[y-1:y+h+1,x-1] = 125
    image3[y-1:y+h+1,1+x+w] = 125

    cv2.imwrite(path_ +'3_after_normal.jpg', image2[y-40:y+h+40, x-40:x+w+40])
    cv2.imwrite(path_ +'4_after_detec.jpg', image3[y-40:y+h+40, x-40:x+w+40])
input_number = 1
while input_number != 'x':
    input_number = int(input("뽑아보기를 원하는 바운딩박스상의 인덱스 입력하시오 :  (끝내고싶으면 x을 입력하라)"))
    get_test_img(objects,input_number)
    print("done!")
# bbox = objects[0]
# line, location = bbox[0], bbox[1]
# x = location[0]
# y = location[1]
# w = location[2]
# h = location[3]
# area = location[4]
# separated_image = image3[y:y+h, x:x+w]
# mask_image = np.zeros(separated_image.shape)
# plt.imshow(separated_image) 
# plt.savefig("./image_test/test_1.jpg")
# xml = elemTree.parse("./resource/high_in_pdf3.xml")
# root = xml.getroot()
# bndbox = root.find(".//bndbox")
# xmin = int(bndbox.find("./xmin").text)
# xmax = int(bndbox.find("./xmax").text)
# ymin = int(bndbox.find("./ymin").text)
# ymax = int(bndbox.find("./ymax").text)
# cv2.imwrite('./image_test/0_original.jpg', image0[ymin:ymax,xmin:xmax, 0])
# cv2.imwrite('./image_test/1_remove_noise.jpg', image[ymin:ymax,xmin:xmax])
# cv2.imwrite('./image_test/2_remove_staves.jpg', image1[ymin:ymax,xmin:xmax])
# cv2.imwrite('./image_test/3_after_normal.jpg', image2[y:y+h, x:x+w])
# cv2.imwrite('./image_test/4_after_detec.jpg', image3[y:y+h, x:x+w])

#resizing and opening image(unfixed)
"""
cv2.namedWindow('Resized Window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', 1048, 1048)
"""

cv2.imshow('Resized Window', image3)

#press esc to escape
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
