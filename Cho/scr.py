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
pdfs=convert_from_path(pdffile_path)
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
    index = int(index)
    bbox = objects[index]
    line, location = bbox[0], bbox[1]
    x = location[0]
    y = location[1]
    w = location[2]
    h = location[3]
    area = location[4]
    separated_image = image3[y:y+h, x:x+w]
    mask_image = np.zeros(separated_image.shape)
    if str(index) not in os.listdir('./test_data/'):
        os.mkdir('./test_data/' + str(index))
    path_ = "./test_data/"+str(index)+'/'
    cv2.imwrite(path_ + '0_original.jpg', image0[int(y*1654/985):int((y+h)*1654/985), int(2339/1394*x):int(2339/1394*(x+w)), 0])
    cv2.imwrite(path_ +'1_remove_noise.jpg', image[int(y*1654/985):int((y+h)*1654/985), int(2339/121394*x):int(2339/1394*(x+w))])
    cv2.imwrite(path_ +'2_remove_staves.jpg', image1[int(y*1654/985):int((y+h)*1654/985), int(2339/1394*x):int(2339/1394*(x+w))])
    cv2.imwrite(path_ +'3_after_normal.jpg', image2[y:y+h, x:x+w])
    cv2.imwrite(path_ +'4_after_detec.jpg', image3[y:y+h, x:x+w])    
input_number = str(input("뽑아보기를 원하는 바운딩박스상의 인덱스 입력하시오 : "))
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
