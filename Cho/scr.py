import cv2
import os
import numpy as np
from pdf2image import convert_from_path
import functions as fs
from PIL import Image
import modules as modules
import matplotlib.pyplot as plt
#getting path
resource_path=os.getcwd()+"/resource/"          #path
file_name="bluewhale.pdf"                       #file name
pdffile_path=resource_path + file_name          

#converting pdf to jpg
pdfs=convert_from_path(pdffile_path)
for i, page in enumerate(pdfs):
    page.save(resource_path + file_name + str(i+1)+ '.jpg')
    
#reading image
i=2
image0=cv2.imread(resource_path + file_name + str(i+1) + '.jpg')

#removing noise
image=modules.remove_noise(image0)

#removing staves
image1, staves=modules.remove_staves(image)

#normalizing image
image2, staves=modules.normalization(image1, staves, 20)

#Detecting each objects
image3, objects=modules.object_detection(image2, staves)
print(objects)
print(image3)
for i, bbox in enumerate(objects):
    line, location = bbox[0], bbox[1]
    x = location[0]
    y = location[1]
    w = location[2]
    h = location[3]
    area = location[4]
    separated_image = image3[y-20:y+h+20, x-20:x+w+20]
    mask_image = separated_image
    mask_image[y:y+h, x:x+w] = 255
    plt.imshow(separated_image) 
    plt.imshow(mask_image, alpha=0.5)
    plt.savefig("./image_test/test_%s.jpg"%i)
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
