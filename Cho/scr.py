import cv2
import os
import numpy as np
from pdf2image import convert_from_path
import functions as fs
from PIL import Image
import modules as modules
import matplotlib.pyplot as plt
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
    separated_image = image3[y:y+h, x:x+w]
    mask_image = np.zeros(separated_image.shape)
    
    # mask_image[20:h+20, 20:w+20]=1
    print(mask_image.shape)
    print(mask_image)

    print(image3.shape, separated_image.shape, mask_image.shape)
    plt.imshow(separated_image) 
    # plt.imshow(mask_image, cmap="gray", alpha=0.5) 
    # plt.imshow(mask_image, cmap='gray',alpha=0.5)
    plt.savefig("./image_test/test_%s.jpg"%i)

    
    high_img = cv2.imread("./resource/high.png", cv2.IMREAD_GRAYSCALE)
    print(high_img)
    new_img = cv2.resize(high_img, dsize=(separated_image.shape[1], separated_image.shape[0]),interpolation = cv2.INTER_AREA)
    new_img = np.where(new_img>100, 5 ,255)
    print(new_img.shape)
    plt.imshow(new_img)
    plt.savefig("./2.png")
    if i ==0:
        break
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
