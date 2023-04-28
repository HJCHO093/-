import cv2
import os
import numpy as np
from pdf2image import convert_from_path
import functions as fs
from PIL import Image
import modules

#getting path
resource_path=os.getcwd()+"\\-\\resource\\"         #path
print(resource_path)
file_name="bluewhale.pdf"                       #file name
pdffile_path=resource_path + file_name          
print(pdffile_path)
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
closing_image=fs.closing(image2)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)
for i in range(1, cnt):
    (x, y, w, h, area)=stats[i]
    if w>=fs.weighted(5) and h>=fs.weighted(5): #setting the threshold that we can choose only what we need
        cv2.rectangle(image2, (x, y, w, h), (255, 0, 0), 1)
    
    """
    #showing heights and widths of each object
    fs.put_text(image2, w, (x, y + h + 30))
    fs.put_text(image2, h, (x, y + h + 60))
    """

#resizing and opening image(unfixed)
"""
cv2.namedWindow('Resized Window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', 1048, 1048)
"""

cv2.imshow('Resized Window', image2)

#press esc to escape
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()