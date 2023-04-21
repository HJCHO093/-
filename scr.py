import cv2
import os
from pdf2image import convert_from_path
import functions as fs
from PIL import Image

#getting path
resource_path=os.getcwd()+"/resource/"          #path
file_name="bluewhale.pdf"                       #file name
pdffile_path=resource_path + file_name          

#converting pdf to jpg
pdfs=convert_from_path(pdffile_path)
for i, page in enumerate(pdfs):
    page.save(resource_path + file_name + str(i+1)+ '.jpg')
    
#reading image
image=cv2.imread(resource_path + file_name + str(i+1) + '.jpg')

#binarization
image=fs.threshold(image)

#resizing and opening image
cv2.namedWindow('Resized Window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', 1048, 1048)
cv2.imshow('Resized Window', image)

#press esc to escape
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()