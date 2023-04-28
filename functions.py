import cv2
import numpy as np

def threshold(image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image=cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image

def weighted(value):
    standard=10
    return int(value * (standard/10))

def closing(image):
    kernel = np.ones((weighted(5), weighted(5)), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def put_text(image, text, loc):
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.6, (255, 0, 0), 2)

def get_center(y, h):
    return (y+y+h)/2

VERTICAL=True
HORIZONTAL=False

def get_line(image, axis, axis_value, start, end, length):
    if axis: #vertical=True, horizontal=False
        points = [(i, axis_value) for i in range(start, end)] #vertical
    else:
        points = [(axis_value, i) for i in range(start, end)] #horizontal
    pixels = 0
    for i in range(len(points)):
        (y, x) = points[i]
        pixels += (image[y][x] == 255) #Counting white pixels
        next_point = image[y+1][x] if axis else image[y][x+1] #next point
        if next_point == 0 or i == len(points) - 1:
            if pixels >= weighted(length):
                break
            else:
                pixels = 0
    return y if axis else x, pixels

def stem_detection(image, stats, length):
    (x, y, w, h, area)=stats
    stems=[]
    for col in range(x, x+w):
        end, pixels = get_line(image, VERTICAL, col, y, y+h, length)
        if pixels:
            if len(stems)==0 or abs(stems[-1][0]+ stems[-1][2]-col)>=1:
                (x, y, w, h)=col, end-pixels+1, 1, pixels
                stems.append([x, y, w, h])
            else:
                stems[-1][2]+=1
    
    return stems
