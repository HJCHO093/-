import cv2
import numpy as np
import functions as fs
import matplotlib.pyplot as plt
def remove_noise(image):
    #binarization, masking(removing redundant parts)
    image=fs.threshold(image)

    cv2.imwrite("./new.png",image)
    mask=np.zeros(image.shape, np.uint8)



    #labeling and drawing rectangle on each object
    cnt, labels, stats, centroids=cv2.connectedComponentsWithStats(image)
    for i in range(1, cnt):
        x, y, w, h, area=stats[i]
        if w>image.shape[1]*0.5:  #Only on stave
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)

    #masking
    masked_image=cv2.bitwise_and(image, mask)
    return masked_image

def remove_staves(image): #removing staves
    height, width=image.shape
    staves=[]
    for row in range(height):
        pixels=0
        for col in range(width):
            pixels += (image[row][col]==255) #counting white pixels in each row
        if pixels >= width * 0.8: #adding staves #width에 곱해지는 실수값이 stave를 인식하기 위해 중요
            if len(staves)==0 or abs(staves[-1][0]+staves[-1][1]-row)>1:
                staves.append([row, 0])
            else:
                staves[-1][1]+=1
    for staff in range(len(staves)):
        top_pixel=staves[staff][0] #top y value
        bot_pixel=staves[staff][0]+staves[staff][1] #bottom y value
        for col in range(width):
            if image[top_pixel - 1][col]==0 and image[bot_pixel + 1][col]==0:
                for row in range(top_pixel, bot_pixel+1):
                    image[row][col]=0 #removing staves
    return image, [x[0] for x in staves]

def normalization(image, staves, standard):
    avg_distance=0
    lines=int(len(staves)/5) #the number of staves
    for line in range(lines):
        for staff in range(4):
            staff_above=staves[line*5 + staff]
            staff_below=staves[line*5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)
    
    avg_distance /= len(staves)-lines #add every distance between staves and divide by the number of spaces
    height, width=image.shape
    weight=standard/avg_distance
    new_width=int(width * weight)
    new_height=int(height * weight)
    image=cv2.resize(image, (new_width, new_height))
    ret, image=cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    staves=[x * weight for x in staves]
    return image, staves

def object_detection(image, staves):
    lines=int(len(staves)/5) #the number of staves
    objects=[]
    closing_image=fs.closing(image)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)
    for i in range(1, cnt):
        (x, y, w, h, area)=stats[i]
        """
        #showing rectangles, heights and widths of each object
        cv2.rectangle(image2, (x, y, w, h), (255, 0, 0), 1)
        fs.put_text(image2, w, (x, y + h + 30))
        fs.put_text(image2, h, (x, y + h + 60))
        """
        if w>=fs.weighted(5) and h>=fs.weighted(5): #setting the threshold that we can choose only what we need
            center = fs.get_center(y, h)
            for line in range(lines):
                area_top=staves[line * 5] - fs.weighted(20)
                area_bot=staves[(line+1)*5 - 1] + fs.weighted(20)
                if area_top <= center <= area_bot: #adding objects with the line numbers contained
                    objects.append([line, (x, y, w, h, area)]) 
    objects.sort()
    return image, objects

def object_analysis(image, objects):
    for obj in objects:
        stats=obj[1]
        stems=fs.stem_detection(image, stats, 30) #detecting every lines in a object
        direction= None
        if len(stems)>0:
            if stems[0][0] - stats[0]>= fs.weighted(5):
                direction= True 
            else:
                direction= False
        obj.append(stems) #adding lists of lines to obj
        obj.append(direction) #adding directions of notes
    return image, objects