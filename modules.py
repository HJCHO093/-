import cv2
import numpy as np
import functions as fs

def remove_noise(image):
    #binarization, masking(removing redundant parts)
    image=fs.threshold(image)
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
        if pixels >= width * 0.8: #adding staves
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