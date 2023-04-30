import cv2 
import matplotlib.pyplot as plt
img = cv2.imread("image_test/test_0.jpg", cv2.IMREAD_GRAYSCALE)
print(img.shape)
high_img = cv2.imread("./resource/high.png", cv2.IMREAD_GRAYSCALE)

print(high_img.shape)
new_img = cv2.resize(high_img, (img.shape[1], img.shape[0]),interpolation = cv2.INTER_AREA)
print(new_img.shape)
plt.imshow(new_img)
plt.savefig("./1.png")