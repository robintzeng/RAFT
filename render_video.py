import cv2
import numpy as np
import glob
 
img_array = []
for filename in sorted(glob.glob('warp_img/paragliding/*.png')):
    img = cv2.imread(filename)
    print(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


img_r = img_array[0]
height, width, layers = img_r.shape
size = (width,height)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)-30):
    out.write(img_array[i])
out.release()