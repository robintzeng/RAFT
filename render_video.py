import os
import cv2
import numpy as np
import glob
import argparse

def render_video(args):
    img_array = []
    for filename in sorted(glob.glob(os.path.join(args.i, '*.png'))):
        img = cv2.imread(filename)
        print(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    img_r = img_array[0]
    height, width, layers = img_r.shape
    size = (width,height)

    out = cv2.VideoWriter(args.o,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)-30):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',default= "warp_img/walking" ,help="img folder path for rendering")
    parser.add_argument('-o',default= "project.avi" ,help="path for output video")
    args = parser.parse_args()
    
    # testing
    #args.i = 'warp_img/walking'
    #args.o = 'a.avi'
    render_video(args)

    