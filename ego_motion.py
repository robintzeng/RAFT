import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
import os 
import argparse
from tqdm import tqdm

# The mask is aligned to the img, but consective imgs are not aligned in DAVIS

def get_orb_features(img):
  descriptor = cv2.ORB_create()
  keypoints, descriptors = descriptor.detectAndCompute(img, None)
  return keypoints, descriptors

def match_keypoints(desc_1, desc_2, ratio=0.75):

  bf = cv2.BFMatcher()
  knn_match = bf.knnMatch(desc_1,desc_2,k=2)
  matches = []
  for m,n in knn_match:
    if m.distance < ratio*n.distance:
        matches.append(m)
  return matches

def boundry_check(mask,y,x):
    '''
    check whethere the keypoint is on the boundry or around the boundry
    ''' 
    y = int(y)
    x = int(x)
    if(mask[y][x] == 1):
        return True
    elif(y-1 >=0 and mask[y-1][x] == 1): # up
        return True
    elif(y+1 < mask.shape[0] and mask[y+1][x] == 1): # down
        return True
    elif(x-1 >=0 and mask[y][x-1] == 1): # left
        return True
    elif(x+1 < mask.shape[1] and mask[y][x+1] == 1): # right
        return True
    
    elif(y+1 < mask.shape[0] and x+1 < mask.shape[1] and mask[y+1][x+1] == 1):
        return True
    elif(y+1 < mask.shape[0] and x-1 >=0 and mask[y+1][x-1] == 1):
        return True
    elif(y-1 >=0 and x+1 < mask.shape[1] and mask[y-1][x+1] == 1):
        return True
    elif(y-1 >=0 and x-1 >=0 and mask[y-1][x-1] == 1):
        return True
    else:
        return False


def warp_images(args):
    # testing
    #img_folder = img_folder[0:2]
    #mask_folder = mask_folder[0:2]
    #img_folder = ['datasets/DAVIS/JPEGImages/480p/bear/']
    #mask_folder= ["datasets/DAVIS/Annotations/480p/bear/"]
    
    img_folder = glob.glob(args.i)
    mask_folder = glob.glob(args.m)
    
    img_folder = sorted(img_folder)
    mask_folder = sorted(mask_folder)

    for (path,mask) in tqdm(zip(img_folder,mask_folder)):
        images = glob.glob(os.path.join(path, '*.png')) + \
                        glob.glob(os.path.join(path, '*.jpg'))
        
        masks = glob.glob(os.path.join(mask, '*.png')) + \
                    glob.glob(os.path.join(mask, '*.jpg'))
        
        
        images = sorted(images)
        masks = sorted(masks)
        
        #print(images)
        #print(masks)
        
        folder = os.path.join(args.o, path.split('/')[-2])
        print(path.split('/')[-2])
        
        for i, (imfile1, imfile2, mask1, mask2) in enumerate(zip(images[:-1], images[1:], masks[:-1],masks[1:])):
            
            img1 = cv2.imread(imfile1)
            img2 = cv2.imread(imfile2)
            mask1 = cv2.imread(mask1,0)
            mask2 = cv2.imread(mask2,0)
            
            rows,cols,_ = img1.shape
            

            ### turn the obj into 1
            object_mask1 = np.where(mask1==0,mask1,1)
            object_mask2 = np.where(mask2==0,mask2,1)

            ## if we want to kill the keypoints in the background, we have to toggle the mask
            if(args.bg): 
                object_mask1 = np.ones_like(object_mask1) - object_mask1
                object_mask2 = np.ones_like(object_mask2) - object_mask2
                    

            
            kp_1, desc_1 = get_orb_features(img1)
            kp_2, desc_2 = get_orb_features(img2)
            
            

            matches = match_keypoints(desc_1, desc_2)
            pt1_list = []
            pt2_list = []
            
            for match in matches:
                (x1, y1) = kp_1[match.queryIdx].pt
                (x2, y2) = kp_2[match.trainIdx].pt
                ## kill the keypoints around the object
                if(boundry_check(object_mask1,y1,x1) == True or boundry_check(object_mask2,y2,x2) == True):
                    continue
                else:
                    pt1_list.append([x1, y1])
                    pt2_list.append([x2, y2])

            pt1 = np.array(pt1_list)
            pt2 = np.array(pt2_list)
            
            try:
                A, _ = cv2.estimateAffinePartial2D(pt2,pt1, method = cv2.RANSAC,ransacReprojThreshold = 5.0)
                dst = cv2.warpAffine(img2,A,(cols,rows))
            except: 
                ## if the background moves a lot and cannot generate Affine Matrix,
                ## we will use a special A for indicating this kind of movement  
                dst = img2
                A = 777*np.ones((2,3))
                print("No A")

            if not os.path.exists(folder):
                os.makedirs(folder)
            if i<10:
                cv2.imwrite( folder + '/0'+ str(i)+'.png',dst)
            else:
                cv2.imwrite( folder + '/'+ str(i)+'.png',dst)

            with open(folder + '/homography.npy', 'wb') as f:
                np.save(f,A)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',default= "datasets/DAVIS/JPEGImages/480p/*/" ,help="img folder path for warping")
    parser.add_argument('-m',default= "datasets/DAVIS/Annotations/480p/*/" ,help="mask folder path for warping")
    parser.add_argument('-o',default= "warp_img" ,help="output path for output imgs")
    parser.add_argument('-bg', action='store_true',help = "kill the keypoints on background")
    args = parser.parse_args()
    
    warp_images(args)
    
        