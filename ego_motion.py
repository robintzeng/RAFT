import cv2
import numpy as np 
import random
import matplotlib.pyplot as plt
import glob
import os 
from tqdm import tqdm
import copy
def get_orb_features(img):
  descriptor = cv2.ORB_create()
  keypoints, descriptors = descriptor.detectAndCompute(img, None)
  return keypoints, descriptors

def match_keypoints(desc_1, desc_2, ratio=0.5):

  bf = cv2.BFMatcher()
  knn_match = bf.knnMatch(desc_1,desc_2,k=2)
  matches = []
  for m,n in knn_match:
    if m.distance < ratio*n.distance:
        matches.append(m)
  return matches

def geometricDistance(pt1,pt2, h):

    p1 = np.array([pt1[0], pt1[1],1])
    est = np.dot(h, p1.T)
    
    p2 = np.array([pt2[0], pt2[1],1])
    
    est1 = np.array([est[0],est[1],1])

    return  np.linalg.norm(est1-p2)

def warp_images(img_folder,mask_folder):
    for (path,mask) in tqdm(zip(img_folder,mask_folder)):
        images = glob.glob(os.path.join(path, '*.png')) + \
                        glob.glob(os.path.join(path, '*.jpg'))
        
        masks = glob.glob(os.path.join(mask, '*.png')) + \
                    glob.glob(os.path.join(mask, '*.jpg'))
        
        
        images = sorted(images)
        mask = sorted(masks)
        
        folder = 'warp_img/'+path.split('/')[-2]
        print(path.split('/')[-2])
        
        for i, (imfile1, imfile2, mask1,mask2) in enumerate(zip(images[:-1], images[1:], masks[:-1],mask[1:])):
            #print(i)
            img1 = cv2.imread(imfile1)
            img2 = cv2.imread(imfile2)
            mask1 = cv2.imread(mask1,0)
            mask2 = cv2.imread(mask2,0)
            
            rows,cols,_ = img1.shape
            
            object_mask1 = np.where(mask1==0,mask1,1)
            object_mask2 = np.where(mask2==0,mask2,1)
            y1,x1 = np.where(object_mask1==1)
            y2,x2 = np.where(object_mask2==1)

            ## tried the crop the obj
            try:
                
                ali1 = mask1.shape[1] - img1.shape[1]
                ali2 = mask2.shape[1] - img2.shape[1]

                if(ali1 < 0):
                    
                    crop_img1 = np.copy(img1[:,:ali1,:])
                    crop_img2 = np.copy(img2[:,:ali2,:])
    
                elif(ali1 ==0):
                    
                    crop_img1 = np.copy(img1[:,:,:])
                    crop_img2 = np.copy(img2[:,:,:])
                
                if(y1.size>0 and x1.size >0):
                    y1_max = np.max(y1)
                    y1_min = np.min(y1)
                    x1_max = np.max(x1)
                    x1_min = np.min(x1)
                    crop_img1[y1_min:y1_max,x1_min:x1_max] = 0
                
                
                if(y2.size>0 and x2.size>0):
                    
                    y2_max = np.max(y2)
                    y2_min = np.min(y2)
                    x2_max = np.max(x2)
                    x2_min = np.min(x2)
                    crop_img2[y2_min:y2_max,x2_min:x2_max] = 0
                        


                kp_1, desc_1 = get_orb_features(crop_img1)
                kp_2, desc_2 = get_orb_features(crop_img2)
                
                matches = match_keypoints(desc_1, desc_2)
                pt1_list = []
                pt2_list = []

                for match in matches:
                    (x1, y1) = kp_1[match.queryIdx].pt
                    (x2, y2) = kp_2[match.trainIdx].pt
                    ## kill the keypoints around the object
                    if (x1_max >= x1 >=  x1_min) and (y1_max >=y1 >= y1_min)and \
                       (x2_max >= x2 >=  x2_min) and (x2_max >= x2 >=  x2_min):
                        continue 
                    else:
                        pt1_list.append([x1, y1])
                        pt2_list.append([x2, y2])
                        
                
                pt1 = np.array(pt1_list)
                pt2 = np.array(pt2_list)
            
                H, masked = cv2.findHomography(pt2,pt1, cv2.RANSAC, 5.0)
                dst = cv2.warpPerspective(img2,H,(cols,rows))
                
            except:
                ## try to use the whole img
                
                try:
                    ali1 = mask1.shape[1] - img1.shape[1]
                    ali2 = mask2.shape[1] - img2.shape[1]

                    if(ali1 < 0):
                        
                        crop_img1 = np.copy(img1[:,:ali1,:])
                        crop_img2 = np.copy(img2[:,:ali2,:])
        
                    elif(ali1 ==0):
                        
                        crop_img1 = np.copy(img1[:,:,:])
                        crop_img2 = np.copy(img2[:,:,:])
                    

                    kp_1, desc_1 = get_orb_features(crop_img1)
                    kp_2, desc_2 = get_orb_features(crop_img2)


                    matches = match_keypoints(desc_1, desc_2)
                    pt1_list = []
                    pt2_list = []

                    for match in matches:
                        (x1, y1) = kp_1[match.queryIdx].pt
                        (x2, y2) = kp_2[match.trainIdx].pt
                        ## kill the keypoints around the object
                        if (x1_max >= x1 >=  x1_min) and (y1_max >=y1 >= y1_min)and \
                            (x2_max >= x2 >=  x2_min) and (x2_max >= x2 >=  x2_min):
                            
                            continue 
                        else:
                            pt1_list.append([x1, y1])
                            pt2_list.append([x2, y2])
                        
                    pt1 = np.array(pt1_list)
                    pt2 = np.array(pt2_list)
                    ### something the camera moving is not smooth enough to warp
                    H, masked = cv2.findHomography(pt2,pt1, cv2.RANSAC, 5.0)
                    dst = cv2.warpPerspective(img2,H,(cols,rows))
                except:
                    print("ex")
                    dst = img2
                    ## only a mark to show that the img cannot be warped
                    H = 777* np.ones((3,3))


            
            if not os.path.exists(folder):
                os.makedirs(folder)
            if i<10:
                cv2.imwrite( folder + '/0'+ str(i)+'.png',dst)
            else:
                cv2.imwrite( folder + '/'+ str(i)+'.png',dst)

            with open(folder + '/homography.npy', 'wb') as f:
                np.save(f,H)

if __name__ == "__main__":
    img_folder = glob.glob("datasets/DAVIS/JPEGImages/480p/*/")
    mask_folder = glob.glob("datasets/DAVIS/Annotations/480p/*/")
    
    img_folder = sorted(img_folder)
    mask_folder = sorted(mask_folder)
    
    #img_folder = img_folder[0:2]
    #mask_folder = mask_folder[0:2]

    # testing
    #img_folder = ['datasets/DAVIS/JPEGImages/480p/bear/']
    #mask_folder= ["datasets/DAVIS/Annotations/480p/bear/"]
    warp_images(img_folder,mask_folder)
    
        