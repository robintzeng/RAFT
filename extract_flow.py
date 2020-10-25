import sys
from numpy.lib.function_base import gradient

from numpy.lib.twodim_base import histogram2d
sys.path.append('core')
import pandas as pd 
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


# The mask is aligned to the img, but consective imgs are not aligned in DAVIS

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image

    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    ## Instead of showing the img , we write it out
    cv2.imwrite('output_img/image'+str(i)+".jpg",img_flo[:, :, [2,1,0]])

def plt_his(obj_angle,back_angle,obj_gradient,back_gradient,his_file_name,folder,bin_size):
    
    back_ave = np.sum(back_gradient) / back_gradient.shape[0] 
    obj_ave = np.sum(obj_gradient)   / obj_gradient.shape[0]
    print(back_gradient.shape[0]+obj_gradient.shape[0])
    
    titles =['obj_'+str(obj_ave),'back_' + str(back_ave)]
    angle = [obj_angle,back_angle] 
    gradient = [obj_gradient,back_gradient]

    f,a = plt.subplots(2,1)
    a = a.ravel()
    for idx,ax in enumerate(a):
        ax.hist(angle[idx], bins=np.arange(-np.pi,np.pi,bin_size),weights=gradient[idx])
        ax.set_title(titles[idx])
        ax.set_xlabel("degree")
        ax.set_ylabel("value")
    plt.tight_layout()
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(os.path.join(folder,his_file_name))
    plt.close()

    return back_ave,obj_ave


def flow_separate(img,mask,flo,i,folder,bin_size):
    
    his_file_name = 'his'+str(i) +'.png'
    img_file_name = 'image'+ str(i)+".jpg"
    
    flo = flo[0].permute(1,2,0).cpu().numpy()
    img = img[0].permute(1,2,0).cpu().numpy()

    #print(flo.shape)
    #print(mask.shape)
    ali = mask.shape[1] - flo.shape[1]
    object_mask = np.where(mask==0,mask,1)
    background_mask = np.ones_like(object_mask) - object_mask
    
    ## calculate the point of obj and background
    object_mask = object_mask.flatten()
    ## Align the 480p mask and img
    if(ali < 0):
        obj_angle = (flo[:,:ali,0]).flatten()
        obj_gradient = np.abs((flo[:,:ali,1]).flatten())
        
        obj_angle = obj_angle[object_mask==1]
        obj_gradient = obj_gradient[object_mask==1]

        
        background_mask = background_mask.flatten()
        
        back_angle = (flo[:,:ali,0]).flatten()
        back_gradient = np.abs((flo[:,:ali,1]).flatten())
        
        back_angle = back_angle[background_mask==1]
        back_gradient = back_gradient[background_mask==1]
    
    elif(ali ==0):
        obj_angle = (flo[:,:,0]).flatten()
        obj_gradient = np.abs((flo[:,:,1]).flatten())
        
        obj_angle = obj_angle[object_mask==1]
        obj_gradient = obj_gradient[object_mask==1]

        
        background_mask = background_mask.flatten()
        
        back_angle = (flo[:,:,0]).flatten()
        back_gradient = np.abs((flo[:,:,1]).flatten())
        
        back_angle = back_angle[background_mask==1]
        back_gradient = back_gradient[background_mask==1]

    ### for image output
    
    #flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    #cv2.imwrite(img_file_name,img_flo[:, :, [2,1,0]])
    
    #plt_his(obj_angle,back_angle,obj_gradient,back_gradient,his_file_name,folder,bin_size)
    return obj_angle,back_angle,obj_gradient,back_gradient

    

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    global_back_ave = []
    global_obj_ave = []
    global_name = []

    
    for (path,mask) in tqdm(zip(img_folder,mask_folder)):
        
        print("\n")
        print(path.split('/')[-2])
        print(mask.split('/')[-2])            
        with torch.no_grad():
            images = glob.glob(os.path.join(path, '*.png')) + \
                    glob.glob(os.path.join(path, '*.jpg'))
            
            masks = glob.glob(os.path.join(mask, '*.png')) + \
                    glob.glob(os.path.join(mask, '*.jpg'))
            

            images = sorted(images)
            masks = sorted(masks)

            global_obj_angle = np.array([])
            global_back_angle = np.array([])
            global_obj_gradient = np.array([])
            global_back_gradient = np.array([])
            
            folder_name = os.path.join('output_img',path.split('/')[-2])
            print(folder_name)
            for i, (imfile1, imfile2, mask) in tqdm(enumerate(zip(images[:-1], images[1:], masks[:-1]))):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                mask = cv2.imread(mask,0) 

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                flow_low, flow_up = model(image1, image2, iters=10, test_mode=True)
                #viz(image1, flow_up,i)
                obj_angle,back_angle,obj_gradient,back_gradient = flow_separate(image1,mask,flow_up,i,folder_name,bin_size=args.bin_size)
                global_obj_angle = np.append(global_obj_angle,obj_angle)
                global_back_angle = np.append(global_back_angle,back_angle)
                global_obj_gradient = np.append(global_obj_gradient,obj_gradient)
                global_back_gradient = np.append(global_back_gradient,back_gradient)



        his_file_name = path.split('/')[-2]+'_his_global.png'
        back_ave,obj_ave = plt_his(global_obj_angle,global_back_angle,global_obj_gradient,global_back_gradient, his_file_name,folder_name,args.bin_size)
        
        global_back_ave.append(back_ave)
        global_obj_ave.append(obj_ave)
        global_name.append(path.split('/')[-2])

    


    fig, ax = plt.subplots()
    ax.scatter(global_back_ave, global_obj_ave)
    ax.set_xlabel("back")
    ax.set_ylabel("obj")
    for i, txt in enumerate(global_name):
        ax.annotate(txt, (global_back_ave[i], global_obj_ave[i]))  
    plt.savefig('output_img/spread.png')
    plt.close()

    ## solve the calculated data 
    df = pd.DataFrame(list(zip(global_name,global_back_ave, global_obj_ave)), 
               columns =['Name', 'Background','Object'])

    df.to_csv ('output_img/spread.csv', index = False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    args.model = "models/raft-sintel.pth"   
    args.bin_size = np.pi/32
    args.mask = False


    img_folder = glob.glob("datasets/DAVIS/JPEGImages/480p/*/")
    mask_folder = glob.glob("datasets/DAVIS/Annotations/480p/*/")
    img_folder = sorted(img_folder)
    mask_folder = sorted(mask_folder)
    

    
    ##testing 
    #img_folder = img_folder[74:76]
    #mask_folder = mask_folder[74:76]
    #print(img_folder)
    
    demo(args)
        
