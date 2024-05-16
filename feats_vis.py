"""
Shanghai Westwell Perception Lab
Author: Fei Wu  Australian National University
Adapted by Danni Wu  Tongji University
Ref: Beyond Cross-view Image Retrieval: Highly Accurate Vehicle Localization Using Satellite Image
Date: 0714/2023
"""
import os
import numpy as np

import cv2
from PIL import Image
from sklearn.decomposition import PCA
import mmcv

def visual_feature(feature,layer="img",path="runs/wf_feat_analysis/lidar_only"):
    '''
    Using PCA to decrease channels of tensor to be 3 channels for visualization saved as .png format
    Args:
        [1] feature: torch tensor (B,C,H,W)
        [2] layer: name.png for visualization
        [3] path: for saving 
    Return:
        None
    '''
    print("feature shape: ",feature.shape)
  
    def reshape_normalize(x):
        """
        Args: 
            x : [B,C,H,W]
        """
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    grd_feat = feature.data.cpu().numpy() 
    B, C, H, W = grd_feat.shape

    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(grd_feat))

    grd_feat=((normalize(pca.transform(reshape_normalize(grd_feat))) + 1) / 2).reshape(B, H, W, 3)
    # print("grd_feat: ",grd_feat.shape,grd_feat[0].shape)

    mmcv.mkdir_or_exist(path)

    for b in range(B):
        grd = Image.fromarray((grd_feat[b] * 255).astype(np.uint8))
        grd = grd.resize((W, H))
        print(f"{layer} saving to.... {path}/{layer}_{b}.png")
        grd.save(f'{path}/{layer}_{b}.png')

        org = cv2.imread(f'{path}/{layer}_{b}.png')
        org = cv2.flip(org,-1)
        cv2.imwrite(f'{path}/{layer}_{b}.png',org)

    # # grd.save(path+layer+".png")
    # count=0
    # while os.path.exists(f"{path}/{layer}.png"):
    #     layer=layer+str(count)
    #     count=count+1

    # print(f"saving feature.............: {path}/{layer}.png")
    # grd.save(f"{path}/{layer}.png")



def save_depth_image(img, tag="depth_img", path="feats_vis"):
    """
    depth image: [B, N, C, H, W]
    """
    if not os.path.exists(path):
        os.mkdir(path)

    img = img.permute(0, 1, 3, 4, 2) #[B, N, H, W, C]
    B, N, H, W, C = img.shape

    for b in range(B):
        for i in range(N):
            image_depth = img[b][i].cpu().detach().numpy()
            # print(image)
            # print(image.shape)
            image_color = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=15), cv2.COLORMAP_JET)
            img_vis = Image.fromarray(image_color)
            print(f"{tag} saving to.... {path}/{tag}_{b}_{i}.png")
            img_vis.save(f'{path}/{tag}_{b}_{i}.png')

            org = cv2.imread(f'{path}/{tag}_{b}_{i}.png')
            org = cv2.flip(org,-1)
            cv2.imwrite(f'{path}/{tag}_{b}_{i}.png',org)


# def save_tensor_image(img, tag="img", path="/cv/wdn/bev_fusion/feats_vis"):
#     """
#     image: [B, N, C, H, W]
#     """
#     img = img.permute(0, 1, 3, 4, 2) # [B, N, H, W, C]
#     B, N, H, W, C = img.shape

#     for b in range(B):
#         for i in range(N):
#             image = img[b][i].cpu().detach().numpy()
#             print(image)
#             print(image.shape)
#             img1 = Image.fromarray((image*255).astype(np.uint8))
#             img1.resize((H, W))
#             print(f"{tag} saving to.... {path}/{tag}_{b}_{i}.png")
#             img1.save(f'{path}/{tag}_{b}_{i}.png')
