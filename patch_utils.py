#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Patch utils file

#---------------------------------------------------------------------
#---------------------------------------------------------------------

import torch
import torch.nn as nn
import pickle
import torchvision
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import math
import kornia


import scipy.misc as miscconvert_tensor_SS_image
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os


from PIL import Image
import imageio


import scipy.misc 
import cv2
import matplotlib.pyplot as plt


import matplotlib
import matplotlib.pyplot as plt

from random import random as get_random
import time


#---------------------------------------------------------------------
# patch clipping class
#---------------------------------------------------------------------
class PatchConstraints(object):
    def __init__(self, set_loader):
        self.max_val = set_loader.max_val
        self.min_val = set_loader.min_val
        print("patch constraints (min-max): " + str(self.min_val) + " - " + str(self.max_val))
        return
    
    def __call__(self,module):
        if hasattr(module,'patch'):
            w=module.patch.data # NCWH
            w[:,0,:,:] = w[:,0,:,:].clamp(self.min_val[0], self.max_val[0])
            w[:,1,:,:] = w[:,1,:,:].clamp(self.min_val[1], self.max_val[1])
            w[:,2,:,:] = w[:,2,:,:].clamp(self.min_val[2], self.max_val[2])
            module.patch.data=w
    


#---------------------------------------------------------------------
# patches clipping class
#---------------------------------------------------------------------
class PatchesConstraints(object):
    def __init__(self, set_loader):
        self.max_val = set_loader.max_val
        self.min_val = set_loader.min_val
        print("patch constraints (min-max): " + str(self.min_val) + " - " + str(self.max_val))
        return
    
    def __call__(self,module):
        if hasattr(module,'patches'):
            for i,patch in enumerate(module.patches):
                w = patch.data #NCWH
                w[:,0,:,:] = w[:,0,:,:].clamp(self.min_val[0], self.max_val[0])
                w[:,1,:,:] = w[:,1,:,:].clamp(self.min_val[1], self.max_val[1])
                w[:,2,:,:] = w[:,2,:,:].clamp(self.min_val[2], self.max_val[2])
                module.patches[i].data=w


#---------------------------------------------------------------------
# patch_params
#---------------------------------------------------------------------

class patch_params(object):
    def __init__(self, 
        x_default = 0, 
        y_default = 0,
        noise_magn_percent = 0.05, 
        eps_x_translation = 1.0, 
        eps_y_translation = 1.0,
        max_scaling = 1.2, 
        min_scaling = 0.8,
        set_loader = None,
        use_transformations = False,
        rw_transformations = False):

            self.x_default = x_default
            self.y_default = y_default
            self.eps_x_translation = eps_x_translation
            self.eps_y_translation = eps_y_translation
            self.rw_transformations = rw_transformations
            self.use_transformations = use_transformations
            self.set_loader = set_loader
            self.noise_magn_percent = noise_magn_percent
            self.noise_magn = np.max(np.abs(self.set_loader.max_val - self.set_loader.min_val)) * \
                self.noise_magn_percent
            self.max_scaling =  max_scaling
            self.min_scaling =  min_scaling



#---------------------------------------------------------------------
# export the patch as numpy
#---------------------------------------------------------------------
def save_patch_numpy(patch, path):
    patch_np = patch.detach().cpu().numpy()
    with open(path, 'wb') as f:
        pickle.dump(patch_np, f)



#---------------------------------------------------------------------
# export an obj into a pkl file
#---------------------------------------------------------------------
def save_obj(path, obj = None):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)




#---------------------------------------------------------------------
# Import a new patch from a png value
#---------------------------------------------------------------------
def get_patch_from_img(path,set_loader):
    patch = imageio.imread(path)
    patch = set_loader.image_transform(patch, resize=False)
    patch = np.expand_dims(patch, 0)
    patch = torch.from_numpy(patch).float()
    return patch



#---------------------------------------------------------------------
# Import N new patches from a png value
#---------------------------------------------------------------------
def get_N_patches_from_img(path,set_loader, N=2):
    patches = []
    for i in range(N):
        patch = get_patch_from_img(path=path+'_'+str(i)+'.png', set_loader =set_loader)
        patches.append(patch)
    return patches



#---------------------------------------------------------------------
# add the patch using add_patch() to each tensor image in the mini-batch 
#---------------------------------------------------------------------
def add_patch_to_batch_randomly(
    images, 
    patch,
    patch_params,
    device='cuda', 
    int_filtering=False):

    patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
    is_patched = torch.empty([images.shape[0]])

    for k in range(images.size(0)):
        if get_random() > 0.5:
            images[k], patch_mask[k] = add_patch(
                image=images[k], 
                patch=patch, 
                patch_params = patch_params,
                use_transformations = patch_params.use_transformations,
                int_filtering=int_filtering)
            is_patched[k] = 1.0
        else:
            images[k] = images[k]
            patch_mask[k] = torch.zeros([1, images.shape[2], images.shape[3]])
            is_patched[k] = 0.0
    return images, patch_mask, is_patched





#---------------------------------------------------------------------
# Create a random patch 
#---------------------------------------------------------------------
def get_random_patch(cfg_patch, set_loader):
    patch = torch.rand(1,3, cfg_patch['height'], cfg_patch['width'])
    if set_loader.img_norm == False:
        patch *= 255.0 
    
    patch[:,0,:,:] -= set_loader.mean[0]
    patch[:,1,:,:] -= set_loader.mean[1]
    patch[:,2,:,:] -= set_loader.mean[2]
    patch[:,0,:,:] /= set_loader.std[0]
    patch[:,1,:,:] /= set_loader.std[1]
    patch[:,2,:,:] /= set_loader.std[2]

    return patch



#---------------------------------------------------------------------
# Create a N-random patches
#---------------------------------------------------------------------
def get_N_random_patches(cfg_patch, set_loader, N=2):
    patches = []
    # compute the width to keep the same side ratio and overall area of a single big patch
    height = int(cfg_patch['height'] / math.sqrt(N))
    print("Patch size: " + str(height))
    patch_size = {'height': height, 'width': 2*height}
    for i in range(N):
        patch = get_random_patch(patch_size, set_loader)
        patches.append(patch)
    return patches





#---------------------------------------------------------------------
# Import the patch from a numpy file
#---------------------------------------------------------------------
def get_patch_from_numpy(path):
    print("retrieving patch from: " + cfg_patch['path'])
    with open(path, 'rb') as f:
        patch = torch.from_numpy(pickle.load(f))
    return patch



#---------------------------------------------------------------------
# Remove mask from a batch of images
#---------------------------------------------------------------------
def remove_mask (images,mask):
    mask = F.interpolate(mask, size=images.shape[1:],mode='bilinear', align_corners=True)
    mask = torch.heaviside(mask - 0.5, values=torch.tensor([0.0]).to('cuda'))
    images[mask.squeeze(1)==1] = 255.0
    return images



#---------------------------------------------------------------------
# Add the patch_obj as a new model parameter
#---------------------------------------------------------------------
def init_model_patch(model, mode = "train", seed_patch = None):
    # add new attribute into the model class
    setattr(model, "patch", None)
    # patch initialization
    if mode =='train':                    
        model.patch = nn.Parameter(seed_patch, requires_grad=True)
    # load an already trained patch for testing
    elif mode =='test':
        model.patch = nn.Parameter(seed_patch, requires_grad=False)
    



#---------------------------------------------------------------------
# Add two patches as new model parameters
#---------------------------------------------------------------------
def init_model_N_patches(model, mode = "train", N = 2, seed_patches = None):
    # add new attribute into the model class
    list_patches = []
    if mode =='train':       
        list_patches = [ nn.Parameter(seed_patches[i], requires_grad=True) for i in range(N)]           
    elif mode =='test':
        list_patches = [ nn.Parameter(seed_patches[i], requires_grad=False) for i in range(N)] 
    setattr(model, "patches", None)  
    model.patches = nn.ParameterList(list_patches)   



#---------------------------------------------------------------------
# Set multiple output during the eval mode
# The new attribute specify in the corresponding network model to return 
# also the auxiliary outputs which are common usually used during the 
# train mode
#---------------------------------------------------------------------
def set_multiple_output(model):
    # add new attribute into the model class
    setattr(model, "multiple_outputs", True)




#---------------------------------------------------------------------
# add the patch using add_patch() to each tensor image in the mini-batch 
#---------------------------------------------------------------------
def add_patch_to_batch(
    images, 
    patch,
    patch_params,
    device='cuda', 
    use_transformations=None,
    int_filtering=False):

    patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])

    use_transf = use_transformations if use_transformations is not None else patch_params.use_transformations

    for k in range(images.size(0)):
        images[k], patch_mask[k]= add_patch(
            image=images[k], 
            patch=patch, 
            patch_params = patch_params,
            use_transformations = use_transf,
            int_filtering=int_filtering)

    return images, patch_mask





#---------------------------------------------------------------------
# add patches using add_patch() to each tensor image in the mini-batch 
#---------------------------------------------------------------------
def add_N_patches_to_batch(
    images, 
    patches, 
    patch_params_array, 
    device = 'cuda', 
    use_transformations = None,
    int_filtering=False):
    
    patches_mask = torch.zeros([images.shape[0], 1, images.shape[2], images.shape[3]])

    for i, patch in enumerate(patches):
        patch_params = patch_params_array[i]

        use_transf = use_transformations if use_transformations is not None else patch_params.use_transformations

        # mask corresponding to a single mask
        patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
        for k in range(images.size(0)):
            images[k], patch_mask[k] = add_patch(
                image=images[k], 
                patch=patch, 
                patch_params = patch_params,
                use_transformations = use_transf,
                int_filtering=int_filtering)

        # add multiple patch together
        patches_mask += patch_mask 
    
    # TODO - check for collision between multiple mask

    return images, patches_mask
        

        




#---------------------------------------------------------------------
# given a single tensor_image, this function creates a patched_image as a
# composition of the original input image and the patch using masks for
# keep everything differentable
#---------------------------------------------------------------------
def add_patch(image, 
    patch, 
    patch_params,
    device='cuda', 
    use_transformations=True, 
    int_filtering=False):

    applied_patch, patch_mask, img_mask, x_location, y_location = mask_generation(
            mask_type='rectangle', 
            patch=patch, 
            patch_params = patch_params,
            image_size=image.shape[:], 
            use_transformations = use_transformations,
            int_filtering=int_filtering)

    patch_mask = Variable(patch_mask, requires_grad=False).to(device)
    img_mask = Variable(img_mask, requires_grad=False).to(device)

    perturbated_image = torch.mul(applied_patch.type(torch.FloatTensor), patch_mask.type(torch.FloatTensor)) + \
        torch.mul(img_mask.type(torch.FloatTensor), image.type(torch.FloatTensor))
    
    return perturbated_image, patch_mask[0,:,:]






#---------------------------------------------------------------------
# TRANSFORMATION : Rotation
# the actual rotation angle is rotation_angle * 90 on all the 3 channels
# TODO: reimplement from scratch. 
#---------------------------------------------------------------------
def rotate_patch(in_patch):
    rotation_angle = np.random.choice(4)
    for i in range(0, rotation_angle):
        in_patch = in_patch.transpose(2,3).flip(3)
    return in_patch





#---------------------------------------------------------------------
# TRANSFORMATION: patch scaling
#---------------------------------------------------------------------
def random_scale_patch(patch, patch_params):
    scaling_factor = np.random.uniform(low=patch_params.min_scaling, high=patch_params.max_scaling)
    new_size_y = int(scaling_factor * patch.shape[2])
    new_size_x = int(scaling_factor * patch.shape[3])
    patch = F.interpolate(patch, size=(new_size_y, new_size_x), mode="bilinear", align_corners=True)
    return patch





#---------------------------------------------------------------------
# TRANSFORMATION: translation
# scale the patch (define the methodologies)
#---------------------------------------------------------------------
def random_pos(patch, image_size):
    x_location, y_location = int(image_size[2]) , int(image_size[1])
    x_location = np.random.randint(low=0, high=x_location - patch.shape[3])
    y_location = np.random.randint(low=0, high=y_location - patch.shape[2])
    return x_location, y_location





#---------------------------------------------------------------------
# TRANSFORMATION: translation
# scale the patch (define the methodologies)
#---------------------------------------------------------------------
def random_pos_local(patch, x_pos, y_pos, patch_params):
    eps_x = patch_params.eps_x_translation
    eps_y= patch_params.eps_y_translation
    x_location = np.random.randint(low= x_pos - eps_x, high=x_pos + eps_x)
    y_location = np.random.randint(low= y_pos - eps_y, high=y_pos + eps_y)
    return x_location, y_location




#---------------------------------------------------------------------
# TRANSFORMATION: uniform noise
#---------------------------------------------------------------------
def unif_noise(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    magn = patch_params.noise_magn
    noise = (-magnitude*2)* torch.rand(patch.size(), requires_grad=False).to('cuda') + magnitude
    patch_noise = (torch.clamp(((patch + noise) * std + mean), 0, max_val) - mean) / std
    return patch_noise

#---------------------------------------------------------------------
# TRANSFORMATION: gaussian noise
#---------------------------------------------------------------------
def gaussian_noise(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    noise = magnitude * torch.randn(patch.size(), requires_grad=False).to('cuda')
    patch_noise = (torch.clamp(((patch + noise) * std + mean), 0, max_val) - mean) / std
    return patch_noise

'''
#---------------------------------------------------------------------
# TRANSFORMATION: int filtering
#---------------------------------------------------------------------
def get_integer_patch(patch, patch_params, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    
    int_patch = patch.clone()
    if patch_params.set_loader.img_norm is False:
        int_patch = (torch.clamp(torch.round(int_patch * std + mean), 0, max_val) - mean) / std
#         int_patch = torch.round(int_patch)
    else:
        int_patch = (torch.clamp(torch.round(255 * (int_patch * std + mean)), 0, max_val) - mean) / std
#         int_patch *= 255.0 
#         int_patch = torch.round(int_patch)
#         int_patch /= 255.0
    return int_patch
'''


#---------------------------------------------------------------------
# TRANSFORMATION: int filtering
#---------------------------------------------------------------------
def get_integer_patch(patch, patch_param):
    
    mean = torch.Tensor(patch_param.set_loader.mean.reshape((1, 3, 1, 1))).to(patch.device) #
    std = torch.Tensor(patch_param.set_loader.std.reshape((1, 3, 1, 1))).to(patch.device) #
    int_patch = patch.clone()
    if patch_param.set_loader.img_norm is False:
        int_patch = (torch.clamp(torch.round(patch * std + mean), 0, 255.) - mean) / std
    else:
        int_patch = (torch.clamp(torch.round(255 * (patch * std + mean)), 0, 255.)/255.0 - mean) / std
#         int_patch *= 255.0 
#         int_patch = torch.round(int_patch)
#         int_patch /= 255.0
    return int_patch

#---------------------------------------------------------------------
# TRANSFORMATION: contrast change
#---------------------------------------------------------------------
def contrast_change(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    contr_delta = 1 + magnitude * torch.randn(1).numpy()[0]
    patch = (torch.clamp((patch * std + mean) * contr_delta, 0, max_val) - mean) / std
    return patch


#---------------------------------------------------------------------
# TRANSFORMATION: brightness change
#---------------------------------------------------------------------
def brightness_change(patch, magnitude, mean=[0, 0, 0], std=[1, 1, 1], max_val=255):
    bright_delta = magnitude * torch.randn(1).numpy()[0]
    patch = (torch.clamp((patch * std + mean) + bright_delta, 0, max_val) - mean) / std
#     for c in range(3):
#         patch[:, c, :, :] = torch.clamp(patch[:, c, :, :] + bright_delta, -mean[0, c, 0, 0].cpu().numpy(), 255-mean[0, c, 0, 0].cpu().numpy())
    return patch



#---------------------------------------------------------------------
# util for patch projection
#---------------------------------------------------------------------
def get_dest_corners(patch, extrinsic, intrinsic, pixel_dim=0.2, offset=[0, 0, 0], device='cuda'):
    # Define corners of each pixel of the patch (sign reference frame)
    p_h, p_w = patch.shape[2:]
    x, y, z = offset
    patch_corners = torch.Tensor([[[x, y, z, 1], 
                         [x, y - p_w*pixel_dim, z, 1],
                         [x, y - p_w*pixel_dim, -p_h*pixel_dim + z, 1],
                         [x, y, -p_h*pixel_dim + z, 1]]]).to(device)
    p = torch.transpose(patch_corners, 1, 2)
    
    # Transform to camera reference frame
    corners_points_homogeneous = extrinsic @ p
    corners_points_3d = corners_points_homogeneous[:, :-1, :] / corners_points_homogeneous[:, -1:, :]

    
    # Project onto image
    corner_pixels_homogeneous = intrinsic @ corners_points_3d
    corner_pixels = corner_pixels_homogeneous[:, :-1, :] / corner_pixels_homogeneous[:, -1:, :]
    
#     print(extrinsic, intrinsic)
#     print(patch_corners)
#     print(corners_points_homogeneous)
#     print(corners_points_3d)
#     print(corner_pixels_homogeneous)
#     print(corner_pixels)
    
    return torch.transpose(corner_pixels, 1, 2)




#---------------------------------------------------------------------
# patch projection for specific attack
#---------------------------------------------------------------------
def project_patch(im, patch, extrinsic, intrinsic, patch_params, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'): # mean=[0, 0, 0], std=[1, 1, 1]):
    use_transformations, int_filtering = True, False
#     mean = torch.Tensor(np.array([103.939, 116.779, 123.68]).reshape((1, 3, 1, 1))).to(device)  
#     mean = torch.Tensor(np.array([123.68, 116.779, 103.939]).reshape((1, 3, 1, 1))).to(device)
    mean = torch.Tensor(patch_params.set_loader.mean.reshape((1, 3, 1, 1))).to(device) #
    std = torch.Tensor(patch_params.set_loader.std.reshape((1, 3, 1, 1))).to(device) #
    max_val = 255
    if patch_params.set_loader.img_norm:
        max_val = 1
        

#     print(mean, std)
#     mean = torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).to(device)
    # Define corners of each pixel of the patch (sign reference frame)
    p_h, p_w = patch.shape[2:]
    h, w = im.shape[-2:]
    
    if use_transformations is True:
        patch = gaussian_noise(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
        patch = brightness_change(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
        patch = contrast_change(patch, magnitude=0.1, mean=mean, std=std, max_val=max_val)
    if int_filtering is True:
        patch = get_integer_patch(patch, patch_params, mean=mean, std=std, max_val=max_val)
        
    

    if p_h != p_w:
        im_p = im
        # project in blocks!
        for i in range(2):
#                 points_src = torch.Tensor([[
#                     [0, i * p_w//2], [p_h, i * p_w//2], [p_h, (i + 1) * p_w//2], [0., (i + 1) * p_w//2],
#                 ]]).to(device)
            points_src = torch.Tensor([[
                [0, 0], [p_h, 0.], [p_h, p_w//2], [0., p_w//2],
            ]]).to(device)
            x_off, y_off, z_off = offset
            points_dst = get_dest_corners(patch[:, :, :, i*p_w//2:(i+1)*p_w//2], extrinsic, intrinsic, pixel_dim=pixel_dim, offset=[x_off, y_off-i * p_w/2 * pixel_dim, z_off], device=device)

            # compute perspective transform
            M: torch.Tensor = kornia.get_perspective_transform(points_src, points_dst).to(device)
            # warp the original image by the found transform
            data_warp: torch.Tensor = kornia.warp_perspective((patch[:, :, :, i*p_w//2:(i+1)*p_w//2].float() * std) + mean, M, dsize=(h, w))

            mask = torch.zeros_like(data_warp[0], device=device)
            mask[data_warp[0] > 0] = 1
            data_warp = ((data_warp - mean)/std)[0]

            mask_img = torch.ones((h, w), device=device) - mask

            im_p = im_p * mask_img  + data_warp * mask
        
        
    else:
        
        points_src = torch.Tensor([[
            [0, 0], [p_h, 0.], [p_h, p_w], [0., p_w],
        ]]).to(device)

        points_dst = get_dest_corners(patch, extrinsic, intrinsic, pixel_dim=pixel_dim, offset=offset, device=device)

        # compute perspective transform
        M: torch.Tensor = kornia.get_perspective_transform(points_src, points_dst).to(device)
        # warp the original image by the found transform
        data_warp: torch.Tensor = kornia.warp_perspective((patch.float() * std) + mean, M, dsize=(h, w))

        mask = torch.zeros_like(data_warp[0], device=device)
        mask[data_warp[0] > 0] = 1
        data_warp = ((data_warp - mean)/std)[0]

        mask_img = torch.ones((h, w), device=device) - mask

        im_p = im * mask_img  + data_warp * mask
        
    if torch.sum(torch.isnan(im_p)) > 0:
        return im
    return im_p, mask[0,:,:]


#---------------------------------------------------------------------
# REPROJECT PATCH ONTO IMAGE (BATCH VERSION)
#---------------------------------------------------------------------
def project_patch_batch(images, patch, extrinsic, intrinsic, patch_params, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'):# mean=[0, 0, 0], std=[1, 1, 1]):
    
    # return torch.stack([project_patch(images[j], patch, extrinsic[j], intrinsic[j], 
    #                                   pixel_dim=pixel_dim, offset=offset, 
    #                                   rescale=rescale, device=device, patch_params=patch_params) #mean=mean, std=std) 
    #                                               for j in range(images.shape[0])], dim=0)
    
    patch_mask = torch.empty([images.shape[0], 1, images.shape[2], images.shape[3]])
    for j in range(images.shape[0]):
        images[j], patch_mask[j] = project_patch(images[j], patch, extrinsic[j], intrinsic[j], 
                                      pixel_dim=pixel_dim, offset=offset,rescale=rescale, device=device, patch_params=patch_params)
    return images, patch_mask


"""
#--------------------------------------------------------------------
# util for patch projection
#---------------------------------------------------------------------
def block_corners(i, j, pixel_dim=.1, offset=[0, 0, 0]):
    x, y, z = offset
    return torch.Tensor([[x, y + i*pixel_dim, j*pixel_dim + z, 1], 
                         [x, y + (i+1)*pixel_dim, j*pixel_dim + z, 1],
                         [x, y + (i+1)*pixel_dim, (j+1)*pixel_dim + z, 1],
                         [x, y + i*pixel_dim, (j+1)*pixel_dim + z, 1]])


#---------------------------------------------------------------------
# REPROJECT PATCH ONTO IMAGE
#---------------------------------------------------------------------
def project_patch(im, patch, extrinsic, intrinsic, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'):
    
    # Define corners of each pixel of the patch (sign reference frame)
    p_h, p_w = patch.shape[2:]
    h, w = im.shape[-2:]
#     print(patch.shape)
    
    if rescale is not None:
        scale = rescale
        h_or, w_or = h, w
        h //= scale
        w //= scale
        im = transforms.Resize((h, w))(im)
#         pixel_dim /= scale
#         height /= scale
#         x_offset /= scale
        intrinsic[0, 0] /= scale
        intrinsic[1, 1] /= scale
        intrinsic[0, 2] /= scale
        intrinsic[1, 2] /= scale
        
    
    patch_pixel_corners = torch.stack([block_corners(i, j, pixel_dim=pixel_dim, offset=offset) 
                                       for i in range(p_w) for j in range(p_h)], dim=0)
    p = torch.transpose(patch_pixel_corners, 1, 2)
    
    # Transform to camera reference frame
    pixel_points_homogeneous = extrinsic @ p
    pixel_points_3d = pixel_points_homogeneous[:, :-1, :] / pixel_points_homogeneous[:, -1:, :]
    
    # Project onto image
    pixel_pixels_homogeneous = intrinsic @ pixel_points_3d
    pixel_pixels = pixel_pixels_homogeneous[:, :-1, :] / pixel_pixels_homogeneous[:, -1:, :]

    
    # Find pixel borders from corners (up and down - left and right bounds are vertical as long as pitch, roll are close to 0)
#     x, y = pixel_pixels[:, 0, :].detach().cpu().numpy(), pixel_pixels[:, 1, :].detach().cpu().numpy()
    x, y = pixel_pixels[:, 0, :], pixel_pixels[:, 1, :]
    del patch_pixel_corners, pixel_points_homogeneous, pixel_points_3d, pixel_pixels_homogeneous, pixel_pixels
    AB = [(y[:, 1] - y[:, 0])/(x[:, 1] - x[:, 0]), y[:, 0] - (y[:, 1] - y[:, 0])/(x[:, 1] - x[:, 0]) * x[:, 0]]
    CD = [(y[:, 3] - y[:, 2])/(x[:, 3] - x[:, 2]), y[:, 2] - (y[:, 3] - y[:, 2])/(x[:, 3] - x[:, 2]) * x[:, 2]]

    # Find pixels that satisfy position conditions
#     xv, yv, zv = np.meshgrid(range(w), range(h), range(p_w * p_h))
    zv, yv, xv = torch.meshgrid(torch.Tensor(range(p_w * p_h)), torch.Tensor(range(h)), torch.Tensor(range(w)))
#     x_cond = (xv < x[:, 0]) & (xv > x[:, 2])
    x_cond = torch.logical_not(xv.ge(x[:, 0].reshape((-1, 1, 1)))) & xv.ge(x[:, 2].reshape((-1, 1, 1)))
#     y_cond = (yv < (AB[0] * xv + AB[1])) & (yv > (CD[0] * xv + CD[1]))
    y_cond = yv.ge(CD[0].reshape((-1, 1, 1)) * xv + CD[1].reshape((-1, 1, 1))) & torch.logical_not(yv.ge(AB[0].reshape((-1, 1, 1)) * xv + AB[1].reshape((-1, 1, 1))))
#     print(np.sum(x_cond))
    
    # Create masks
#     im_p = im
    mask = torch.zeros((3, p_w * p_h, h, w), requires_grad=False, device=device)
    patch_mask = patch.reshape((p_w * p_h, -1, 1, 1))
    
    # Mask is 1 only where the patch is present
#     mask[:, (x_cond & y_cond).transpose((2, 0, 1))] = 1
    mask[:, (x_cond & y_cond)] = 1
    
    del x_cond, y_cond, xv, yv, zv, x, y, AB, CD
    
    mask = torch.transpose(mask, 0, 1)
        
    # Required to sum up all the pixels in the same mask
    tot_mask = torch.sum(mask, dim=0)
    tot_patch_mask = torch.sum(mask * patch_mask, dim=0)
#     print(tot_patch_mask.shape)
    mask_im = torch.ones((3, h, w), device=device, requires_grad=False) - tot_mask
    
    # Resulting image is the mask-weighted sum of patch and image.
    im_p = mask_im * im + tot_patch_mask #* 255
    
    if rescale is not None:
        im_p = transforms.Resize((h_or, w_or))(im_p)
        intrinsic[0, 0] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[0, 2] *= scale
        intrinsic[1, 2] *= scale
    del mask_im, mask, tot_patch_mask, tot_mask
    return im_p



#---------------------------------------------------------------------
# REPROJECT PATCH ONTO IMAGE (BATCH VERSION)
#---------------------------------------------------------------------
def project_patch_batch(images, patch, extrinsic, intrinsic, pixel_dim=0.2, offset=[0, 0, 0], rescale=None, device='cuda'):
    return torch.stack([project_patch(images[j], patch, extrinsic[j], intrinsic[j], 
                                      pixel_dim=pixel_dim, offset=offset, 
                                      rescale=rescale, device=device) 
                                                  for j in range(images.shape[0])], dim=0)


#---------------------------------------------------------------------
# REPROJECT PATCH IN BLOCKS
#---------------------------------------------------------------------
def project_patch_blocks(im, patch, extrinsic, intrinsic, pixel_width=1, block_width=20, offset=[0, 0, 0], rescale=None, device='cuda'):
    # divide the patch in blocks.
    # Define corners of each pixel of the patch (sign reference frame)
    p_h, p_w = patch.shape[2:]
    h, w = im.shape[-2:]
    x, y, z = offset
    
    blocks_h = p_h // block_width
    blocks_w = p_w // block_width
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            block_dim = block_width * pixel_width
            offset_block = [0 + x, j *  block_dim + y, i * block_dim + z]
            patch_block = patch[:, :, i * block_width:(i+1)*block_width, j*block_width:(j+1)*block_width]
            im = project_patch(im, patch_block, extrinsic, intrinsic, pixel_dim=pixel_width, rescale=rescale, offset=offset_block, device=device)
            
    return im
    
#     return torch.stack([project_patch(images[j], patch, extrinsic[j], intrinsic[j], 
#                                       pixel_dim=pixel_dim, height=height, x_offset=x_offset, 
#                                       rescale=rescale, device=device) 
#                                                   for j in range(images.shape[0])], dim=0)


#---------------------------------------------------------------------
# REPROJECT PATCH ONTO IMAGE IN BLOCKS (BATCH VERSION)
#---------------------------------------------------------------------
def project_patch_blocks_batch(images, patch, extrinsic, intrinsic, pixel_width=0.2, block_width=20, offset=[0, 0, 0], rescale=None, device='cuda'):
    return torch.stack([project_patch_blocks(images[j], patch, extrinsic[j], intrinsic[j], 
                                      pixel_width=pixel_width, block_width=block_width, offset=offset, 
                                      rescale=rescale, device=device) for j in range(images.shape[0])], dim=0)
"""

#---------------------------------------------------------------------
# Apply transformation to the patch and generate masks
#---------------------------------------------------------------------
def mask_generation(
    patch,
    patch_params,
    mask_type='rectangle', 
    image_size=(3, 224, 224), 
    use_transformations = True,
    int_filtering=False):

    mean = torch.Tensor(patch_params.set_loader.mean.reshape((1, 3, 1, 1))).to('cuda') #
    std = torch.Tensor(patch_params.set_loader.std.reshape((1, 3, 1, 1))).to('cuda') #
    max_val = 255.0
    if patch_params.set_loader.img_norm:
        max_val = 1.0
    
    x_location = patch_params.x_default
    y_location = patch_params.y_default
    applied_patch = torch.zeros(image_size, requires_grad=False).to('cuda')

    if use_transformations is True:
        patch = random_scale_patch(patch, patch_params)
        patch = gaussian_noise(patch, patch_params.noise_magn, mean=mean, std=std, max_val=max_val)

        if patch_params.rw_transformations is True:
            patch = brightness_change(patch, magnitude=patch_params.noise_magn, mean=mean, std=std, max_val=max_val)
            patch = contrast_change(patch, magnitude=0.1, mean=mean, std=std, max_val=max_val)

        x_location, y_location = random_pos_local(patch, x_pos = x_location, y_pos = y_location, patch_params=patch_params)
        #patch = rotate_patch(patch)


    if int_filtering is True:
        patch = get_integer_patch(patch, patch_params)
    applied_patch[:,  y_location:y_location + patch.shape[2], x_location:x_location + patch.shape[3]] = patch[0]

    patch_mask = torch.zeros_like(applied_patch)
    patch_mask[:,  y_location:y_location + patch.shape[2], x_location:x_location + patch.shape[3]] = 1
    #patch_mask = applied_patch.clone()
    #patch_mask[patch_mask != 0.0] = 1.0
    img_mask = torch.ones([3,image_size[1], image_size[2]]).to('cuda') - patch_mask

    return applied_patch, patch_mask, img_mask, x_location, y_location



#---------------------------------------------------------------------
# export a tensor as png (for visualization)
# similar to save_patch_png
#---------------------------------------------------------------------
#def save_tensor_png(im_tensor, path, bgr=True, img_norm=False, mean = 0.0):
def save_tensor_png(im_tensor, path, set_loader):
    im_data = im_tensor.clone().reshape(im_tensor.shape[1],im_tensor.shape[2],im_tensor.shape[3])
    im_data = im_data.detach().cpu().numpy()
    im_data = set_loader.to_image_transform(im_data)
    im_data = im_data.astype('uint8')
    data_img = Image.fromarray(im_data)
    print("save patch as img ", path)
    data_img.save(path)
    del im_data


#---------------------------------------------------------------------
# convert a tensor to png 
#---------------------------------------------------------------------
#def convert_tensor_image(im_tensor, bgr=True, img_norm=False, mean = 0.0):
def convert_tensor_image(im_tensor, set_loader):
    im_data = im_tensor.clone().reshape(im_tensor.shape[1],im_tensor.shape[2],im_tensor.shape[3])
    im_data = im_data.detach().cpu().numpy()
    im_data = set_loader.to_image_transform(im_data)
    im_data = im_data.astype('uint8')
    im_data = Image.fromarray(im_data)
    return im_data


#---------------------------------------------------------------------
# convert a tensor semantic segmentation to png 
#---------------------------------------------------------------------
def convert_tensor_SS_image(im_tensor, model_name = None, orig_size = None, set_loader = None, input_classes = False):
    if 'mask2former' in model_name:
        im_data = im_tensor.clone().reshape(im_tensor.shape[2],im_tensor.shape[3])
        p_out = im_data.cpu().numpy()
    else:
        if input_classes:
            p_out = np.squeeze(im_tensor.cpu().numpy(), axis=0)
        else:
            p_out = np.squeeze(im_tensor.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        p_out = p_out.astype(np.float32)
         # float32 with F mode, resize back to orig_size
        p_out = cv2.resize(p_out, orig_size, interpolation=cv2.INTER_NEAREST)
        #p_out = misc.imresize(p_out, orig_size, "nearest", mode="F")
    decoded_p_out = set_loader.decode_segmap(p_out)
    return decoded_p_out






#---------------------------------------------------------------------
# export the patch as png (for visualization)
#---------------------------------------------------------------------
#def save_patch_png(patch, path, bgr=True, img_norm=False, mean = 0.0):
def save_patch_png(patch, path, set_loader):
    np_patch = patch.clone()

    #  (NCHW -> CHW -> HWC) 
    np_patch = np_patch[0].detach().cpu().numpy()
    np_patch = set_loader.to_image_transform(np_patch)
    np_patch = np_patch.astype('uint8')
    patch_img = Image.fromarray(np_patch)
    print("save patch as img ", path)
    patch_img.save(path)
    del np_patch

    



#-------------------------------------------------------------------
# plot a subfigure for visualizing the adversarial patch effect
#-------------------------------------------------------------------
#def save_summary_img(tensor_list, path, model_name, orig_size, loader,  bgr=True, img_norm=False, count=0, imm_num=0):
def save_summary_img(tensor_list, path, model_name, orig_size, set_loader, count=0, img_num=0):
    p_image = tensor_list[0]
    c_image = tensor_list[1]
    p_out = tensor_list[2]
    c_out = tensor_list[3]

    #  (NCHW -> CHW -> HWC) 
    p_image = p_image.detach().cpu().numpy()
    c_image = c_image.detach().cpu().numpy()

    p_image = set_loader.to_image_transform(p_image)
    c_image = set_loader.to_image_transform(c_image)
        
    p_image = p_image.astype('uint8')
    c_image = c_image.astype('uint8')
    
    p_out = np.squeeze(p_out.data.max(1)[1].cpu().numpy(), axis=0)
    c_out = np.squeeze(c_out.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        p_out = p_out.astype(np.float32)
        c_out = c_out.astype(np.float32)
         # float32 with F mode, resize back to orig_size
        p_out = misc.imresize(p_out, orig_size, "nearest", mode="F")
        c_out = misc.imresize(c_out, orig_size, "nearest", mode="F")
    
    decoded_p_out = set_loader.decode_segmap(p_out)
    decoded_c_out = set_loader.decode_segmap(c_out)

    # clear and adversarial images and predictions
    fig, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(c_image)
    axarr[0,0].title.set_text('original image')
    axarr[0,1].imshow(decoded_c_out)
    axarr[0,1].title.set_text('original prediction')
    axarr[1,0].imshow(p_image)
    axarr[1,0].title.set_text('adversarial image')
    axarr[1,1].imshow(decoded_p_out)
    axarr[1,1].title.set_text('adversarial prediction')
    for ax in axarr.reshape(-1) : ax.set_axis_off()
    figname = os.path.join(path, "summary_patch%d_%d.png" % (count, img_num))
    fig.savefig(figname, bbox_inches='tight', dpi = 500) #high-quality
        
    print("summary_patch" + str(count) + "_" + str(img_num) + ".png" + " saved ")


    
#---------------------------------------------------------------------
# Basic implementation of the neighrest neighbors labels 
# substitution 
# params:
# - in_label: original target to modify
# - targets: list of target classes to be removed in the output label
#---------------------------------------------------------------------
def remove_target_class(label, attacked, target, scale=1, maxd=10):
    if target == -1:
        # nearest neighbor
        label = nearest_neighbor(label, attacked, scale=scale, maxd=maxd)
    elif target == -2:
        label = untargeted_labeling(label, attacked)
    else:
        print("Number of pixels labeled as class %d: %d" % (attacked, (label==attacked).sum()))
        label[label == attacked] = target
        print("Number of pixels labeled as class %d: %d" % (attacked, (label==attacked).sum()))
    return label.long()


"""
def nearest_neighbor(label, attacked, dmax=250, scale=1):
    device = label.device
    
    
    label = F.interpolate(label.float(), scale_factor=scale, mode='nearest')
    N, H, W = label.shape
    attacked_mask = (label == attacked)
    index = attacked_mask.nonzero(as_tuple=False)
    index_tuple = attacked_mask.nonzero(as_tuple=True)
#     I_ = torch.tensor(range(H), device=device, dtype=torch.uint8, requires_grad=False).reshape((H, 1, 1)).repeat_interleave(W, axis=1)
#     J_ = torch.tensor(range(W), device=device, dtype=torch.uint8, requires_grad=False).reshape((1, W, 1)).repeat_interleave(H, axis=0)
    I_ = torch.tensor(range(-maxd//2, maxd//2), device=device, dtype=torch.int, requires_grad=False).reshape((maxd, 1, 1)).repeat_interleave(maxd, axis=1)
    J_ = torch.tensor(range(-maxd//2, maxd//2), device=device, dtype=torch.int, requires_grad=False).reshape((1, maxd, 1)).repeat_interleave(maxd, axis=0)
    
#     mask = torch.zeros_like(label)
    for n in range(N):
        pixels_in_image = (index_tuple[0] == n).sum()
        print("Find %d pixels in image %d" % (pixels_in_image, n))
        pixel_index = (index_tuple[0] == n).nonzero(as_tuple=True)[0]
#         D = torch.zeros((H, W, pixels_in_image), device=device, dtype=torch.int, requires_grad=False)
#         I = pixel_index[pixels_in_image, 1].reshape((1, 1, pixels_in_image)) + torch.tensor(range(-maxd//2, maxd//2), dtype=torch.uint8, requires_grad=False).reshape((maxd, 1, 1))
#         J = pixel_index[:, 2] + torch.tensor(range(-maxd/2, maxd/2), dtype=torch.uint8, requires_grad=False).reshape((1, maxd, 1))
        I = I_.repeat_interleave(pixels_in_image, axis=2)
        J = J_.repeat_interleave(pixels_in_image, axis=2)
#         print(pixels_in_image)
#         print(pixel_index)
#         print(index[pixel_index, 1])
#         D_i = (index[pixel_index[0], 1].reshape((1, 1, pixels_in_image)) - I)**2
#         D_j = (index[pixel_index, 2].reshape((1, 1, pixels_in_image)) - J)**2
        D = I**2 + J**2
        fake_index = # TODO has to be limited.
        
#         D = (index[pixel_index, 1].reshape((1, 1, pixels_in_image)) - I)**2 + (index[pixel_index, 2].reshape((1, 1, pixels_in_image)) - J)**2
        # D = D_i + D_j
#         print(D.shape)
        
#         D[index_tuple[1:]] = 1e6
#         min_dist_i = torch.argmin(torch.argmin(D, axis=1), axis=0)
        
#         label[n, index_tuple[1:]] = label[n, ]
    label = F.interpolate(label, scale_factor=1./scale, interpolation='nearest')
    return label
"""

#---------------------------------------------------------------------
# Remove mask from a batch of images
#---------------------------------------------------------------------
def remove_mask (images,mask):
    mask = F.interpolate(mask, size=images.shape[1:],mode='bilinear', align_corners=True)
    images[mask.squeeze(1)==1] = 250
    return images



def nearest_neighbor(label, attacked, maxd=250, scale=1):
    device = label.device
    
#     label = F.interpolate(label.float(), scale_factor=scale, mode='nearest')
    N, H, W = label.shape
    attacked_mask = (label == attacked)
    index = attacked_mask.nonzero(as_tuple=False)
    index_tuple = attacked_mask.nonzero(as_tuple=True)
    
#     nearest = torch.zeros((index.shape[0]), device=device, dtype=torch.long, requires_grad=False)
#     count_pixel = 0
    for n in range(N):
        pixels_in_image = (index_tuple[0] == n).sum()
        pixel_index = (index_tuple[0] == n).nonzero(as_tuple=True)[0]
        print("Find %d pixels in image %d" % (pixels_in_image, n))
        for i in range(pixels_in_image):
#             print("Finding %d-th nearest neighbor" % i)
            pixel_center = index[pixel_index[i], 1:]
#             print("_________________")
#             print(pixel_center)
#             print(maxd)
#             print(pixel_center[0] - maxd//2)
#             print(pixel_center[0] + maxd//2)
            # Consider limited area around pixel center
            min_i, max_i = pixel_center[0] - maxd//2, pixel_center[0] + maxd//2
            min_j, max_j = pixel_center[1] - maxd//2, pixel_center[1] + maxd//2
    
            corners_i = (torch.clip(min_i, 0, H), torch.clip(max_i, 0, H))
            corners_j = (torch.clip(min_j, 0, W), torch.clip(max_j, 0, W))
#             print(pixel_center)
#             print(max_i)
#             print(corners_i)
            
#             elim_min_i, elim_max_i = min_i - corners_i[0], max_i - corners_i[1]
#             elim_min_j, elim_max_j = min_j - corners_j[0], max_j - corners_j[1]
#             print(corners_i)
#             print(corners_j)
            h_, w_ = corners_i[1] - corners_i[0], corners_j[1] - corners_j[0]
#             print(h_)
#             print(w_)
            I = torch.tensor(range(corners_i[0], corners_i[1]), device=device, dtype=torch.int, requires_grad=False).reshape((h_, 1)).repeat_interleave(w_, axis=1) - pixel_center[0]
#             print(I)
            J = torch.tensor(range(corners_j[0], corners_j[1]), device=device, dtype=torch.int, requires_grad=False).reshape((1, w_)).repeat_interleave(h_, axis=0) - pixel_center[1]
#             print(J)
            D = I**2 + J**2 + maxd**2 * torch.where(label==attacked, 1, 0)[n, corners_i[0]:corners_i[1], corners_j[0]: corners_j[1]]
            
            nearest_pix = (D==torch.min(D)).nonzero()[0] #[torch.argmin(D, axis=1)[0], torch.argmin(D, axis=0)[0]]
            nearest = [corners_i[0] + nearest_pix[0], corners_j[0] + nearest_pix[1]]
#             print(torch.argmin(D, axis=))
#             print(nearest)
#             print("_________________")
#             print(torch.argmin(D, axis=1))
#             print(torch.argmin(D, axis=0))
#             print(D)
#             print(nearest_pix)
#             print(nearest)
            label[n, pixel_center[0], pixel_center[1]] = label[n, nearest[0], nearest[1]]
            
            
#             labels_mod = torch.where(labels_mod==attacked, )
#             labels_considered = labels[corners_i[0]:corners_i[1], corners_j[0]:corners_j[1]]
            
            
        
            
            
            """
            found_nearest = False
            old_considered = pixel_center.clone()
            while d < 250 or not found_nearest:
                corners_i = (torch.clip(pixel_center[0] - d, 0, H), torch.clip(pixel_center[0] + d, 0, H))
                corners_j = (torch.clip(pixel_center[1] - d, 0, W), torch.clip(pixel_center[1] + d, 0, W))
                
                pixels_considered = label[n, corners_i[0]:corners_i[1]+1, corners_j[0]:corners_j[1]+1]
#                 pixels_considered = torch.cat((label[n, corners_i[0]:corners_i[1] + 1, corners_j[0]].reshape(-1), 
#                                               label[n, corners_i[0]:corners_i[1] + 1, corners_j[1]].reshape(-1)))
                print(pixels_considered.shape)
#                  = torch.cat((
#                     label[n, corners_i[0]:corners_i[1], corners_j[0]].reshape(-1)
#                     label[n, corners_i[0]:corners_i[1], corners_j[1]].reshape(-1)
#                     label[n, corners_i[0], corners_j[0] + 1: corners_j[1] - 1].reshape(-1)
#                     label[n, corners_i[1], corners_j[0] + 1: corners_j[1] - 1].reshape(-1)
#                 ))
                
                d = d + 1 + (d // 10)
                found_nearest = (pixels_considered != attacked).any()
                if found_nearest:
                    print("Found nearest")
                    print((pixels_considered != attacked).nonzero())
                    ll = label[n, (pixels_considered != attacked).nonzero()[0]]
                    print(ll)
                    label[n, pixel_center] = ll
           """ 
            
            
    return label


def untargeted_labeling(label, attacked):
    torch.where(label != attacked, label, 255)
#     label[label != attacked] = 255
    
    return label

    

'''
#-------------------------------------------------------------------
#
#-------------------------------------------------------------------
def change_labels_to_static(labels, static_label):
    print(labels)
    print(static_label)
    for i in range(labels.shape[0]):
        labels[i] = static_label
'''


'''
TO CHECK 
make a class for all the possible transformation

# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total

'''





'''
# ------------------------------------------------------------------
def create_img_mask(in_features, patch_mask):
    mask = torch.ones([3,in_features.size(1), in_features.size(2)])
    img_mask = mask - patch_mask
    return img_mask




# ------------------------------------------------------------------

def create_patch_mask(in_features, my_patch, cfg_ctrs):
    patch_size = 40 #TODO JUST NOW, TO BE CHANGED
    width = in_features.size(1)
    height = in_features.size(2)
    patch_mask = torch.zeros([3, width,height])

    p_w = patch_size + cfg.patch_x
    p_h = patch_size + cfg.patch_y
    patch_mask[:, int(cfg.patch_x):int(p_w), int(cfg.patch_y):int(p_h)]= 1

    return patch_mask
'''


'''
TO CHECK 
make a class for all the possible transformation

def transform_patch(width, x_shift, y_shift, im_scale, rot_in_degree):
    """
      If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
      then it maps the output point (x, y) to a transformed input point 
      (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
      where k = c0 x + c1 y + 1. 
      The transforms are inverted compared to the transform mapping input points to output points.
     """
    rot = float(rot_in_degree) /90. *(math.pi/2)
     
    # rotation matrix
    rot_matrix = np.array( 
                [[math.cos(rot), -math.sin(rot)],
                 [math.sin(rot), math.cos(rot)]] )

    # scale it
'''  



'''
TO CHECK 
'''
'''
def create_patch_mask_bbox(im_data, bbox, advpatch):
    width = im_data.size(1)
    height = im_data.size(2)
    patch_mask = torch.zeros([3,width,height])

    p_w = bbox[2]-bbox[0]
    p_h = bbox[3]-bbox[1]
    patch_mask[:, 0:p_w,0:p_h]=1
    return patch_mask
'''


'''
def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        # noise in the initial size (why???)
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch
'''


'''
#---------------------------------------------------------------------
# Basic implementation of the neighrest neighbors labels 
# substitution 
# params:
# - in_label: original target to modify
# - targets: list of target classes to be removed in the output label
#---------------------------------------------------------------------
def remove_target_class_with_NN(label, targets):
    i_t = 0
    for v1 in label:
        j_t = 0
        for e1 in v1:
            if(label[i_t][j_t] in target):
                # the element [i_t][j_t] correponds to a target class
                min_dist = 99999
                s_x, s_y = -1,-1
                    
                i = 0
                for vec in label:
                    j = 0
                    for elem in vec:
                        if(elem not in target):
                            distance = (i - i_t)**2 + (j - j_t)**2 
                            if(distance < min_dist):
                                min_dist = distance
                                s_x, s_y = j,i
                        j += 1
                    i += 1
                    
                label[i_t][j_t] = label[s_y][s_x]
                   
            j_t += 1
        i_t += 1
    return label
'''