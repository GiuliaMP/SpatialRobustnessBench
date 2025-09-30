import os
import torch
import numpy as np
#import scipy.misc as m
import imageio
import random
import cv2
from torch.utils import data
from torchvision.transforms.functional import crop
import torch.nn.functional as F

from ptsemseg.utils import recursive_glob
from ptsemseg.loader.base_cityscapes_loader import baseCityscapesLoader
from ptsemseg.augmentations import imagenet_c



class cityscapesSRLoader(baseCityscapesLoader):

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(1024, 2048),
        augmentations=None,
        img_norm=True,
        version=None,
        bgr = True, 
        std_version = None,
        bottom_crop = 0,
        images_base_set = False,
        augmentation_params = None
    ):
        super(cityscapesSRLoader, self).__init__(
                                                root,
                                                split=split,
                                                is_transform=is_transform,
                                                img_size=img_size,
                                                augmentations=augmentations,
                                                img_norm=img_norm,
                                                version=version,
                                                bgr = bgr, 
                                                std_version = std_version,
                                                bottom_crop = bottom_crop,
                                                images_base_set = images_base_set
                                            )


        self.patch_size = augmentation_params["patch_size"] if augmentation_params is not None else None
        self.perturbation_ratio = augmentation_params["perturbation_ratio"] if augmentation_params is not None else None
        self.perturbation_type = augmentation_params["perturbation_type"] if augmentation_params is not None else None
        self.allow_mixed_perturbations = augmentation_params["allow_mixed_perturbations"] if augmentation_params is not None else None
        self.severity = augmentation_params["severity_level"] if augmentation_params is not None else None
            
        # list of available augmentations
        self.list_augs = imagenet_c.return_list_transformations()
        
    

    def get_list_augs(self):
        return self.list_augs

    
    #------------------------------------------------------------------
    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
    
        # Read image
        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        orig_img = img.copy()
        img_width, img_height, _ = img.shape

        if self.perturbation_type is not None: 
            # Determine which patches to perturb
            patch_ids = self.get_patches_to_perturb(img_width, img_height, self.patch_size, self.perturbation_ratio)
            # Apply perturbations and get the binary mask
            img, binary_mask = self.apply_perturbations(img, patch_ids, self.patch_size, self.perturbation_type)
        else:
            binary_mask = None

        
    
        # Read label
        lbl = imageio.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
    
        # Apply any additional augmentations if specified
        # if self.augmentations is not None:
        #     img, lbl = self.augmentations(img, lbl)
    
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
            orig_img,_ = self.transform(orig_img, None)


        if binary_mask is None: 
            binary_mask = torch.zeros((1, img.shape[1], img.shape[2]))
        else:
            binary_mask = cv2.resize(binary_mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
    
        return img, orig_img, lbl, binary_mask



    def __len__(self):
        return len(self.cityscapes_dataset)


    #------------------------------------------------------------------
    def get_patches_to_perturb(self, img_width, img_height, patch_size, perturbation_ratio):
        """Step 1: Select the patches that should be perturbed."""
        patch_width, patch_height = patch_size

        # Calculate the number of patches along width and height
        num_patches_x = img_width // patch_width
        num_patches_y = img_height // patch_height

        # Total number of patches
        total_patches = num_patches_x * num_patches_y
        # Number of patches to perturb based on the ratio
        num_patches_to_perturb = int(total_patches * perturbation_ratio)

        # Generate list of patch IDs to perturb
        all_patches = [(x, y) for x in range(num_patches_x) for y in range(num_patches_y)]
        patch_ids_to_perturb = random.sample(all_patches, num_patches_to_perturb)
        return patch_ids_to_perturb


    #------------------------------------------------------------------
    def apply_perturbations(self, img, patch_ids, patch_size, perturbation_types):
        """Step 2: Apply perturbations to the selected patches and create binary mask."""
        patch_width, patch_height = patch_size
        img_height, img_width, _ = img.shape  # Assuming img is in (H, W, C) format
    
        # Create a binary mask initialized to zeros
        binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
        for patch_id in patch_ids:
            x_idx, y_idx = patch_id
            x_start, y_start = x_idx * patch_width, y_idx * patch_height
    
            # Ensure that the patch boundaries do not exceed the image size
            x_end = x_start + patch_width
            y_end = y_start + patch_height
    
            # Extract the region to be perturbed
            patch = img[x_start:x_end, y_start:y_end]  
    
            # Determine the perturbation to apply
            if len(perturbation_types) == 1:
                perturbation_fn = self.get_perturbation_fn(perturbation_types[0])
            else:
                perturbation_fn = self.get_perturbation_fn(random.choice(perturbation_types))
    
            # Apply perturbation
            #perturbed_patch = perturbation_fn(patch, self.severity)

            if self.severity == 0: 
                perturbed_patch = patch
            else:
                perturbed_patch = perturbation_fn(patch, self.severity)
    
            # Place perturbed patch back into the original image
            img[x_start:x_end, y_start:y_end] = perturbed_patch
    
            # Update the binary mask to indicate the perturbed area
            binary_mask[x_start:x_end, y_start:y_end] = 1
    
        return img, binary_mask

    def apply_perturbations_old(self, img, patch_ids, patch_size, perturbation_types):
        """
        Step 2: Apply a global perturbation to the image, then selectively apply 
        perturbed patches based on the mask and create a binary mask.
        """
        patch_height, patch_width = patch_size
        img_height, img_width, _ = img.shape  # Assuming img is in (H, W, C) format
    
        # Create a binary mask initialized to zeros
        binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
        # Apply a global perturbation to the entire image
        if len(perturbation_types) == 1:
            perturbation_fn = self.get_perturbation_fn(perturbation_types[0])
        else:
            perturbation_fn = self.get_perturbation_fn(random.choice(perturbation_types))

        if self.severity == 0: 
            perturbed_img = img
        else:
            perturbed_img = perturbation_fn(img, self.severity)
    
        # Iterate over the selected patches and apply the perturbation
        for patch_id in patch_ids:
            x_idx, y_idx = patch_id
            x_start, y_start = x_idx * patch_width, y_idx * patch_height
    
            # Ensure that the patch boundaries do not exceed the image size
            x_end = min(x_start + patch_width, img_width)
            y_end = min(y_start + patch_height, img_height)
    
            # Replace the corresponding patch in the original image with the perturbed patch
            img[x_start:x_end, y_start:y_end] = perturbed_img[x_start:x_end, y_start:y_end]
    
            # Update the binary mask to indicate the perturbed area
            binary_mask[x_start:x_end, y_start:y_end] = 1
    
        return img, binary_mask


    #------------------------------------------------------------------
    def get_perturbation_fn(self, perturbation_name):
        # Check if perturbation list is properly defined
        if not hasattr(self, "list_augs") or not isinstance(self.list_augs, dict):
            raise AttributeError("The attribute 'list_augs' must be a dictionary of perturbations.")
    
        # Check if the perturbation_name is in the list of augmentations
        if perturbation_name in self.list_augs:
            return self.list_augs[perturbation_name]
        else:
            # Provide a helpful message listing the available perturbations
            available_perturbations = ", ".join(self.list_augs.keys())
            raise ValueError(
                f"Unknown perturbation type: '{perturbation_name}'. "
                f"Available perturbations are: {available_perturbations}."
            )

    


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])



