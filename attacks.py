import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import models_utils

# Unnormalization function
def unnormalize(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(image.device)
    std = torch.tensor(std).view(-1, 1, 1).to(image.device)
    return image * std + mean

# Normalization function
def normalize(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(image.device)
    std = torch.tensor(std).view(-1, 1, 1).to(image.device)
    return (image - mean) / std


# Updated distance-based PGD attack function
def distance_based_pgd_attack(model, images, ground_truths, binary_masks, mean, std, epsilon=0.03, alpha=0.01, num_steps=40, patch_size=512, model_name = 'bisenetX39', random_selection=True):
    """
    Applies a distance-based PGD attack to an image tensor within the area specified by a binary mask.
    The attack focuses on fooling the model by applying a distance-based weighted approach for each patch.

    Args:
        model (torch.nn.Module): The segmentation model.
        image (torch.Tensor): The input image tensor of shape (C, H, W), normalized.
        ground_truth (torch.Tensor): The ground truth labels tensor of shape (H, W).
        binary_mask (torch.Tensor): The binary mask specifying regions to attack, shape (H, W).
        mean (list or tuple): Normalization mean for each channel.
        std (list or tuple): Normalization standard deviation for each channel.
        epsilon (float): The maximum allowed perturbation.
        alpha (float): The step size for the attack.
        num_steps (int): The number of iterations for PGD.
        patch_size (int): Size of the patches (K x K).

    Returns:
        torch.Tensor: The adversarially perturbed image.
    """

    # Loop for PGD iterations
    adv_images = []
    for (image, ground_truth, binary_mask) in zip(images, ground_truths, binary_masks):
        
        # Unnormalize the original image to work in the pixel space
        original_image_unnorm = unnormalize(image, mean, std).to(image.device)

        
        # Ensure binary_mask has the same device and dtype
        binary_mask = binary_mask.to(image.device).float()
    
        # Add batch dimension to the ground truth tensor
        ground_truth = ground_truth.unsqueeze(0).to(image.device)  # Shape: (1, H, W)
    
        # Calculate the distance-based weight for the fooling mask
        fooling_mask_tensor = torch.ones_like(binary_mask)
        #fooling_mask_tensor = 1 - binary_mask.clone()  # Fooling mask is the complement of the binary mask
        fooling_mask_tensor = fooling_mask_tensor.unsqueeze(0).to(image.device)  # Shape: (1, H, W)

        delta = torch.zeros_like(original_image_unnorm, requires_grad=True).to(image.device)
        
        for step in range(num_steps):
            # Zero out gradients before each step
            if delta.grad is not None:
                delta.grad.zero_()
    
            # Create adversarial image by adding perturbation
            adv_image_unnorm = original_image_unnorm + delta
            adv_image_unnorm = torch.clamp(adv_image_unnorm, 0, 1)  # Clamp to valid image range [0, 1]
    
            # Normalize the adversarial image before passing it to the model
            adv_image_norm = normalize(adv_image_unnorm, mean, std).to(image.device)
    
            # Forward pass for the entire image
            adv_image_norm = adv_image_norm.unsqueeze(0)  # Add batch dimension
            output = model(adv_image_norm)  # Output shape: (1, C, H, W)
            output = models_utils.output_handle(adv_image_norm, output, model_name)
    
            # Initialize an empty gradient tensor for aggregation
            full_gradient = torch.zeros_like(delta).to(image.device)
    
            # Loop over patches
            _, H, W = image.shape

            random_selection = random_selection
            if random_selection is True:
                # Randomly select one patch and compute the gradient for it
                total_patches_H = H // patch_size + (1 if H % patch_size != 0 else 0)
                total_patches_W = W // patch_size + (1 if W % patch_size != 0 else 0)
                
                # Randomly select a patch coordinate
                patch_idx_H = torch.randint(0, total_patches_H, (1,)).item()
                patch_idx_W = torch.randint(0, total_patches_W, (1,)).item()
                
                # Define patch bounds
                i_start = patch_idx_H * patch_size
                j_start = patch_idx_W * patch_size
                i_end = min(i_start + patch_size, H)
                j_end = min(j_start + patch_size, W)
                
                # Compute the loss for the selected patch
                loss_patch = F.cross_entropy(output, ground_truth, reduction='none', ignore_index=250)  # Shape: (1, H, W)
                attention_map = torch.zeros_like(fooling_mask_tensor).to(image.device)
                attention_map[:, i_start:i_end, j_start:j_end] = 1
                attention_map *= fooling_mask_tensor.detach()
                patch_loss_values = loss_patch[attention_map > 0]
                
                # Skip if no valid loss values
                if patch_loss_values.numel() > 0:
                    loss_patch = patch_loss_values.mean()
                
                    # Compute gradient of the loss w.r.t. the perturbation for this patch
                    model.zero_grad()  # Zero gradients from previous steps
                    delta.grad = None  # Reset gradients for delta
                    loss_patch.backward(retain_graph=True)
                
                    # Extract and normalize the gradient for the patch
                    grad_patch = delta.grad.clone() * binary_mask  # Extract the relevant gradient
                    if grad_patch.norm() != 0:  # Normalize the gradient if non-zero
                        grad_patch = grad_patch / grad_patch.norm()
                
                    # Accumulate the gradient into the full gradient tensor
                    full_gradient += grad_patch.clone()
            else:
                for i in range(0, H, patch_size):
                    for j in range(0, W, patch_size):
                        # Define patch bounds
                        i_end = min(i + patch_size, H)
                        j_end = min(j + patch_size, W)
        
                        # Compute the loss for the patch
                        loss_patch = F.cross_entropy(output, ground_truth, reduction='none', ignore_index=250)   # Shape: (1, H, W)
                        attention_map = torch.zeros_like(fooling_mask_tensor).to(image.device)
                        attention_map[:, i:i_end, j:j_end] = 1
                        attention_map *= fooling_mask_tensor.detach()
                        patch_loss_values = loss_patch[attention_map > 0]
        
                        if patch_loss_values.numel() == 0:
                            continue  # Skip if there are no values to compute
        
                        loss_patch = patch_loss_values.mean()
        
                        # Compute gradient of the loss w.r.t. the perturbation for this patch
                        model.zero_grad()  # Zero gradients from previous steps
                        delta.grad = None  # Reset gradients for delta
                        loss_patch.backward(retain_graph=True)
        
                        # Extract and normalize the gradient for the patch
                        grad_patch = delta.grad.clone() * binary_mask  # Extract the relevant gradient
                        if grad_patch.norm() != 0:  # Normalize the gradient if non-zero
                            grad_patch = grad_patch / grad_patch.norm()
        
                        # Place the normalized gradient back in the full gradient tensor
                        full_gradient += grad_patch.clone()
    
            # Apply the patch-wise normalized gradient to the perturbation
            delta = delta + alpha * full_gradient.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)  # Clamp the perturbation to the epsilon ball
            delta = delta.detach().requires_grad_(True)  # Detach and re-enable gradient computation
    
        # Compute final adversarial image in the unnormalized space
        adv_image_unnorm = original_image_unnorm + delta
        adv_image_unnorm = torch.clamp(adv_image_unnorm, 0, 1)  # Ensure valid image range
    
        # Normalize the final adversarial image before returning
        adv_image = normalize(adv_image_unnorm, mean, std)
        adv_images.append(adv_image.unsqueeze(0))
    adv_images = torch.cat(adv_images)

    return adv_images


def distance_based_pgd_attack_batch(model, images, ground_truths, binary_masks, mean, std, epsilon=0.03, alpha=0.01, num_steps=40, patch_size=512, model_name='bisenetX39', random_selection=True):
    """
    Applies a distance-based PGD attack to a batch of image tensors within areas specified by binary masks.
    The attack focuses on fooling the model using a distance-based weighted approach for each patch.

    Args:
        model (torch.nn.Module): The segmentation model.
        images (torch.Tensor): Batch of input images of shape (B, C, H, W), normalized.
        ground_truths (torch.Tensor): Batch of ground truth labels of shape (B, H, W).
        binary_masks (torch.Tensor): Batch of binary masks of shape (B, H, W).
        mean (list or tuple): Normalization mean for each channel.
        std (list or tuple): Normalization standard deviation for each channel.
        epsilon (float): The maximum allowed perturbation.
        alpha (float): The step size for the attack.
        num_steps (int): The number of iterations for PGD.
        patch_size (int): Size of the patches (K x K).

    Returns:
        torch.Tensor: Batch of adversarially perturbed images.
    """
    B, C, H, W = images.shape

    # Unnormalize the original images to work in the pixel space
    original_images_unnorm = unnormalize(images, mean, std).to(images.device)

    # Ensure binary_masks and ground_truths have the same device and dtype
    binary_masks = binary_masks.to(images.device).float().unsqueeze(1)
    ground_truths = ground_truths.to(images.device)  # Shape: (B, H, W)

    # Initialize perturbation tensor
    delta = torch.zeros_like(original_images_unnorm, requires_grad=True).to(images.device)

    for step in range(num_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        # Create adversarial images by adding perturbation
        adv_images_unnorm = original_images_unnorm + delta
        adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)  # Clamp to valid image range [0, 1]

        # Normalize the adversarial images before passing to the model
        adv_images_norm = normalize(adv_images_unnorm, mean, std).to(images.device)

        # Forward pass for the batch of adversarial images
        outputs = model(adv_images_norm)  # Output shape: (B, C, H, W)
        outputs = models_utils.output_handle(adv_images_norm, outputs, model_name)

        # Compute loss for the entire batch
        loss = F.cross_entropy(outputs, ground_truths, reduction='none', ignore_index=250) 

        # Apply masks and compute gradients for each patch
        full_gradient = torch.zeros_like(delta).to(images.device)

        if random_selection:
            # Randomly select patches for each image in the batch
            total_patches_H = H // patch_size + (1 if H % patch_size != 0 else 0)
            total_patches_W = W // patch_size + (1 if W % patch_size != 0 else 0)

            patch_idx_H = torch.randint(0, total_patches_H, (B,)).to(images.device)
            patch_idx_W = torch.randint(0, total_patches_W, (B,)).to(images.device)

            for b in range(B):
                i_start = patch_idx_H[b] * patch_size
                j_start = patch_idx_W[b] * patch_size
                i_end = min(i_start + patch_size, H)
                j_end = min(j_start + patch_size, W)

                attention_map = torch.zeros_like(loss[b:b+1]).to(images.device)  # Shape: (1, H, W)
                attention_map[:, i_start:i_end, j_start:j_end] = 1
                attention_map *= binary_masks[b:b+1]

                patch_loss_values = loss[b:b+1][attention_map > 0]
                if patch_loss_values.numel() > 0:
                    patch_loss = patch_loss_values.mean()
                    patch_loss.backward(retain_graph=True)

                    grad_patch = delta.grad[b].clone() * binary_masks[b]
                    if grad_patch.norm() != 0:
                        grad_patch = grad_patch / grad_patch.norm()

                    full_gradient[b] += grad_patch
        else:
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    i_end = min(i + patch_size, H)
                    j_end = min(j + patch_size, W)

                    attention_map = torch.zeros_like(loss).to(images.device) 
                    attention_map[:, i:i_end, j:j_end] = 1
                    attention_map *= (1-binary_masks.squeeze(1))


                    patch_loss_values = loss * attention_map
                    if patch_loss_values.numel() > 0:
                        model.zero_grad()  # Zero gradients from previous steps
                        delta.grad = None  # Reset gradients for delta
                        patch_loss = patch_loss_values.mean()
                        patch_loss.backward(retain_graph=True)

                        grad_patch = delta.grad.clone() * binary_masks
                        norm_val = grad_patch.abs().sum(dim=(1,2,3), keepdim=True)/binary_masks.sum(dim=(1,2,3), keepdim=True)
                        #norm_val = grad_patch.view(B,-1).sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                        if norm_val.sum() > 0:
                            grad_patch = grad_patch / norm_val
                        full_gradient += grad_patch
                    else:
                        print("bubulala")


        # Update delta with the aggregated gradients
        delta = delta + alpha * full_gradient.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = delta.detach().requires_grad_(True)

    adv_images_unnorm = original_images_unnorm + delta
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)
    adv_images = normalize(adv_images_unnorm, mean, std)

    return adv_images

def distance_based_pgd_attack_batch_2(model, images, ground_truths, binary_masks, area_to_attack, mean, std, epsilon=0.03, alpha=0.01, num_steps=40, patch_size=512, model_name='bisenetX39', random_selection=True):
    """
    Applies a distance-based PGD attack to a batch of image tensors within areas specified by binary masks.
    The attack focuses on fooling the model using a distance-based weighted approach for each patch.

    Args:
        model (torch.nn.Module): The segmentation model.
        images (torch.Tensor): Batch of input images of shape (B, C, H, W), normalized.
        ground_truths (torch.Tensor): Batch of ground truth labels of shape (B, H, W).
        binary_masks (torch.Tensor): Batch of binary masks of shape (B, H, W).
        mean (list or tuple): Normalization mean for each channel.
        std (list or tuple): Normalization standard deviation for each channel.
        epsilon (float): The maximum allowed perturbation.
        alpha (float): The step size for the attack.
        num_steps (int): The number of iterations for PGD.
        patch_size (int): Size of the patches (K x K).

    Returns:
        torch.Tensor: Batch of adversarially perturbed images.
    """
    B, C, H, W = images.shape

    # Unnormalize the original images to work in the pixel space
    original_images_unnorm = unnormalize(images, mean, std).to(images.device)

    # Ensure binary_masks and ground_truths have the same device and dtype
    binary_masks = binary_masks.to(images.device).float().unsqueeze(1)
    ground_truths = ground_truths.to(images.device)  # Shape: (B, H, W)

    # Get original predictions before adversarial perturbation
    with torch.no_grad():
        original_outputs = model(normalize(original_images_unnorm, mean, std))  # Shape: (B, C, H, W)
        original_outputs = models_utils.output_handle(original_images_unnorm, original_outputs, model_name)
        original_predicted_classes = original_outputs.argmax(dim=1)  # Shape: (B, H, W)


    # Initialize perturbation tensor
    delta = torch.zeros_like(original_images_unnorm, requires_grad=True).to(images.device)
    for step in range(num_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        # Create adversarial images by adding perturbation
        adv_images_unnorm = original_images_unnorm + delta
        adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)  # Clamp to valid image range [0, 1]

        # Normalize the adversarial images before passing to the model
        adv_images_norm = normalize(adv_images_unnorm, mean, std).to(images.device)

        # Forward pass for the batch of adversarial images
        outputs = model(adv_images_norm)  # Output shape: (B, C, H, W)
        outputs = models_utils.output_handle(adv_images_norm, outputs, model_name)

        # Compute loss for the entire batch
        loss = F.cross_entropy(outputs, ground_truths, reduction='none', ignore_index=250)  # Shape: (B, H, W)

        # Initialize tensors for correct and incorrect classification masks
        predicted_classes = outputs.argmax(dim=1)  # Shape: (B, H, W)
        correct_mask = (predicted_classes == ground_truths) & (ground_truths != 250)  # Ignore index mask
        
        incorrect_mask = (
            (predicted_classes != ground_truths) &
            (ground_truths != 250) &
            (original_predicted_classes == ground_truths))

        avoid_mask = area_to_attack 

        # Compute losses for correctly and incorrectly classified pixels
        correct_loss = (loss * correct_mask * avoid_mask).sum(dim=(1, 2)) / correct_mask.sum(dim=(1, 2)).clamp(min=1)
        incorrect_loss = (loss * incorrect_mask * avoid_mask).sum(dim=(1, 2)) / incorrect_mask.sum(dim=(1, 2)).clamp(min=1)

        

        correct_loss = correct_loss.mean()
        incorrect_loss = incorrect_loss.mean()

        # Compute gradients for correct and incorrect losses
        kappa = 0.5
        total_loss =  kappa *correct_loss + (1-kappa) * incorrect_loss
        model.zero_grad()
        total_loss.backward(retain_graph=True)

        grad = delta.grad.clone() * binary_masks
        if kappa != 0.5:
            norm = grad.abs().sum(dim=(1, 2, 3), keepdim=True) / binary_masks.sum(dim=(1, 2, 3), keepdim=True)
            grad = grad / norm.clamp(min=1e-8)

        # Update delta with the aggregated gradients
        delta = delta + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = delta.detach().requires_grad_(True)

    adv_images_unnorm = original_images_unnorm + delta
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)
    adv_images = normalize(adv_images_unnorm, mean, std)

    return adv_images



# The original localized PGD attack function with epsilon and normalization handling
def localized_pgd_attack(model, images, ground_truths, binary_masks, mean, std, epsilon=0.03, alpha=0.01, num_steps=40, fooling_mask=None,  model_name = 'bisenetX39'):
    """
    Applies a localized PGD attack to an image tensor within the area specified by a binary mask.
    The attack is optimized to fool the region specified by the fooling mask.
    
    Args:
        model (torch.nn.Module): The segmentation model.
        image (torch.Tensor): The input image tensor of shape (C, H, W), normalized.
        ground_truth (torch.Tensor): The ground truth labels tensor of shape (H, W).
        binary_mask (torch.Tensor): The binary mask specifying regions to attack, shape (H, W).
        mean (list or tuple): Normalization mean for each channel.
        std (list or tuple): Normalization standard deviation for each channel.
        epsilon (float): The maximum allowed perturbation.
        alpha (float): The step size for the attack.
        num_steps (int): The number of iterations for PGD.
        fooling_mask (str or torch.Tensor or None): The binary mask specifying regions to fool, shape (H, W).
            If None, the entire ground truth is used. If "out_of_perturbed", it is the opposite of the binary mask.
            If "distance_based", it computes a weighted distance-based fooling mask.
    
    Returns:
        torch.Tensor: The adversarially perturbed image.
    """

    # Make a copy of the normalized image to modify
    original_image = images.clone().detach()  # Store the original image for epsilon ball constraint
    original_image_unnorm = unnormalize(images, mean, std).to(images.device)
    adv_image = images.clone().detach().requires_grad_(True).to(images.device)
    
    # Ensure binary_mask has the same device and dtype
    binary_mask = binary_masks.to(images.device).float().unsqueeze(1)

    # Add batch dimension to the ground truth and image tensors
    ground_truth = ground_truths.to(images.device)#.unsqueeze(0)  # Shape: (1, H, W)

    delta = torch.zeros_like(original_image_unnorm, requires_grad=True).to(images.device)

    # Handle the different fooling mask scenarios
    if fooling_mask is None:
        fooling_mask_tensor = torch.ones_like(binary_mask).to(images.device)
    elif fooling_mask == "out_of_perturbed":
        fooling_mask_tensor = 1 - binary_mask
    elif fooling_mask == "distance_based":
        binary_mask_np = binary_mask.cpu().numpy()
        distance_map = distance_transform_edt(1 - binary_mask_np)  # Compute distance from 0s to 1s
        fooling_mask_np = 1 + distance_map  # Increase importance for more distant regions
        fooling_mask_tensor = torch.from_numpy(fooling_mask_np).float().to(images.device)
    elif isinstance(fooling_mask, torch.Tensor):
        fooling_mask_tensor = fooling_mask.to(images.device).float()
    else:
        raise ValueError("Invalid fooling mask type")

    fooling_mask_tensor = fooling_mask_tensor.unsqueeze(0)  # Shape: (1, H, W)

    # Loop for PGD iterations
    for step in range(num_steps):
        # Unnormalize the adversarial image before applying perturbation
        adv_image_unnorm = unnormalize(adv_image, mean, std)

        # Forward pass through the model with the normalized image
        adv_image_norm = normalize(adv_image_unnorm, mean, std)
        output = model(adv_image_norm)  # Add batch dimension: (1, C, H, W)
        output = models_utils.output_handle(adv_image_norm, output, model_name)
        #output = output.squeeze(0)  # Remove batch dimension: (C, H, W)

        # Compute the loss
        loss = (F.cross_entropy(output, ground_truth, reduction='none', ignore_index=250) * fooling_mask_tensor).mean()

        # Compute gradients of loss w.r.t. the adversarial image
        loss.backward()

        # Generate the gradient sign
        grad_sign = adv_image.grad.data.sign()

        # Apply the localized perturbation using the binary mask
        localized_grad = grad_sign * binary_mask

        delta = delta + alpha * localized_grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)  # Clamp the perturbation to the epsilon ball
        delta = delta.detach().requires_grad_(True)  # Detach and re-enable gradient computation
    
        # Compute final adversarial image in the unnormalized space
        adv_image_unnorm = original_image_unnorm + delta
        adv_image_unnorm = torch.clamp(adv_image_unnorm, 0, 1)  # Ensure valid image range
    
        # Normalize the final adversarial image before returning
        #adv_image = normalize(adv_image_unnorm, mean, std)

        # Update the adversarial image with the step size
        #adv_image_unnorm = adv_image_unnorm + alpha * localized_grad
        #adv_image_unnorm = torch.clamp(adv_image_unnorm, 0, 1)  # Clamp to valid image range [0, 1]

        # Project the adversarial image back to the epsilon ball around the original image
        #delta = torch.clamp(adv_image_unnorm - unnormalize(original_image, mean, std), -epsilon, epsilon)
        #adv_image_unnorm = torch.clamp(unnormalize(original_image, mean, std) + delta, 0, 1)

        #print(adv_image_unnorm.sum())

        # Renormalize the adversarial image for the next step
        adv_image = normalize(adv_image_unnorm, mean, std).detach().requires_grad_(True)
    return adv_image

#------------------------------------------------------------
#------------------------------------------------------------


# Wrapper function to select and execute the attack type
def attack_image(model, image, ground_truth, binary_mask, attack_type, attacked_area, mean, std, epsilon=0.03, alpha=0.01, num_steps=40, model_name = 'bisenetX39'):
    """
    A wrapper function to apply different types of attacks on an image.

    Args:
        model (torch.nn.Module): The segmentation model.
        image (torch.Tensor): The input image tensor of shape (C, H, W), normalized.
        ground_truth (torch.Tensor): The ground truth labels tensor of shape (H, W).
        binary_mask (torch.Tensor): The binary mask specifying regions to attack, shape (H, W).
        attack_type (str): Type of attack ("fgsm", "pgd_standard", "pgd_out_patch", "pgd_distance_based").
        mean (list or tuple): Normalization mean for each channel.
        std (list or tuple): Normalization standard deviation for each channel.
        epsilon (float): The maximum allowed perturbation.
        alpha (float): The step size for the attack.
        num_steps (int): The number of iterations for PGD (ignored for FGSM).

    Returns:
        torch.Tensor: The adversarially perturbed image.
    """
    if attack_type == "fgsm":
        # FGSM is equivalent to one-step PGD with fooling_mask = None
        return localized_pgd_attack(model, image, ground_truth, binary_mask, mean, std, epsilon, alpha=epsilon, num_steps=1, fooling_mask=None, model_name=model_name)
    elif attack_type == "pgd_standard":
        # PGD standard with fooling_mask = None
        return localized_pgd_attack(model, image, ground_truth, binary_mask, mean, std, epsilon, alpha, num_steps, fooling_mask=None, model_name=model_name)
    elif attack_type == "pgd_out_patch":
        # PGD with fooling mask = out_of_perturbed
        return localized_pgd_attack(model, image, ground_truth, binary_mask, mean, std, epsilon, alpha, num_steps, fooling_mask="out_of_perturbed", model_name=model_name)
    elif attack_type == "pgd_distance_based":
        # PGD with fooling mask = distance_based
        # distance_based_pgd_attack
        return distance_based_pgd_attack_batch_2(model, image, ground_truth, binary_mask, attacked_area, mean, std, epsilon=epsilon, alpha=alpha, num_steps=num_steps, patch_size=128, model_name = model_name, random_selection=False)
    else:
        raise ValueError(f"Invalid attack type: {attack_type}")



