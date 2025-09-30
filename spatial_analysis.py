import models_utils
import torch

from attacks import distance_based_pgd_attack_batch_2


#----------------------------------------------------------------
def generate_heatmap(model, input_image, target_area_mask, model_name):
    input_image.requires_grad = True
    model.eval()
    output = model(input_image)  
    output = model_utils.ouput(input_image, output, model_name)
    target_area_mask = target_area_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    target_output = output.max(1)[0] * target_area_mask
    target_score = target_output.mean()
    model.zero_grad()
    target_score.backward()
    grads = input_image.grad.data.clone()
    cam = grads.squeeze(0).sum(0)
    cam += cam.min()
    cam /= cam.max()
    cam = cam**2
    cam = cam.detach().cpu().numpy()
    return cam


#----------------------------------------------------------------
def get_scores(model, input_image, target_area_mask, model_name, labels):
    input_image.requires_grad = True
    model.eval()
    output = model(input_image)  
    output = models_utils.output_handle(input_image, output, model_name)
    target_area_mask = target_area_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    #softmax_output = torch.nn.functional.softmax(output, dim=1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=250)
    pixel_wise_loss = cross_entropy_loss(output, labels)  # Pixel-wise 
    mask = output.max(dim=1)[0]
    mask[mask < 0.] = 0.
    count_label = ((labels != 250)*target_area_mask).sum()
    val_ = ((output.max(dim=1)[1] == labels)*target_area_mask).sum()/count_label  #(pixel_wise_loss*target_area_mask).sum()
    loss_output = pixel_wise_loss * target_area_mask
    target_score = loss_output.sum()
    model.zero_grad()
    target_score.backward()
    grads = input_image.grad.data.clone()
    cam = grads.squeeze(0).abs().sum(0)
    cam_noabs = grads.squeeze(0).sum(0)

    int_area = target_area_mask
    int_cam = ((cam * int_area).sum() / (int_area.sum()))
    int_cam_noabs = (cam_noabs * int_area).mean() 

    ext_area = (1-target_area_mask)
    ext_cam = (cam * ext_area).sum() / (ext_area.sum())
    ext_cam_noabs = (cam_noabs * ext_area).mean() 
    
    return int_cam, ext_cam, int_cam_noabs, ext_cam_noabs, val_.item()



#----------------------------------------------------------------
def get_scores_attacks(model, input_image, target_area_mask, fooling_area, model_name, labels, mean, std, epsilon, alpha, num_steps):
    input_image.requires_grad = True
    model.eval()

    # apply attack here
    adv_image = distance_based_pgd_attack_batch_2(model, input_image, labels, target_area_mask,fooling_area, mean, std, epsilon=epsilon, alpha=alpha, num_steps=num_steps, model_name=model_name)


    output = model(adv_image)  
    output = models_utils.output_handle(input_image, output, model_name)
    #target_area_mask = target_area_mask.unsqueeze(1)  # Shape: (1, 1, H, W)
    
    #softmax_output = torch.nn.functional.softmax(output, dim=1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=250)
    pixel_wise_loss = cross_entropy_loss(output, labels)  # Pixel-wise 
    count_label = ((labels != 250)*fooling_area).sum()
    val_ = ((output.max(dim=1)[1] == labels)*fooling_area).sum()/count_label#(pixel_wise_loss*target_area_mask).sum()
    loss_output = pixel_wise_loss * fooling_area
    target_score = loss_output.sum()
    model.zero_grad()
    target_score.backward()
    grads = input_image.grad.data.clone()
    cam = grads.squeeze(0).abs().sum(0)

    int_area = fooling_area
    int_cam = ((cam * int_area).sum() / (int_area.sum()))

    ext_area = target_area_mask
    ext_cam = (cam * ext_area).sum() / (ext_area.sum())
    
    return int_cam, ext_cam, val_
