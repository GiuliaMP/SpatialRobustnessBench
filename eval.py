from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.metrics import runningScoreSpatial
from scipy.ndimage import distance_transform_edt
from ptsemseg.utils import convert_state_dict
from ptsemseg.utils import get_model_state
from attacks import attack_image
import torch
import models_utils

from transformers import AutoImageProcessor

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
def validate(model, dataloader, device, test_mode = False):
    n_classes = dataloader.dataset.n_classes
    running_metrics = runningScore(n_classes)
    running_metrics_clean = runningScore(n_classes)
    
    # Setup Model    
    model.to(device)
    model.eval()

    # Initialize tqdm for progress tracking
    total_batches = len(dataloader)
    times = []

    for i, (images, orig_images, labels, p_mask) in enumerate(dataloader):
        # Measure time per batch
        print(images.shape)
        with torch.no_grad():
            images = images.to(device)
            image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
            outputs = image_processor.post_process_semantic_segmentation(model(images), target_sizes=[(images.shape[2], images.shape[3]) for a in range(images.shape[0])])
            outputs = torch.cat(outputs).view(images.shape[0], images.shape[2], images.shape[3]).cpu().numpy()
            pred = outputs

            if isinstance(labels, list):
                labels = labels[0]
            gt = labels.numpy()

            running_metrics.update(gt, pred)

        if test_mode:
            break
            
    # Calculate final scores
    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    for i in range(n_classes):
        print(i, class_iou[i])
    
    return running_metrics


#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------

# Set weight parameter to control the influence of distance
weight = 0.005

# Function to compute gravity-inspired heatmap with a distance weight parameter
def compute_gravity_heatmap(binary_mask, weight=0.005):
    # Convert binary mask to numpy for distance transform
    binary_mask_np = binary_mask.cpu().numpy()
    # Compute distance transform (distance from each zero pixel to the nearest one pixel)
    distance_map = distance_transform_edt(1 - binary_mask_np)  # EDT for the inverse mask
    # Compute the heatmap: H(x, y) = 1 / (1 + w * D(x, y))
    heatmap_np = 1 / (1 + weight * distance_map)
    # Convert back to torch tensor
    heatmap = torch.from_numpy(heatmap_np).float()
    return 1-heatmap



#def process_output_by_models(outputs, name):
    


def validate_spatial(model, dataloader, device, test_mode = True, model_name = 'bisenetX39', n_classes = None):
    if n_classes is None:
        n_classes = dataloader.dataset.n_classes

    scores = {
        'normal': {
            'corrupted': runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': None
            },

         'local': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'local'
            },

         'spatial': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'spatial'
            },
    }
    
    
    # Setup Model    
    model.to(device)
    model.eval()

    # Initialize tqdm for progress tracking
    total_batches = len(dataloader)
    for i, (images, orig_images, labels, p_mask) in enumerate(dataloader):            
        
                
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = models_utils.output_handle(images, outputs, model_name)
            pred = outputs.data.max(1)[1].cpu().numpy()

            if isinstance(labels, list):
                labels = labels[0]
            gt = labels.numpy()

            # Compute the gravity heatmap with the weight parameter
            for types_metrics in scores.keys():
                spatial_importance = []
                p_mask = p_mask.to(device)

                if scores[types_metrics]['spatial_mode'] is None:
                    spatial_importance = torch.ones_like(p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'spatial':
                    spatial_importance = (1-p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'local':
                    spatial_importance = p_mask
                
   
                scores[types_metrics]['corrupted'].update(gt, pred, (spatial_importance).view(images.shape[0],-1).cpu().numpy())

            #---------------------------------
            orig_images = orig_images.to(device)
            outputs = model(orig_images)
            outputs = models_utils.output_handle(orig_images, outputs, model_name)
            pred = outputs.data.max(1)[1].cpu().numpy()

            if isinstance(labels, list):
                labels = labels[0]
            gt = labels.numpy()

            # Compute the gravity heatmap with the weight parameter
            for types_metrics in scores.keys():
                spatial_importance = []
                p_mask = p_mask.to(device)

                if scores[types_metrics]['spatial_mode'] is None:
                    spatial_importance = torch.ones_like(p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'spatial':
                    spatial_importance = (1-p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'local':
                    spatial_importance = p_mask
                
   
                scores[types_metrics]['clean'].update(gt, pred, (spatial_importance).view(images.shape[0],-1).cpu().numpy())

        if test_mode and i > 5:
            break
            

    # Calculate final scores
    #score, class_iou = running_metrics.get_scores()
    #score, class_iou = running_metrics_clean.get_scores()
    #for k, v in score.items():
        #print(k, v)
    #for i in range(n_classes):
        #print(i, class_iou[i])
    
    return scores



def eval_attacks(model, dataloader, device, test_mode = True,
                 attack_type=None, mean=None, std=None, epsilon=0.03, alpha=0.01, 
                 attack_position='center', attack_size=(50, 50), num_steps=1, batch_size = 1,  model_name = 'bisenetX39', n_classes = None):
    
    if n_classes is None: 
        n_classes = dataloader.dataset.n_classes
    #running_metrics = runningScoreSpatial(n_classes)
    #running_metrics_original = runningScoreSpatial(n_classes)

    scores = {
        'normal': {
            'corrupted': runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': None
            },

         'local': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'local'
            },

         'spatial': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'spatial'
            },
    }
    
    # Setup Model    
    model.to(device)
    model.eval()

    #total_batches = len(dataloader)
    batch_size, img_height, img_width  = batch_size, dataloader.dataset[0][0].shape[1], dataloader.dataset[0][0].shape[2]
    for i, (images, orig_images, labels, _) in enumerate(dataloader):          

        p_mask = []
        
        for _ in range(batch_size):
            mask = torch.zeros((img_height, img_width), device=device)

            if attack_position == 'center':
                center_y, center_x = img_height // 2, img_width // 2
                start_y = max(center_y - attack_size[0] // 2, 0)
                start_x = max(center_x - attack_size[1] // 2, 0)
            elif attack_position == 'left_corner':
                start_y, start_x = 0, 0
            elif attack_position == 'random':
                start_y = torch.randint(0, max(1, img_height - attack_size[0] + 1), (1,)).item()
                start_x = torch.randint(0, max(1, img_width - attack_size[1] + 1), (1,)).item()
            else:
                raise ValueError(f"Invalid attack_position: {attack_position}")

            end_y = min(start_y + attack_size[0], img_height)
            end_x = min(start_x + attack_size[1], img_width)
            mask[start_y:end_y, start_x:end_x] = 1
            p_mask.append(mask)
        p_mask = torch.stack(p_mask)
        
        images = images.to(device)
        attacked_image = attack_image(model = model, 
                                      image = images, 
                                      ground_truth = labels, 
                                      binary_mask = p_mask, 
                                      attack_type = attack_type, 
                                      attacked_area = (1-p_mask),
                                      mean = mean, 
                                      std = std, 
                                      epsilon = epsilon, 
                                      alpha = alpha, 
                                      num_steps = num_steps, 
                                      model_name = model_name)
        attacked_image = attacked_image.to(device)
        with torch.no_grad():
            outputs = model(attacked_image)
            outputs = models_utils.output_handle(images, outputs, model_name)
            pred = outputs.data.max(1)[1].cpu().numpy()

            if isinstance(labels, list):
                labels = labels[0]
            gt = labels.numpy()

            for types_metrics in scores.keys():
                spatial_importance = []
                p_mask = p_mask.to(device)

                if scores[types_metrics]['spatial_mode'] is None:
                    spatial_importance = torch.ones_like(p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'spatial':
                    spatial_importance = (1-p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'local':
                    spatial_importance = p_mask
                scores[types_metrics]['corrupted'].update(gt, pred, (spatial_importance).view(images.shape[0],-1).cpu().numpy())

            print(attacked_image.mean())


            #---------------------------------
            orig_images = orig_images.to(device)
            outputs = model(orig_images)
            outputs = models_utils.output_handle(orig_images, outputs, model_name)
            pred = outputs.data.max(1)[1].cpu().numpy()

            if isinstance(labels, list):
                labels = labels[0]
            gt = labels.numpy()

            #running_metrics_original.update(gt, pred, (spatial_importance).view(orig_images.shape[0],-1).cpu().numpy())

            for types_metrics in scores.keys():
                spatial_importance = []
                p_mask = p_mask.to(device)

                if scores[types_metrics]['spatial_mode'] is None:
                    spatial_importance = torch.ones_like(p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'spatial':
                    spatial_importance = (1-p_mask)
                elif scores[types_metrics]['spatial_mode'] == 'local':
                    spatial_importance = p_mask
                scores[types_metrics]['clean'].update(gt, pred, (spatial_importance).view(images.shape[0],-1).cpu().numpy())


        if test_mode and i > -1:
            break
            

    # Calculate final scores
    #score, class_iou = running_metrics.get_scores()
    #for k, v in score.items():
        #print(k, v)
    #for i in range(n_classes):
        #print(i, class_iou[i])
    
    return scores


#--------------------------------------------------------------
# Ripetitive Attacks
#--------------------------------------------------------------
def eval_attacks_ripetitive(model, dataloader, device, test_mode = True,
                 attack_type=None, mean=None, std=None, epsilon=0.03, alpha=0.01, attack_position='center', attack_size=(50, 50),
                            num_steps=1, model_name='bisenetX39', max_iterations=5, n_classes = None):

    from attacks import normalize, unnormalize
    if n_classes is None: 
        n_classes = dataloader.dataset.n_classes

    scores = {
        'normal': {
            'corrupted': runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': None
            },

         'local': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'local'
            },

         'spatial': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'spatial'
            },
    }
    
    # Setup Model    
    model.to(device)
    model.eval()

    #total_batches = len(dataloader)
    image_computed = 0
    img_height, img_width  =dataloader.dataset[0][0].shape[1], dataloader.dataset[0][0].shape[2]
    for i, (images, orig_images, labels, _) in enumerate(dataloader):          

        p_mask = []

        batch_size = images.shape[0]
        image_computed += batch_size
        images = images.to(device)
        labels = labels.to(device)
        
        for _ in range(batch_size):
            mask = torch.zeros((img_height, img_width), device=device)

            if attack_position == 'center':
                center_y, center_x = img_height // 2, img_width // 2
                start_y = max(center_y - attack_size[0] // 2, 0)
                start_x = max(center_x - attack_size[1] // 2, 0)
            elif attack_position == 'left_corner':
                start_y, start_x = 0, 0
            elif attack_position == 'random':
                start_y = torch.randint(0, max(1, img_height - attack_size[0] + 1), (1,)).item()
                start_x = torch.randint(0, max(1, img_width - attack_size[1] + 1), (1,)).item()
            else:
                raise ValueError(f"Invalid attack_position: {attack_position}")

            end_y = min(start_y + attack_size[0], img_height)
            end_x = min(start_x + attack_size[1], img_width)
            mask[start_y:end_y, start_x:end_x] = 1
            p_mask.append(mask)
        p_mask = torch.stack(p_mask).to(device)

        # this mask take trace of what is mispredicted with respect to the original prediction
        correct_masks = torch.zeros((batch_size, img_height, img_width), device=device)
        cumulative_outputs = torch.zeros((batch_size, n_classes, img_height, img_width), device=device)

        with torch.no_grad():
            original_outputs = model(images)  # Shape: (B, C, H, W)
            original_outputs = models_utils.output_handle(images, original_outputs, model_name)
            original_predicted_classes = original_outputs.argmax(dim=1)  # Shape: (B, H, W)

        # Compute the correct mask
        # at the beggining it is the entire image less the wrong parts and the attacked region 
        ignore_index = 250
        correct_masks = (original_predicted_classes == labels) * (1 - p_mask) * (labels != ignore_index)
        attack_mask =  correct_masks.float()
        cumulative_outputs = original_predicted_classes

        
        for iteration in range(max_iterations):

            # the attack mask is the area targeted by the attack and correspond to those area never attacked before.
            attack_mask = attack_mask * (correct_masks.float())      
        
            attacked_image = attack_image(model = model, 
                                          image = images,
                                          ground_truth = labels, 
                                          binary_mask = p_mask, 
                                          attack_type = attack_type, 
                                          attacked_area = attack_mask, 
                                          mean = mean, 
                                          std = std, 
                                          epsilon = epsilon, 
                                          alpha = alpha, 
                                          num_steps = num_steps, 
                                          model_name = model_name)
            
            attacked_image = attacked_image.to(device)
            with torch.no_grad():
                outputs = model(attacked_image)
                outputs = models_utils.output_handle(images, outputs, model_name)
                pred = outputs.data.max(1)[1]

            # compute the wrong predicted part
            current_correct_masks = (pred == labels) * (1 - p_mask) * (labels != ignore_index)
            current_diff_masks = (1- current_correct_masks) * correct_masks
            correct_masks = current_correct_masks * correct_masks

            # update the current mask and the cumulative prediction here.
            cumulative_outputs = cumulative_outputs * (1 - current_diff_masks) + pred * current_diff_masks

        if isinstance(labels, list):
            labels = labels[0]
        gt = labels.cpu().numpy()

        cumulative_outputs = cumulative_outputs.long().cpu().numpy()

        for types_metrics in scores.keys():
            spatial_importance = []
            p_mask = p_mask.to(device)

            if scores[types_metrics]['spatial_mode'] is None:
                spatial_importance = torch.ones_like(p_mask)
            elif scores[types_metrics]['spatial_mode'] == 'spatial':
                spatial_importance = (1-p_mask)
            elif scores[types_metrics]['spatial_mode'] == 'local':
                spatial_importance = p_mask
            scores[types_metrics]['corrupted'].update(gt, cumulative_outputs, (spatial_importance).view(images.shape[0],-1).cpu().numpy())

        #print(attacked_image.mean())


        #---------------------------------
        orig_images = orig_images.to(device)
        outputs = model(orig_images)
        outputs = models_utils.output_handle(orig_images, outputs, model_name)
        pred = outputs.data.max(1)[1].cpu().numpy()


        #running_metrics_original.update(gt, pred, (spatial_importance).view(orig_images.shape[0],-1).cpu().numpy())

        for types_metrics in scores.keys():
            spatial_importance = []
            p_mask = p_mask.to(device)

            if scores[types_metrics]['spatial_mode'] is None:
                spatial_importance = torch.ones_like(p_mask)
            elif scores[types_metrics]['spatial_mode'] == 'spatial':
                spatial_importance = (1-p_mask)
            elif scores[types_metrics]['spatial_mode'] == 'local':
                spatial_importance = p_mask
            scores[types_metrics]['clean'].update(gt, pred, (spatial_importance).view(images.shape[0],-1).cpu().numpy())


        if test_mode and i > -1:
            break

        if image_computed % 1 == 0:
            print("batch completed: " + str(image_computed) + "/" + str(len(dataloader.dataset)))
            
    
    return scores



#-------------------------------------------------
# Attack Analysis
#-------------------------------------------------
def analysis_attacks(model, images, labels, device, num_classes = 19, fooling_class=None,
                 attack_type=None, mean=None, std=None, epsilon=0.03, alpha=0.01, attack_position='center', attack_size=(50, 50), num_steps=1, model_name='bisenetX39', max_iterations=1):

    from attacks import normalize, unnormalize
    n_classes = num_classes

    orig_images = images.clone()

    scores = {
        'normal': {
            'corrupted': runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': None
            },

         'local': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'local'
            },

         'spatial': {
            'corrupted':  runningScoreSpatial(n_classes), 
            'clean':  runningScoreSpatial(n_classes),
            'spatial_mode': 'spatial'
            },
    }
    
    # Setup Model    
    model.to(device)
    model.eval()

    #total_batches = len(dataloader)
    img_height, img_width  =images.shape[2],images.shape[3]         

    p_mask = []

    batch_size = images.shape[0]
    images = images.to(device)
    labels = labels.to(device)

    for _ in range(batch_size):
        mask = torch.zeros((img_height, img_width), device=device)

        if attack_position == 'center':
            center_y, center_x = img_height // 2, img_width // 2
            start_y = max(center_y - attack_size[0] // 2, 0)
            start_x = max(center_x - attack_size[1] // 2, 0)
        elif attack_position == 'left_corner':
            start_y, start_x = 0, 0
        elif attack_position == 'random':
            start_y = torch.randint(0, max(1, img_height - attack_size[0] + 1), (1,)).item()
            start_x = torch.randint(0, max(1, img_width - attack_size[1] + 1), (1,)).item()
        else:
            raise ValueError(f"Invalid attack_position: {attack_position}")

        end_y = min(start_y + attack_size[0], img_height)
        end_x = min(start_x + attack_size[1], img_width)
        mask[start_y:end_y, start_x:end_x] = 1
        p_mask.append(mask)
    p_mask = torch.stack(p_mask).to(device)

    # this mask take trace of what is mispredicted with respect to the original prediction
    correct_masks = torch.zeros((batch_size, img_height, img_width), device=device)
    cumulative_outputs = torch.zeros((batch_size, n_classes, img_height, img_width), device=device)

    with torch.no_grad():
        original_outputs = model(images)  # Shape: (B, C, H, W)
        original_outputs = models_utils.output_handle(images, original_outputs, model_name)
        original_predicted_classes = original_outputs.argmax(dim=1)  # Shape: (B, H, W)

    # Compute the correct mask
    # at the beggining it is the entire image less the wrong parts and the attacked region 
    ignore_index = 250
    correct_masks = (original_predicted_classes == labels) * (1 - p_mask) * (labels != ignore_index) 
    print(correct_masks.sum())
    attack_mask =  correct_masks.float()
    cumulative_outputs = original_predicted_classes

    if max_iterations > 1:
        attacked_images = []
        d_attacked_mask = []
        d_attacked_images = []

        
    for iteration in range(max_iterations):

        # the attack mask is the area targeted by the attack and correspond to those area never attacked before.
        attack_mask = attack_mask * (correct_masks.float())     

        if fooling_class is not None:
            attack_mask = attack_mask * (labels == fooling_class).float()
            print(attack_mask.sum())
    
        attacked_image = attack_image(model = model, 
                                      image = images,
                                      ground_truth = labels, 
                                      binary_mask = p_mask, 
                                      attack_type = attack_type, 
                                      attacked_area = attack_mask, 
                                      mean = mean, 
                                      std = std, 
                                      epsilon = epsilon, 
                                      alpha = alpha, 
                                      num_steps = num_steps, 
                                      model_name = model_name)
        
        attacked_image = attacked_image.to(device)
        with torch.no_grad():
            outputs = model(attacked_image)
            outputs = models_utils.output_handle(images, outputs, model_name)
            pred = outputs.data.max(1)[1]

        if max_iterations > 1:
            attacked_images.append(attacked_image)

        # compute the wrong predicted part
        current_correct_masks = (pred == labels) * (1 - p_mask) * (labels != ignore_index)
        current_diff_masks = (1- current_correct_masks) * correct_masks
        if fooling_class is not None:
            current_diff_masks = current_diff_masks.float()
        if max_iterations > 1:
            d_attacked_images.append((1- current_correct_masks)  * (correct_masks) * (labels == fooling_class))
            d_attacked_mask.append(attack_mask)
        print("diff " + str(current_diff_masks.sum().item()))

        correct_masks = current_correct_masks * correct_masks
        print(correct_masks.sum())
        print()

        # update the current mask and the cumulative prediction here.
        cumulative_outputs = cumulative_outputs * (1 - current_diff_masks) + pred * current_diff_masks

    if isinstance(labels, list):
        labels = labels[0]
    gt = labels.cpu().numpy()

    cumulative_outputs = cumulative_outputs.long().cpu().numpy()

    for types_metrics in scores.keys():
        spatial_importance = []
        p_mask = p_mask.to(device)

        if scores[types_metrics]['spatial_mode'] is None:
            spatial_importance = torch.ones_like(p_mask)
        elif scores[types_metrics]['spatial_mode'] == 'spatial':
            spatial_importance = (1-p_mask)
        elif scores[types_metrics]['spatial_mode'] == 'local':
            spatial_importance = p_mask
        scores[types_metrics]['corrupted'].update(gt, cumulative_outputs, (spatial_importance).view(images.shape[0],-1).cpu().numpy())

    #print(attacked_image.mean())


    #---------------------------------
    orig_images = orig_images.to(device)
    outputs = model(orig_images)
    outputs = models_utils.output_handle(orig_images, outputs, model_name)
    pred = outputs.data.max(1)[1].cpu().numpy()


    #running_metrics_original.update(gt, pred, (spatial_importance).view(orig_images.shape[0],-1).cpu().numpy())

    for types_metrics in scores.keys():
        spatial_importance = []
        p_mask = p_mask.to(device)

        if scores[types_metrics]['spatial_mode'] is None:
            spatial_importance = torch.ones_like(p_mask)
        elif scores[types_metrics]['spatial_mode'] == 'spatial':
            spatial_importance = (1-p_mask)
        elif scores[types_metrics]['spatial_mode'] == 'local':
            spatial_importance = p_mask
        scores[types_metrics]['clean'].update(gt, pred, (spatial_importance).view(images.shape[0],-1).cpu().numpy())


            

    # Calculate final scores
    #score, class_iou = running_metrics.get_scores()
    #for k, v in score.items():
        #print(k, v)
    #for i in range(n_classes):
        #print(i, class_iou[i])
    if max_iterations > 1:
        return scores, orig_images, attacked_images, d_attacked_images, d_attacked_mask, cumulative_outputs, pred
    else:    
        return scores, orig_images, attacked_image,cumulative_outputs, pred