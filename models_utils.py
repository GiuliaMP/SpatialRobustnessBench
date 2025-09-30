import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

from transformers import (
    SegformerFeatureExtractor, SegformerForSemanticSegmentation,
    Mask2FormerImageProcessor,  Mask2FormerForUniversalSegmentation,
    OneFormerImageProcessor,  OneFormerForUniversalSegmentation
)

from ptsemseg.loader import get_loader
from ptsemseg.loader import get_loader
from ptsemseg.models import get_model
from ptsemseg.utils import get_model_state

from torch import nn


def input_handle(image, model_name):
    if model_name == 'oneformer_large':
        from transformers import OneFormerProcessor
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
        image = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to('cuda')
    return image
    
def output_handle(image, outputs, model_name):
    if model_name == 'mask2former_large':
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
        outputs = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.shape[2], image.shape[3]) for a in range(image.shape[0])])
        outputs = torch.cat(outputs).view(image.shape[0], image.shape[2], image.shape[3]).unsqueeze(1)

    elif model_name == 'segformer_bo':
        outputs = outputs.logits
        outputs = torch.nn.functional.interpolate(
                outputs,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False
            )

    elif model_name == 'segformer_b1':
        outputs = outputs.logits
        outputs = torch.nn.functional.interpolate(
                outputs,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False
            )

    elif model_name == 'segformer_b5':
        outputs = outputs.logits
        outputs = torch.nn.functional.interpolate(
                outputs,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False
            )

    elif 'pidnet' in model_name:
        outputs = outputs[1]
        outputs = torch.nn.functional.interpolate(
                outputs,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False
            )

    elif 'deeplab' in model_name:
        outputs = outputs
        outputs = torch.nn.functional.interpolate(
                outputs,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False
            )
        
    return outputs


# Define the normalization block
import torch
import torch.nn as nn

class ChangeNormalize(nn.Module):
    def __init__(self, original_mean, original_std, new_mean, new_std):
        super(ChangeNormalize, self).__init__()
        # Convert lists to tensors for broadcasting
        self.original_mean = torch.tensor(original_mean).view(1, -1, 1, 1)
        self.original_std = torch.tensor(original_std).view(1, -1, 1, 1)
        self.new_mean = torch.tensor(new_mean).view(1, -1, 1, 1)
        self.new_std = torch.tensor(new_std).view(1, -1, 1, 1)

    def forward(self, x):
        # Ensure tensors are on the same device as input x
        self.original_mean = self.original_mean.to(x.device)
        self.original_std = self.original_std.to(x.device)
        self.new_mean = self.new_mean.to(x.device)
        self.new_std = self.new_std.to(x.device)

        # Denormalize to original mean and std
        x = x * self.original_std + self.original_mean

        # Convert RGB to BGR by reversing the channel order
        x = x[:, [2, 1, 0], :, :]

        # Scale to 0-255 range (if required, adjust based on your use case)
        x *= 255.

        # Normalize to new mean and std
        x = (x - self.new_mean) / self.new_std
        return x


def load_model_with_weights(name):

    model = None    
    if name == 'mask2former_large':
        feature_extractor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-cityscapes-semantic"
        )
        
    elif name == 'oneformer_large':
        feature_extractor = OneFormerImageProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
        model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_cityscapes_swin_large"
        )

    elif name == 'segformer_bo':
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )

    elif name == 'segformer_b1':
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        )

    elif name == 'segformer_b5':
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        )

    elif name == 'bisenetX39':
        model_dict = {'arch':'bisenetX39'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/bisenetX39_cityscapes.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

    elif name == 'bisenetR18':
        model_dict = {'arch':'bisenetR18'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/bisenetR18_cityscapes.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

    elif name == 'bisenetR101':
        model_dict = {'arch':'bisenetR101'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/bisenetR101_cityscapes.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

    elif name == 'ddrnet23': 
        model_dict = {'arch':'ddrnet23'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/ddrnet23_cityscapes.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

    elif name == 'ddrnet23Slim': 
        model_dict = {'arch':'ddrnet23Slim'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/ddrnet23Slim_cityscapes.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)


    
    elif name == 'icnet':
        model_dict = {'arch':'icnetBN'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/icnetBN_cityscapes_trainval_90k.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)
        
        original_mean = [0.485, 0.456, 0.40]  # Example: ImageNet means
        original_std = [0.229, 0.224, 0.225]   # Example: ImageNet stds
        new_mean = [103.939, 116.779, 123.68]            # Example: New dataset means
        new_std = [1., 1., 1.]             # Example: New dataset stds

        normalization_block = ChangeNormalize(original_mean, original_std, new_mean, new_std)
        # Your original model
        
        # Create the final sequential model
        model = nn.Sequential(
            normalization_block,
            model
        )
                

        
    elif name == 'pspnet':
        model_dict = {'arch': 'pspnet'}
        n_classes = 19
        path = '/home/g.rossolini/pretrained/pspnet_101_cityscapes.pth'
        model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

        original_mean = [0.485, 0.456, 0.40]  # Example: ImageNet means
        original_std = [0.229, 0.224, 0.225]   # Example: ImageNet stds
        new_mean = [103.939, 116.779, 123.68]            # Example: New dataset means
        new_std = [1., 1., 1.]             # Example: New dataset stds

        normalization_block = ChangeNormalize(original_mean, original_std, new_mean, new_std)
        # Your original model
        
        # Create the final sequential model
        model = nn.Sequential(
            normalization_block,
            model
        )

    elif name == 'deeplabv3_mobilenet':
        import DeepLabV3PlusPytorch
        from DeepLabV3PlusPytorch import network
        model_name = 'deeplabv3plus_mobilenet'
        num_classes = 19
        outputs_stride = 16
        separable_conv = False
        model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=outputs_stride)
        if separable_conv and 'plus' in model_name:
            network.convert_to_separable_conv(model.classifier)

        path = '/home/g.rossolini/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
        #model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

    elif name == 'deeplabv3_resnet':
        import DeepLabV3PlusPytorch
        from DeepLabV3PlusPytorch import network
        model_name = 'deeplabv3plus_resnet101'
        num_classes = 19
        outputs_stride = 16
        separable_conv = False
        model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=outputs_stride)
        if separable_conv and 'plus' in model_name:
            network.convert_to_separable_conv(model.classifier)
            
        path = '/home/g.rossolini/pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth'
        #model = get_model(model_dict, n_classes, version="citscyapes")
        state = torch.load(path, map_location = 'cpu')
        state = get_model_state(state, name)
        model.load_state_dict(state, strict=False)

    
    elif name == 'pidnet_s':
        from PIDNet.models import pidnet
        #config = 'configs/cityscapes/pidnet_small_cityscapes.yaml'
        model = pidnet.get_seg_model(model_name='pidnet_s', num_classes=19)
        
        path = '/home/g.rossolini/pretrained/PIDNet_S_Cityscapes_val.pt'
        pretrained_dict = torch.load(path)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    elif name == 'pidnet_m':
        from PIDNet.models import pidnet
        #config = 'configs/cityscapes/pidnet_small_cityscapes.yaml'
        model = pidnet.get_seg_model(model_name='pidnet_m', num_classes=19)

        path = '/home/g.rossolini/pretrained/PIDNet_M_Cityscapes_val.pt'
        pretrained_dict = torch.load(path)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    elif name == 'pidnet_l':
        from PIDNet.models import pidnet
        #config = 'configs/cityscapes/pidnet_small_cityscapes.yaml'
        model = pidnet.get_seg_model(model_name='pidnet_l', num_classes=19)

        path = '/home/g.rossolini/pretrained/PIDNet_L_Cityscapes_val.pt'
        pretrained_dict = torch.load(path)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        


    

    model = model.eval()
    return model, name