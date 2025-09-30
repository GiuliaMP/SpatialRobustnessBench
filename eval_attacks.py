import yaml
import sys 
sys.path.append('..')

from ptsemseg.loader import get_loader
from ptsemseg.loader import get_loader
from ptsemseg.models import get_model
from ptsemseg.utils import get_model_state
import matplotlib.pyplot as plt
import matplotlib
import patch_utils as patch_utils
from torch.utils import data
import numpy as np
from torch import nn
from PIL import Image
import torch
import os
import pickle
from eval import eval_attacks_ripetitive
import random
import models_utils

print('torch version:       {}'.format(torch.__version__))
print('numpy version:       {}'.format(np.__version__))
print('matplotlib version:  {}'.format(matplotlib.__version__))

use_cuda = torch.cuda.is_available()
print('CUDA available:      {}'.format(use_cuda))
print('cuDNN enabled:       {}'.format(torch.backends.cudnn.enabled))
print('num gpus:            {}'.format(torch.cuda.device_count()))



BATCH_SIZE = 4 


def set_loader(cfg, augmentation, batch_size=1):
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    train_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"],
        std_version = cfg["data"]["std_version"],
        augmentation_params = augmentation
        )

    validation_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"], 
        std_version = cfg["data"]["std_version"],
        bottom_crop = 0,
        augmentation_params = augmentation
        )
    
    
    n_classes = train_loader.n_classes    
    validationloader = data.DataLoader(
        validation_loader, 
        batch_size=batch_size, 
        num_workers=cfg["device"]["n_workers"],
        shuffle=False
    )

    return validationloader, n_classes

def reset_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)



import sys
import yaml
import torch
import json
from models_utils import load_model_with_weights  # Assuming this module exists

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python script.py <config_path_loader> <config_path_test>")
        sys.exit(1)

    config_path_loader = args[0]
    config_path_test = args[1]


    # Load configurations
    with open(config_path_loader, "r") as file:
        cfg = yaml.safe_load(file)

    with open(config_path_test, "r") as file:
        configs_test = yaml.safe_load(file)

    device = torch.device("cuda")
    torch.cuda.set_device(configs_test["device"]["gpu"])

    results = {}

    for config in configs_test['attack_configurations']:
        config_name = config['name']
        augmentation_config = None #config['augmentation']
        attack_config = config['settings']

        epsilon = attack_config['epsilon']
        alpha = attack_config['alpha']
        print(alpha)
        max_iterations = attack_config['max_iterations']
        num_steps = attack_config['num_steps']
        attack_size = attack_config['patch_size']
        attack_position = attack_config['attack_position']
        mean = [0.485, 0.456, 0.40]  # Example: ImageNet means
        std = [0.229, 0.224, 0.225]   # Example: ImageNet stds
        attack_type = 'pgd_distance_based'

        validationloader, n_classes = set_loader(cfg, augmentation_config, batch_size=BATCH_SIZE)
        
        evaluation_score = {}

        for i, name in enumerate(configs_test['models_name']): 
            model, name = load_model_with_weights(name)
            model = model.to('cuda')
            reset_seed(42)  # Resets the seed for reproducibility
            result = 
            (
                model, 
                validationloader, 
                device, 
                test_mode = False, 
                attack_type=attack_type, 
                mean=mean, 
                std=std, 
                epsilon=epsilon, 
                alpha=alpha, 
                attack_position=attack_position, 
                attack_size=attack_size, 
                num_steps=num_steps,  
                model_name = name, 
                max_iterations = max_iterations
            )
            #result = validate_spatial(model, validationloader, device, test_mode=True, model_name=name)
            evaluation_score[name] = result
            del model

        # Save results
        results[config_name] = {
            "config": attack_config,
            "evaluation_score": evaluation_score
        }

    # Save results to JSON
    with open("evaluation_results"+configs_test["test_name"]+".pkl", "wb") as result_file:
        pickle.dump(results, result_file)

    print("Evaluations completed and results saved.")

    print(results)

if __name__ == "__main__":
    main()
