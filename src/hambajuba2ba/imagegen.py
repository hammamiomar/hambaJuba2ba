from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision

def get_noise_img(shape:int) -> Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_tensor = torch.rand(3,shape,shape,device=device)
    image = torchvision.transforms.ToPILImage()(image_tensor)

    return image
