import os
#import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class TestImageFolder(data.Dataset):
    def __init__(self, root,image_size=224):
        """Initializes image paths and preprocessing module."""
        self.root = root
        
        self.image_paths = list(map(lambda x: os.path.join(root + "images/", x), os.listdir(root + "images/")))
        self.image_size = image_size
        print("image count in path :{}".format(len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1]
        image = Image.open(image_path)
        
        aspect_ratio = image.size[1]/image.size[0]

        Transform = []

        #ResizeRange = random.randint(300,320)
        #Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))


        Transform.append(T.Resize((int(self.image_size*aspect_ratio)-int(self.image_size*aspect_ratio)%16,self.image_size)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        return image, filename

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_testloader(image_path, image_size, batch_size, num_workers=2):
    """Builds and returns Dataloader."""
    dataset = TestImageFolder(root = image_path, image_size = image_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader
